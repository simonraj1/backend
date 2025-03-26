#!/usr/bin/env python3
"""
PDF to Questions Web UI

A web interface for the PDF to Questions script that allows users to:
1. Upload a PDF file
2. Configure extraction parameters
3. Download the resulting Excel file with extracted questions
"""

import os
import time
import uuid
import logging
from pathlib import Path
from flask import Flask, render_template, request, redirect, url_for, flash, send_from_directory, jsonify

# Import the question extraction functionality
from pdf_to_questions import convert_pdf_to_images, extract_text_with_gemini, extract_questions_with_gemini
from pdf_to_questions import improve_questions, process_pdf_page

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S'
)

# Initialize Flask app
app = Flask(__name__)
app.secret_key = os.urandom(24)

# Create upload and results directories
UPLOAD_FOLDER = 'uploads'
RESULTS_FOLDER = 'results'
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
os.makedirs(RESULTS_FOLDER, exist_ok=True)

# Load environment variables
from dotenv import load_dotenv
load_dotenv()

# Get API key from environment
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")
if not GEMINI_API_KEY:
    logging.error("GEMINI_API_KEY not found in environment variables. Please set it in .env file.")
    raise ValueError("GEMINI_API_KEY not found. Please set it in .env file.")

# Hardcoded DPI value
DPI = 300

# Track job status
jobs = {}

@app.route('/')
def index():
    """Render the main page"""
    return render_template('index.html')

@app.route('/upload', methods=['POST'])
def upload_file():
    """Handle file upload and start processing"""
    if 'pdf_file' not in request.files:
        flash('No file part')
        return redirect(request.url)
    
    file = request.files['pdf_file']
    if file.filename == '':
        flash('No selected file')
        return redirect(request.url)
    
    if file and file.filename.lower().endswith('.pdf'):
        # Generate a unique ID for this job
        job_id = str(uuid.uuid4())
        
        # Save the uploaded file
        filename = f"{job_id}_{file.filename}"
        file_path = os.path.join(UPLOAD_FOLDER, filename)
        file.save(file_path)
        
        # Get parameters from form
        start_page = int(request.form.get('start_page', 1))
        max_pages = request.form.get('max_pages', '')
        max_pages = int(max_pages) if max_pages.isdigit() else None
        delay = int(request.form.get('delay', 10))
        
        # Create output filename
        output_filename = f"questions_{job_id}.xlsx"
        output_path = os.path.join(RESULTS_FOLDER, output_filename)
        
        # Initialize job status
        jobs[job_id] = {
            'status': 'processing',
            'progress': 0,
            'total_pages': max_pages or 0,
            'questions_extracted': 0,
            'output_file': output_filename,
            'start_time': time.time(),
            'status_message': 'Initializing extraction process...',
            'questions_per_page': {}
        }
        
        # Start processing in a background thread
        import threading
        thread = threading.Thread(
            target=process_pdf,
            args=(file_path, output_path, GEMINI_API_KEY, start_page, max_pages, delay, job_id)
        )
        thread.daemon = True
        thread.start()
        
        return redirect(url_for('job_status', job_id=job_id))
    
    flash('Invalid file type. Please upload a PDF file.')
    return redirect(url_for('index'))

def process_pdf(pdf_path, output_path, api_key, start_page, max_pages, delay, job_id):
    """Process the PDF and extract questions"""
    try:
        # Determine the number of pages to process
        if max_pages is None:
            # Get the total number of pages in the PDF
            try:
                from pdf2image.pdf2image import pdfinfo_from_path
                pdf_info = pdfinfo_from_path(pdf_path)
                total_pages = pdf_info["Pages"]
                max_pages = total_pages - start_page + 1
                
                # Update job status with total pages
                jobs[job_id]['total_pages'] = max_pages
                jobs[job_id]['status_message'] = f"Found {total_pages} pages in PDF. Will process {max_pages} pages."
            except Exception as e:
                logging.error(f"Failed to get PDF info: {str(e)}")
                logging.info("Processing only the first page")
                max_pages = 1
                jobs[job_id]['total_pages'] = 1
                jobs[job_id]['status_message'] = "Failed to get PDF info. Processing only the first page."
        
        logging.info(f"Processing {max_pages} pages starting from page {start_page}")
        
        # Process each page
        all_questions = []
        temp_dir = f"temp_{job_id}"
        os.makedirs(temp_dir, exist_ok=True)
        
        for i, page_num in enumerate(range(start_page, start_page + max_pages)):
            logging.info(f"Processing page {page_num}")
            
            # Update job progress
            jobs[job_id]['progress'] = (i / max_pages) * 100
            jobs[job_id]['current_page'] = page_num
            jobs[job_id]['status_message'] = f"Processing page {page_num} of {start_page + max_pages - 1}..."
            
            # Convert PDF page to image
            jobs[job_id]['status_message'] = f"Converting page {page_num} to image..."
            image_paths = convert_pdf_to_images(
                pdf_path,
                start_page=page_num,
                max_pages=1,
                temp_dir=temp_dir
            )
            
            if not image_paths:
                jobs[job_id]['status_message'] = f"Failed to convert page {page_num} to image."
                logging.error(f"Failed to convert page {page_num} to image")
                continue
            
            image_path = image_paths[0]
            
            # Extract text from image
            jobs[job_id]['status_message'] = f"Extracting text from page {page_num} using Gemini Vision..."
            extracted_text = extract_text_with_gemini(
                image_path,
                api_key,
                retry_count=3,
                delay=delay
            )
            
            if not extracted_text:
                jobs[job_id]['status_message'] = f"Failed to extract text from page {page_num}."
                logging.error(f"Failed to extract text from page {page_num}")
                continue
            
            # Extract questions from text
            jobs[job_id]['status_message'] = f"Analyzing text from page {page_num} to identify questions..."
            questions = extract_questions_with_gemini(
                extracted_text,
                api_key,
                retry_count=3,
                delay=delay
            )
            
            if not questions:
                jobs[job_id]['status_message'] = f"No questions found on page {page_num}."
                logging.warning(f"No questions were extracted from page {page_num}")
                continue
            
            # Add source page to each question
            for question in questions:
                question['source_page'] = page_num
            
            # Improve questions
            jobs[job_id]['status_message'] = f"Found {len(questions)} questions on page {page_num}. Improving questions..."
            improved_questions = improve_questions(questions, api_key)
            
            # Add questions to the list
            all_questions.extend(improved_questions)
            jobs[job_id]['questions_extracted'] = len(all_questions)
            jobs[job_id]['questions_per_page'] = jobs[job_id].get('questions_per_page', {})
            jobs[job_id]['questions_per_page'][str(page_num)] = len(improved_questions)
            jobs[job_id]['status_message'] = f"Extracted and improved {len(improved_questions)} questions from page {page_num}. Total: {len(all_questions)} questions."
            
            # Add delay between pages to avoid rate limits
            if page_num < start_page + max_pages - 1:
                jobs[job_id]['status_message'] = f"Waiting {delay} seconds before processing next page..."
                logging.info(f"Waiting {delay} seconds before processing next page")
                time.sleep(delay)
        
        # Save questions to Excel
        import pandas as pd
        if all_questions:
            jobs[job_id]['status_message'] = f"Saving {len(all_questions)} questions to Excel file..."
            df = pd.DataFrame(all_questions)
            df.to_excel(output_path, index=False)
            logging.info(f"Saved {len(all_questions)} questions to {output_path}")
        else:
            jobs[job_id]['status_message'] = "No questions were extracted. Creating empty output file."
            logging.warning("No questions were extracted. Creating empty output file.")
            pd.DataFrame(columns=["question_number", "question", "option_a", "option_b", "option_c", "option_d", 
                                "correct_answer", "answer_text", "explanation", "source_page"]).to_excel(output_path, index=False)
        
        # Clean up temporary files
        import shutil
        try:
            jobs[job_id]['status_message'] = "Cleaning up temporary files..."
            shutil.rmtree(temp_dir)
            logging.info(f"Removed temporary directory {temp_dir}")
        except Exception as e:
            logging.warning(f"Failed to remove temporary directory {temp_dir}: {str(e)}")
        
        # Update job status
        elapsed_time = time.time() - jobs[job_id]['start_time']
        jobs[job_id].update({
            'status': 'completed',
            'progress': 100,
            'elapsed_time': f"{elapsed_time:.2f}",
            'message': f"Extracted {len(all_questions)} questions from {max_pages} pages in {elapsed_time:.2f} seconds",
            'status_message': "Processing complete! You can now download the Excel file."
        })
        
    except Exception as e:
        logging.error(f"Error processing PDF: {str(e)}")
        import traceback
        logging.error(traceback.format_exc())
        
        # Update job status
        jobs[job_id].update({
            'status': 'failed',
            'progress': 100,
            'message': f"Error: {str(e)}",
            'status_message': f"Error: {str(e)}"
        })

@app.route('/job/<job_id>')
def job_status(job_id):
    """Show job status page"""
    if job_id not in jobs:
        flash('Job not found')
        return redirect(url_for('index'))
    
    return render_template('job_status.html', job_id=job_id, job=jobs[job_id])

@app.route('/api/job/<job_id>')
def api_job_status(job_id):
    """API endpoint for getting job status"""
    if job_id not in jobs:
        return jsonify({'error': 'Job not found'}), 404
    
    return jsonify(jobs[job_id])

@app.route('/download/<filename>')
def download_file(filename):
    """Download the result file"""
    return send_from_directory(RESULTS_FOLDER, filename, as_attachment=True)

if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=5000) 
