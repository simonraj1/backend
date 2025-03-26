#!/usr/bin/env python3
"""
PDF to Questions

A streamlined script that takes a PDF and directly produces an Excel file with improved questions.
This script:
1. Converts PDF pages to images
2. Uses Gemini's vision capabilities to extract text from each page
3. Processes the extracted text to identify multiple-choice questions
4. Improves the extracted questions for clarity and formatting
5. Saves the questions to an Excel file
"""

import os
import re
import time
import json
import logging
import argparse
import tempfile
from pathlib import Path
from typing import List, Dict, Any, Optional

# PDF and image processing
from PIL import Image
from pdf2image import convert_from_path

# Gemini API
from dotenv import load_dotenv
import google.generativeai as genai

# Data handling
import pandas as pd

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S'
)

# Load environment variables
load_dotenv()

# Initialize Gemini API
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")
if not GEMINI_API_KEY:
    logging.warning("GEMINI_API_KEY not found in environment variables. Please provide it as an argument.")

# Hardcoded DPI value
DPI = 300

def convert_pdf_to_images(pdf_path: str, start_page: int = 1, max_pages: Optional[int] = None, 
                          temp_dir: str = "temp_images") -> List[str]:
    """
    Convert PDF pages to images.
    
    Args:
        pdf_path: Path to the PDF file
        start_page: Page to start from (1-indexed)
        max_pages: Maximum number of pages to process
        temp_dir: Directory to save temporary images
    
    Returns:
        List of paths to the generated images
    """
    try:
        # Create temp directory if it doesn't exist
        os.makedirs(temp_dir, exist_ok=True)
        
        # Adjust start_page for pdf2image (0-indexed)
        pdf_start_page = start_page - 1
        
        # Calculate end page
        if max_pages is not None:
            pdf_end_page = pdf_start_page + max_pages - 1
        else:
            pdf_end_page = None
        
        logging.info(f"Converting PDF to images with DPI={DPI}")
        logging.info(f"Processing pages {start_page} to {pdf_end_page + 1 if pdf_end_page is not None else 'end'}")
        
        # Open the PDF file to check if it exists and is valid
        try:
            images = convert_from_path(
                pdf_path,
                dpi=DPI,
                first_page=pdf_start_page + 1,  # pdf2image uses 1-indexed pages
                last_page=pdf_end_page + 1 if pdf_end_page is not None else None,
                fmt="jpeg",
                output_folder=temp_dir,
                paths_only=True,
                output_file=f"page_{start_page}"
            )
            logging.info(f"Successfully opened PDF and converted {len(images)} pages to images")
        except Exception as e:
            logging.error(f"Failed to open PDF file: {str(e)}")
            return []
        
        # Rename files to include page numbers
        image_paths = []
        for i, img_path in enumerate(images):
            page_num = start_page + i
            new_path = os.path.join(temp_dir, f"page_{page_num}.jpg")
            
            # If the file already exists with the correct name, no need to rename
            if img_path != new_path and os.path.exists(img_path):
                try:
                    os.rename(img_path, new_path)
                    image_paths.append(new_path)
                except Exception as e:
                    logging.error(f"Failed to rename {img_path} to {new_path}: {str(e)}")
                    image_paths.append(img_path)  # Use original path if rename fails
            else:
                image_paths.append(img_path)
        
        logging.info(f"Converted {len(image_paths)} pages to images")
        return image_paths
    except Exception as e:
        logging.error(f"Error converting PDF to images: {str(e)}")
        return []

def extract_text_with_gemini(image_path: str, api_key: str, retry_count: int = 3, delay: int = 5) -> str:
    """
    Extract text from an image using Gemini's vision capabilities.
    
    Args:
        image_path: Path to the image
        api_key: Gemini API key
        retry_count: Number of retry attempts
        delay: Delay between retries in seconds
    
    Returns:
        Extracted text
    """
    # Configure the Gemini API
    genai.configure(api_key=api_key)
    
    # Get the page number from the image path
    page_num = int(re.search(r'page_(\d+)\.jpg', image_path).group(1))
    
    # Load the image
    try:
        img = Image.open(image_path)
        width, height = img.size
        logging.info(f"Processing image for page {page_num} ({width}x{height})")
    except Exception as e:
        logging.error(f"Failed to open image {image_path}: {str(e)}")
        return ""
    
    # Set up the model
    generation_config = {
        "temperature": 0.1,
        "top_p": 0.95,
        "top_k": 64,
        "max_output_tokens": 8192,
    }
    
    model = genai.GenerativeModel(
        model_name="gemini-1.5-flash",
        generation_config=generation_config
    )
    
    # Create a system prompt that emphasizes maintaining formatting and capturing multiple-choice questions
    system_prompt = """
    You are an expert OCR system. Extract ALL text from the image, maintaining the original formatting as much as possible.
    Pay special attention to multiple-choice questions, their options (A, B, C, D), and any explanations or answers.
    Preserve question numbers, bullet points, and indentation.
    If there are tables, maintain their structure.
    If there are any special characters or symbols, represent them accurately.
    Do not summarize or paraphrase - extract the complete text exactly as it appears.
    """
    
    # Set up the prompt parts
    prompt_parts = [
        system_prompt,
        Image.open(image_path),
    ]
    
    # Try to extract text with retries
    for attempt in range(1, retry_count + 1):
        try:
            logging.info(f"Attempt {attempt} to extract text from page {page_num}")
            response = model.generate_content(prompt_parts)
            
            if response.text:
                logging.info(f"Successfully extracted text from page {page_num}")
                return response.text
            else:
                logging.warning(f"Empty response from Gemini for page {page_num} on attempt {attempt}")
        except Exception as e:
            logging.error(f"Error extracting text from page {page_num} on attempt {attempt}: {str(e)}")
        
        # Wait before retrying
        if attempt < retry_count:
            logging.info(f"Waiting {delay} seconds before retry")
            time.sleep(delay)
    
    logging.error(f"Failed to extract text with Gemini after {retry_count} attempts")
    return ""

def extract_questions_with_gemini(text: str, api_key: str, retry_count: int = 3, delay: int = 5) -> List[Dict[str, str]]:
    """
    Extract multiple-choice questions from text using Gemini AI.
    
    Args:
        text: Text to extract questions from
        api_key: Gemini API key
        retry_count: Number of retry attempts
        delay: Delay between retries in seconds
    
    Returns:
        List of extracted questions as dictionaries
    """
    # Configure the Gemini API
    genai.configure(api_key=api_key)
    
    # Set up the model
    generation_config = {
        "temperature": 0.1,
        "top_p": 0.95,
        "top_k": 64,
        "max_output_tokens": 8192,
    }
    
    model = genai.GenerativeModel(
        model_name="gemini-1.5-flash",
        generation_config=generation_config
    )
    
    # Create a system prompt for question extraction
    system_prompt = """
    Extract all multiple-choice questions from the provided text. Be thorough and extract every question, even if it's partially formatted.
    
    For each question, extract:
    1. Question number (if available)
    2. Question text
    3. Options A, B, C, D (if available)
    4. Correct answer (if indicated)
    5. Explanation or answer text (if available)
    
    Format your response as a JSON array of objects with these fields:
    - question_number: The question number (string, can be empty)
    - question: The full question text
    - option_a: Text for option A
    - option_b: Text for option B
    - option_c: Text for option C
    - option_d: Text for option D
    - correct_answer: The letter of the correct answer (A, B, C, or D)
    - answer_text: The text of the correct answer
    - explanation: Any explanation provided for the answer
    
    If any field is not available in the text, include it with an empty string value.
    Only extract complete questions that have at least the question text and some options.
    
    Look for patterns like:
    - Numbered questions followed by options labeled A, B, C, D
    - Questions may span multiple paragraphs
    - Answers may be indicated by "Answer: X" or similar
    - Explanations often follow the answer
    
    Be aware that OCR text might have formatting issues, so be flexible in your extraction.
    """
    
    # Try to extract questions with retries
    for attempt in range(1, retry_count + 1):
        try:
            logging.info(f"Attempt {attempt} to send text to Gemini for question extraction")
            response = model.generate_content([system_prompt, text])
            
            if response.text:
                # Try to parse the response as JSON
                try:
                    # Look for JSON content in the response
                    json_match = re.search(r'```json\s*([\s\S]*?)\s*```', response.text)
                    if json_match:
                        json_content = json_match.group(1)
                    else:
                        json_content = response.text
                    
                    # Clean up the JSON content
                    json_content = json_content.strip()
                    if json_content.startswith('```') and json_content.endswith('```'):
                        json_content = json_content[3:-3].strip()
                    
                    questions = json.loads(json_content)
                    
                    # Ensure the result is a list
                    if not isinstance(questions, list):
                        if isinstance(questions, dict) and any(key in questions for key in ['questions', 'results', 'items']):
                            # Extract the list from the dictionary
                            for key in ['questions', 'results', 'items']:
                                if key in questions:
                                    questions = questions[key]
                                    break
                        else:
                            questions = [questions]
                    
                    logging.info(f"Extracted {len(questions)} questions")
                    return questions
                except json.JSONDecodeError as e:
                    logging.error(f"Failed to parse JSON response: {str(e)}")
            else:
                logging.warning(f"Empty response from Gemini on attempt {attempt}")
        except Exception as e:
            logging.error(f"Error extracting questions on attempt {attempt}: {str(e)}")
        
        # Wait before retrying
        if attempt < retry_count:
            logging.info(f"Waiting {delay} seconds before retry")
            time.sleep(delay)
    
    logging.error(f"Failed to extract questions with Gemini after {retry_count} attempts")
    return []

def improve_questions(questions: List[Dict[str, str]], api_key: str) -> List[Dict[str, str]]:
    """
    Improve the extracted questions using Gemini AI.
    
    Args:
        questions: List of question dictionaries
        api_key: Gemini API key
        
    Returns:
        List of improved question dictionaries
    """
    if not questions:
        logging.warning("No questions to improve")
        return []
        
    # Configure the Gemini API
    genai.configure(api_key=api_key)
    
    logging.info(f"Improving {len(questions)} questions")
    
    # Set up the model
    generation_config = {
        "temperature": 0.2,
        "top_p": 0.95,
        "top_k": 64,
        "max_output_tokens": 8192,
    }
    
    model = genai.GenerativeModel(
        model_name="gemini-1.5-flash",
        generation_config=generation_config
    )
    
    # Create a system prompt for question improvement
    system_prompt = """
    Improve the following multiple-choice questions. For each question:
    
    1. Fix any formatting or grammatical issues
    2. Ensure the question is clear and unambiguous
    3. Make sure all options are properly formatted
    4. Verify that the correct answer is clearly indicated
    5. Enhance the explanation if provided
    
    Return the improved questions in the same JSON format:
    - question_number: The question number (string)
    - question: The full question text
    - option_a: Text for option A
    - option_b: Text for option B
    - option_c: Text for option C
    - option_d: Text for option D
    - correct_answer: The letter of the correct answer (A, B, C, or D)
    - answer_text: The text of the correct answer
    - explanation: Any explanation provided for the answer
    
    Maintain the original meaning and intent of each question.
    """
    
    try:
        # Convert questions to JSON string
        questions_json = json.dumps(questions, indent=2)
        
        # Send to Gemini for improvement
        logging.info("Sending questions to Gemini for improvement")
        response = model.generate_content([system_prompt, questions_json])
        
        if response.text:
            # Try to parse the response as JSON
            try:
                # Look for JSON content in the response
                json_match = re.search(r'```json\s*([\s\S]*?)\s*```', response.text)
                if json_match:
                    json_content = json_match.group(1)
                else:
                    json_content = response.text
                
                # Clean up the JSON content
                json_content = json_content.strip()
                if json_content.startswith('```') and json_content.endswith('```'):
                    json_content = json_content[3:-3].strip()
                
                improved_questions = json.loads(json_content)
                
                # Ensure the result is a list
                if not isinstance(improved_questions, list):
                    if isinstance(improved_questions, dict) and any(key in improved_questions for key in ['questions', 'results', 'items']):
                        # Extract the list from the dictionary
                        for key in ['questions', 'results', 'items']:
                            if key in improved_questions:
                                improved_questions = improved_questions[key]
                                break
                    else:
                        improved_questions = [improved_questions]
                
                logging.info(f"Improved {len(improved_questions)} questions")
                return improved_questions
            except json.JSONDecodeError as e:
                logging.error(f"Failed to parse JSON response for question improvement: {str(e)}")
                return questions  # Return original questions if improvement fails
        else:
            logging.warning("Empty response from Gemini for question improvement")
            return questions
    except Exception as e:
        logging.error(f"Error improving questions: {str(e)}")
        return questions

def process_pdf_page(pdf_path: str, page_num: int, api_key: str, 
                    retry_count: int = 3, delay: int = 5, temp_dir: str = "temp_images") -> List[Dict[str, str]]:
    """
    Process a single page of a PDF file.
    
    Args:
        pdf_path: Path to the PDF file
        page_num: Page number to process (1-indexed)
        api_key: Gemini API key
        retry_count: Number of retry attempts
        delay: Delay between retries in seconds
        temp_dir: Directory to save temporary images
    
    Returns:
        List of extracted questions
    """
    try:
        # Convert the specific page to an image
        image_paths = convert_pdf_to_images(
            pdf_path=pdf_path,
            start_page=page_num,
            max_pages=1,
            temp_dir=temp_dir
        )
        
        if not image_paths:
            logging.error(f"Failed to convert page {page_num} to image")
            return []
        
        image_path = image_paths[0]
        
        # Extract text from the image
        extracted_text = extract_text_with_gemini(
            image_path=image_path,
            api_key=api_key,
            retry_count=retry_count,
            delay=delay
        )
        
        if not extracted_text:
            logging.error(f"Failed to extract text from page {page_num}")
            return []
        
        # Extract questions from the text
        questions = extract_questions_with_gemini(
            text=extracted_text,
            api_key=api_key,
            retry_count=retry_count,
            delay=delay
        )
        
        # Add page number to each question
        for q in questions:
            q['page_number'] = page_num
        
        logging.info(f"Extracted {len(questions)} questions from page {page_num}")
        return questions
    except Exception as e:
        logging.error(f"Error processing page {page_num}: {str(e)}")
        return []

def main():
    """Main function to process the PDF and extract questions."""
    parser = argparse.ArgumentParser(description="Extract multiple-choice questions from a PDF file")
    parser.add_argument("pdf_path", help="Path to the PDF file")
    parser.add_argument("--output", default="extracted_questions.xlsx", help="Output Excel file")
    parser.add_argument("--start", type=int, default=1, help="First page to process (1-indexed)")
    parser.add_argument("--max-pages", type=int, help="Maximum number of pages to process")
    parser.add_argument("--api-key", help="Gemini API key (optional, will use .env if provided)")
    parser.add_argument("--retry-count", type=int, default=3, help="Number of retries for API calls")
    parser.add_argument("--delay", type=int, default=10, help="Delay between API calls in seconds")
    parser.add_argument("--debug", action="store_true", help="Enable debug mode")
    parser.add_argument("--keep-temp", action="store_true", help="Keep temporary files")
    
    args = parser.parse_args()
    
    # Set up logging level
    if args.debug:
        logging.getLogger().setLevel(logging.DEBUG)
        logging.debug("Debug mode enabled")
    
    # Use API key from arguments or environment
    api_key = args.api_key or GEMINI_API_KEY
    if not api_key:
        logging.error("No Gemini API key provided. Please set GEMINI_API_KEY in .env or provide --api-key")
        return
    
    # Create a temporary directory for images
    temp_dir = tempfile.mkdtemp(prefix="pdf_questions_")
    logging.info(f"Created temporary directory: {temp_dir}")
    
    try:
        # Convert PDF to images
        image_paths = convert_pdf_to_images(
            pdf_path=args.pdf_path,
            start_page=args.start,
            max_pages=args.max_pages,
            temp_dir=temp_dir
        )
        
        if not image_paths:
            logging.error("Failed to convert PDF to images")
            return
        
        # Process each page
        all_questions = []
        for i, image_path in enumerate(image_paths):
            page_num = args.start + i
            logging.info(f"Processing page {page_num}")
            
            # Extract questions from the page
            questions = process_pdf_page(
                pdf_path=args.pdf_path,
                page_num=page_num,
                api_key=api_key,
                retry_count=args.retry_count,
                delay=args.delay,
                temp_dir=temp_dir
            )
            
            all_questions.extend(questions)
            
            # Add a delay between pages to avoid rate limits
            if i < len(image_paths) - 1:
                logging.info(f"Waiting {args.delay} seconds before processing next page")
                time.sleep(args.delay)
        
        # Improve questions
        if all_questions:
            logging.info(f"Improving {len(all_questions)} extracted questions")
            improved_questions = improve_questions(all_questions, api_key)
            
            # Create a DataFrame from the questions
            df = pd.DataFrame(improved_questions)
            
            # Save to Excel
            df.to_excel(args.output, index=False)
            logging.info(f"Saved {len(improved_questions)} questions to {args.output}")
        else:
            logging.warning("No questions extracted from the PDF")
    
    finally:
        # Clean up temporary files
        if not args.keep_temp:
            try:
                import shutil
                shutil.rmtree(temp_dir)
                logging.info(f"Removed temporary directory: {temp_dir}")
            except Exception as e:
                logging.error(f"Failed to remove temporary directory: {str(e)}")

if __name__ == "__main__":
    main() 