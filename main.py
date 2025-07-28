#!/usr/bin/env python3
"""
Adobe Hackathon Problem Statement 1A - PDF Document Outline Extractor
Competition-ready main entry point that processes PDFs from /app/input 
and outputs JSON files to /app/output as per hackathon requirements.
"""

import os
import json
import time
import logging
import sys
from pathlib import Path

# Import your existing analyzer
from pdf_analyzer import DocumentProcessor

def setup_logging():
    """Configure logging for competition environment."""
    # Minimal logging for competition to avoid noise
    logging.basicConfig(
        level=logging.WARNING,
        format='%(levelname)s: %(message)s',
        handlers=[logging.StreamHandler(sys.stdout)]
    )

def process_single_pdf(processor, pdf_path, output_dir):
    """Process a single PDF and save the result."""
    pdf_name = Path(pdf_path).stem
    output_file = os.path.join(output_dir, f"{pdf_name}.json")
    
    try:
        start_time = time.time()
        
        # Use your existing process_document method
        result = processor.process_document(pdf_path)
        
        processing_time = time.time() - start_time
        
        if result and 'title' in result and 'outline' in result:
            # Ensure output format matches competition requirements exactly
            formatted_result = {
                "title": result['title'],
                "outline": []
            }
            
            # Format outline according to competition spec
            for heading in result['outline']:
                formatted_heading = {
                    "level": heading['level'],
                    "text": heading['text'],
                    "page": heading['page'] + 1  # Convert from 0-based to 1-based indexing
                }
                formatted_result['outline'].append(formatted_heading)
            
            # Save to JSON file
            with open(output_file, 'w', encoding='utf-8') as f:
                json.dump(formatted_result, f, indent=2, ensure_ascii=False)
            
            print(f"✓ Processed {pdf_name}.pdf in {processing_time:.2f}s -> {len(formatted_result['outline'])} headings")
            return True
            
        else:
            print(f"✗ Failed to extract outline from {pdf_name}.pdf")
            # Create empty result for failed processing
            empty_result = {"title": "Document", "outline": []}
            with open(output_file, 'w', encoding='utf-8') as f:
                json.dump(empty_result, f, indent=2, ensure_ascii=False)
            return False
            
    except Exception as e:
        print(f"✗ Error processing {pdf_name}.pdf: {str(e)}")
        # Create empty result for error cases
        empty_result = {"title": "Document", "outline": []}
        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump(empty_result, f, indent=2, ensure_ascii=False)
        return False

def main():
    """Main entry point for competition execution."""
    setup_logging()
    
    # Competition-specified paths
    input_dir = "/app/input"
    output_dir = "/app/output"
    
    # Verify input directory exists
    if not os.path.exists(input_dir):
        print(f"ERROR: Input directory {input_dir} does not exist")
        sys.exit(1)
    
    # Create output directory
    os.makedirs(output_dir, exist_ok=True)
    
    # Find all PDF files
    pdf_files = []
    for file in os.listdir(input_dir):
        if file.lower().endswith('.pdf'):
            pdf_files.append(file)
    
    if not pdf_files:
        print("No PDF files found in input directory")
        sys.exit(0)
    
    print(f"Processing {len(pdf_files)} PDF file(s)...")
    
    # Initialize the document processor
    processor = DocumentProcessor()
    
    # Process each PDF
    total_start_time = time.time()
    successful_count = 0
    
    for pdf_file in pdf_files:
        pdf_path = os.path.join(input_dir, pdf_file)
        
        if process_single_pdf(processor, pdf_path, output_dir):
            successful_count += 1
    
    total_time = time.time() - total_start_time
    
    print(f"\nCompleted: {successful_count}/{len(pdf_files)} files processed successfully")
    print(f"Total processing time: {total_time:.2f}s")
    
    # Exit successfully
    sys.exit(0)

if __name__ == "__main__":
    main()