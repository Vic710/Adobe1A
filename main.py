#!/usr/bin/env python3
"""
Simple test runner for the PDF analyzer.
"""

import os
import json
import time
from pathlib import Path
from pdf_analyzer import PDFAnalyzer

def main():
    """Test the PDF analyzer with files in input_pdfs directory."""
    input_dir = "/app/input"
    output_dir = "/app/output"
    
    if not os.path.exists(input_dir):
        print(f"Input directory {input_dir} does not exist")
        return
    
    pdf_files = [f for f in os.listdir(input_dir) if f.endswith('.pdf')]
    
    if not pdf_files:
        print("No PDF files found in input_pdfs directory")
        return
    
    print(f"Found {len(pdf_files)} PDF files to test")
    
    analyzer = PDFAnalyzer()
    
    for pdf_file in pdf_files:
        pdf_path = os.path.join(input_dir, pdf_file)
        start_time = time.time()
        
        print(f"\n{'='*60}")
        print(f"Processing: {pdf_file}")
        print(f"{'='*60}")
        
        try:
            result = analyzer.analyze_pdf(pdf_path)
            processing_time = time.time() - start_time
            
            if result:
                print(f"✓ Completed in {processing_time:.2f}s")
                print(f"Title: {result['title']}")
                print(f"Found {len(result['outline'])} headings:")
                
                for i, heading in enumerate(result['outline'], 1):
                    indent = "  " * (int(heading['level'][1]) - 1)
                    print(f"  {i:2d}. {indent}{heading['level']}: {heading['text']} (page {heading['page'] + 1})")
                
                # Save result as JSON
                output_file = Path(output_dir) / f"{os.path.splitext(pdf_file)[0]}.json"
                with open(output_file, 'w', encoding='utf-8') as f:
                    json.dump(result, f, indent=2, ensure_ascii=False)
                print(f"\nSaved result to: {output_file}")
                
            else:
                print("✗ No result returned")
                
        except Exception as e:
            processing_time = time.time() - start_time
            print(f"✗ Error after {processing_time:.2f}s: {str(e)}")

if __name__ == "__main__":
    main()
