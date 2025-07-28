# Adobe Hackathon Problem Statement 1A - PDF Document Outline Extractor

# Use official Python runtime as parent image (AMD64 compatible)
FROM --platform=linux/amd64 python:3.9-slim

WORKDIR /app

# Install system dependencies required for PyMuPDF and PDF processing
RUN apt-get update && apt-get install -y \
    gcc \
    g++ \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements first for better Docker layer caching
COPY requirements.txt .

# Install Python dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Copy all source files from the adobe directory
COPY main.py .
COPY pdf_analyzer.py .
COPY heading_classifier.pkl .

# Create input and output directories
RUN mkdir -p /app/input /app/output

# Make the main script executable
RUN chmod +x main.py

# Set the default command to run the main script
CMD ["python", "main.py"]