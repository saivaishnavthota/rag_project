#!/bin/bash

# RAG Server Startup Script

echo "========================================="
echo "Local RAG Server with Qwen 2.5 7B"
echo "========================================="

# Check if virtual environment exists
if [ ! -d "venv" ]; then
    echo "Creating virtual environment..."
    python -m venv venv
fi

# Activate virtual environment
source venv/bin/activate

# Install requirements
echo "Installing dependencies..."
pip install -r requirements.txt

# Run the server
echo "Starting server on http://0.0.0.0:8000"
python main.py
