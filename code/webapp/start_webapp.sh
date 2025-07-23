#!/bin/bash

echo "Starting YOLO People Counter Web Application..."
echo

# Check if Python is installed
if ! command -v python3 &> /dev/null; then
    echo "Error: Python 3 is not installed or not in PATH"
    exit 1
fi

# Check if we're in the right directory
if [ ! -f "app.py" ]; then
    echo "Error: app.py not found. Please run this script from the webapp directory."
    exit 1
fi

# Install requirements if needed
echo "Installing/updating requirements..."
pip3 install -r requirements.txt

echo
echo "Starting the application..."
echo

# Start the application
python3 start_webapp.py 