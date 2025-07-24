#!/usr/bin/env python3
"""
Main entry point for Railway deployment
This file allows Railway to easily detect and run the Flask application
"""

import os
import sys

# Add the webapp directory to the Python path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'webapp'))

# Change to the webapp directory
os.chdir(os.path.join(os.path.dirname(__file__), 'webapp'))

# Import and run the Flask app
from app import app

if __name__ == '__main__':
    port = int(os.environ.get('PORT', 5000))
    app.run(host='0.0.0.0', port=port) 