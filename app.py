#!/usr/bin/env python3
"""
YOLOv8 Web Application - Main entry point for Railway deployment
"""

import os
import sys

# Add the webapp directory to the Python path
webapp_path = os.path.join(os.path.dirname(__file__), 'code', 'webapp')
sys.path.insert(0, webapp_path)

# Change to the webapp directory for proper file resolution
os.chdir(webapp_path)

# Import the Flask app from webapp directory
from app import app

if __name__ == '__main__':
    port = int(os.environ.get('PORT', 5000))
    app.run(host='0.0.0.0', port=port, debug=False) 