#!/usr/bin/env python3
"""
Flask application entry point for Railway deployment
"""

import os
import sys

# Add webapp directory to Python path
webapp_path = os.path.join(os.path.dirname(__file__), 'webapp')
sys.path.insert(0, webapp_path)

# Import the Flask app from webapp directory
from webapp.app import app

if __name__ == '__main__':
    port = int(os.environ.get('PORT', 5000))
    app.run(host='0.0.0.0', port=port, debug=False) 