#!/usr/bin/env python3
"""
YOLOv8 Web Application - Lightweight version for Railway deployment
"""

import os
import sys
from flask import Flask, render_template, jsonify, request
from flask_cors import CORS

app = Flask(__name__)
CORS(app)

# Add the webapp directory to the Python path for templates
webapp_path = os.path.join(os.path.dirname(__file__), 'code', 'webapp')
sys.path.insert(0, webapp_path)

# Set template folder
app.template_folder = os.path.join(webapp_path, 'templates')

@app.route('/')
def index():
    """Main dashboard page"""
    return render_template('index.html')

@app.route('/api/status')
def status():
    """API status endpoint"""
    return jsonify({
        "status": "running",
        "message": "YOLOv8 Web App is deployed on Railway",
        "version": "lightweight",
        "note": "ML features can be added later"
    })

@app.route('/api/test')
def test():
    """Test endpoint"""
    return jsonify({
        "message": "API is working!",
        "deployment": "Railway",
        "size": "optimized"
    })

@app.route('/health')
def health():
    """Health check endpoint"""
    return jsonify({"status": "healthy"})

if __name__ == '__main__':
    port = int(os.environ.get('PORT', 5000))
    app.run(host='0.0.0.0', port=port, debug=False) 