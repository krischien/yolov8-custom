#!/usr/bin/env python3
"""
Simple Flask app for Railway deployment testing
"""

from flask import Flask, jsonify
import os

app = Flask(__name__)

@app.route('/')
def hello():
    return jsonify({
        "message": "YOLOv8 Web App is running!",
        "status": "success",
        "deployment": "Railway",
        "note": "This is the root-level app for testing"
    })

@app.route('/health')
def health():
    return jsonify({"status": "healthy"})

@app.route('/test')
def test():
    return jsonify({
        "message": "Railway deployment test successful!",
        "structure": "Root-level app.py detected"
    })

if __name__ == '__main__':
    port = int(os.environ.get('PORT', 5000))
    app.run(host='0.0.0.0', port=port, debug=False) 