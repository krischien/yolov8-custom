#!/usr/bin/env python3
"""
Simple Camera Preview
Direct camera access without WebSocket complexity
"""

import cv2
import time
import threading
import base64
from flask import Flask, render_template, Response, jsonify
from flask_cors import CORS

app = Flask(__name__)
CORS(app)

# Global variables
camera_active = False
current_camera = None
camera_thread = None

def generate_frames(camera_index=0):
    """Generate camera frames"""
    global camera_active, current_camera
    
    # Release any existing camera
    if current_camera:
        current_camera.release()
    
    # Open camera
    current_camera = cv2.VideoCapture(camera_index)
    
    if not current_camera.isOpened():
        print(f"Cannot open camera {camera_index}")
        return
    
    # Set camera properties
    current_camera.set(cv2.CAP_PROP_FRAME_WIDTH, 320)
    current_camera.set(cv2.CAP_PROP_FRAME_HEIGHT, 240)
    current_camera.set(cv2.CAP_PROP_FPS, 15)
    
    print(f"Camera {camera_index} opened successfully")
    
    while camera_active:
        ret, frame = current_camera.read()
        if not ret:
            print("Failed to read frame")
            break
        
        # Resize frame
        frame = cv2.resize(frame, (320, 240))
        
        # Encode frame
        ret, buffer = cv2.imencode('.jpg', frame, [cv2.IMWRITE_JPEG_QUALITY, 80])
        if not ret:
            continue
        
        # Convert to bytes
        frame_bytes = buffer.tobytes()
        
        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + frame_bytes + b'\r\n')
        
        time.sleep(1/15)  # 15 FPS
    
    if current_camera:
        current_camera.release()

@app.route('/')
def index():
    return """
    <!DOCTYPE html>
    <html>
    <head>
        <title>Simple Camera Preview</title>
        <style>
            body { font-family: Arial, sans-serif; margin: 20px; }
            .container { max-width: 800px; margin: 0 auto; }
            .controls { margin: 20px 0; }
            .preview { text-align: center; }
            img { border: 2px solid #ccc; border-radius: 8px; }
            button { padding: 10px 20px; margin: 5px; font-size: 16px; }
            .status { padding: 10px; margin: 10px 0; border-radius: 5px; }
            .success { background-color: #d4edda; color: #155724; }
            .error { background-color: #f8d7da; color: #721c24; }
        </style>
    </head>
    <body>
        <div class="container">
            <h1>🎥 Simple Camera Preview</h1>
            
            <div class="controls">
                <button onclick="startCamera(0)">Start Camera 0</button>
                <button onclick="startCamera(1)">Start Camera 1</button>
                <button onclick="stopCamera()">Stop Camera</button>
                <button onclick="testCamera()">Test Camera Access</button>
            </div>
            
            <div id="status" class="status" style="display: none;"></div>
            
            <div class="preview">
                <img id="cameraFeed" src="" style="display: none; max-width: 100%; height: auto;">
                <div id="placeholder">
                    <h3>Click "Start Camera" to begin preview</h3>
                </div>
            </div>
        </div>
        
        <script>
            let currentCamera = null;
            
            function showStatus(message, type) {
                const status = document.getElementById('status');
                status.textContent = message;
                status.className = 'status ' + type;
                status.style.display = 'block';
            }
            
            function startCamera(cameraIndex) {
                showStatus('Starting camera...', 'success');
                
                const img = document.getElementById('cameraFeed');
                const placeholder = document.getElementById('placeholder');
                
                img.src = `/video_feed/${cameraIndex}`;
                img.style.display = 'block';
                placeholder.style.display = 'none';
                
                currentCamera = cameraIndex;
                showStatus(`Camera ${cameraIndex} started successfully`, 'success');
            }
            
            function stopCamera() {
                const img = document.getElementById('cameraFeed');
                const placeholder = document.getElementById('placeholder');
                
                img.src = '';
                img.style.display = 'none';
                placeholder.style.display = 'block';
                
                currentCamera = null;
                showStatus('Camera stopped', 'success');
            }
            
            async function testCamera() {
                showStatus('Testing camera access...', 'success');
                
                try {
                    const response = await fetch('/test_camera');
                    const data = await response.json();
                    
                    if (data.success) {
                        showStatus(`Camera test successful: ${data.message}`, 'success');
                    } else {
                        showStatus(`Camera test failed: ${data.message}`, 'error');
                    }
                } catch (error) {
                    showStatus(`Test error: ${error.message}`, 'error');
                }
            }
        </script>
    </body>
    </html>
    """

@app.route('/video_feed/<int:camera_index>')
def video_feed(camera_index):
    """Video streaming route"""
    global camera_active
    camera_active = True
    return Response(generate_frames(camera_index),
                   mimetype='multipart/x-mixed-replace; boundary=frame')

@app.route('/test_camera')
def test_camera():
    """Test camera access"""
    try:
        cap = cv2.VideoCapture(0)
        if not cap.isOpened():
            return jsonify({
                "success": False,
                "message": "Cannot open camera 0"
            })
        
        ret, frame = cap.read()
        cap.release()
        
        if ret:
            return jsonify({
                "success": True,
                "message": "Camera 0 is accessible and can capture frames"
            })
        else:
            return jsonify({
                "success": False,
                "message": "Camera 0 opened but cannot capture frames"
            })
            
    except Exception as e:
        return jsonify({
            "success": False,
            "message": f"Camera test error: {str(e)}"
        })

if __name__ == '__main__':
    print("🎥 Starting Simple Camera Preview Server...")
    print("📱 Open http://127.0.0.1:5002 in your browser")
    app.run(host='0.0.0.0', port=5002, debug=True) 