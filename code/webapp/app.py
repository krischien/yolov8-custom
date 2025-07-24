from flask import Flask, render_template, request, jsonify, Response, stream_with_context, send_from_directory
from flask_cors import CORS
from flask_socketio import SocketIO, emit
import requests
import json
import time
import threading
import cv2
import numpy as np
from datetime import datetime
import os
import re
import base64
import sys
from config import *
import tempfile

app = Flask(__name__)
app.config['MAX_CONTENT_LENGTH'] = 500 * 1024 * 1024  # 500MB max file size
CORS(app)

# Initialize SocketIO for real-time communication
socketio = SocketIO(app, cors_allowed_origins="*")

# Global variables for people counting
people_counter = {
    "total_count": 0,
    "current_count": 0,
    "direction": "in",  # "in" or "out"
    "line_position": DEFAULT_LINE_POSITION,  # Vertical line position (0-1)
    "is_counting": False,
    "last_update": datetime.now()
}

# Global variables for video streaming
video_streams = {}
active_streams = {}
preview_streams = {}
camera_handles = {}  # Store camera handles for proper cleanup

def validate_video_source(source):
    """Validate and categorize video input source"""
    source = source.strip()
    
    # Camera source (0, 1, 2, etc.)
    if source.isdigit():
        return {
            "type": "camera",
            "source": int(source),
            "valid": True,
            "description": f"Camera {source}"
        }
    
    # IP Camera (RTSP, HTTP, etc.)
    if any(source.startswith(protocol) for protocol in SUPPORTED_STREAM_PROTOCOLS):
        return {
            "type": "ip_camera",
            "source": source,
            "valid": True,
            "description": f"IP Camera: {source}"
        }
    
    # Video file
    if os.path.exists(source) and any(source.lower().endswith(format) for format in SUPPORTED_VIDEO_FORMATS):
        return {
            "type": "video_file",
            "source": source,
            "valid": True,
            "description": f"Video File: {os.path.basename(source)}"
        }
    
    # YouTube URL
    if 'youtube.com' in source or 'youtu.be' in source:
        return {
            "type": "youtube",
            "source": source,
            "valid": True,
            "description": f"YouTube: {source}"
        }
    
    # Try as video file path
    if any(source.lower().endswith(format) for format in SUPPORTED_VIDEO_FORMATS):
        return {
            "type": "video_file",
            "source": source,
            "valid": False,
            "description": f"Video File (not found): {source}"
        }
    
    return {
        "type": "unknown",
        "source": source,
        "valid": False,
        "description": f"Unknown source: {source}"
    }

def get_available_cameras():
    """Get list of available camera devices"""
    available_cameras = []
    
    # Check camera indices up to MAX_CAMERA_INDEX
    for i in range(MAX_CAMERA_INDEX):
        cap = cv2.VideoCapture(i)
        if cap.isOpened():
            available_cameras.append({
                "index": i,
                "name": f"Camera {i}",
                "description": f"Camera device {i}"
            })
            cap.release()
    
    return available_cameras

def stream_preview_frames(source, stream_id):
    """Stream preview frames from a video source"""
    cap = None
    try:
        # Try to convert source to int if it's a camera index
        try:
            if source.isdigit():
                source = int(source)
        except:
            pass
        
        # Release any existing camera handle for this source
        if source in camera_handles:
            try:
                camera_handles[source].release()
                del camera_handles[source]
            except:
                pass
        
        # Wait a moment for camera to be released
        time.sleep(0.5)
        
        cap = cv2.VideoCapture(source)
        if not cap.isOpened():
            error_msg = f'Cannot open camera source: {source}. Please check if camera is available and not in use by another application.'
            socketio.emit('preview_error', {'message': error_msg}, room=stream_id)
            return
        
        # Store camera handle for cleanup
        camera_handles[source] = cap
        
        # Set camera properties for better compatibility
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, 320)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 240)
        cap.set(cv2.CAP_PROP_FPS, 30)
        
        # Wait for camera to initialize
        time.sleep(1)
        
        frame_count = 0
        start_time = time.time()
        consecutive_failures = 0
        
        while stream_id in preview_streams:
            ret, frame = cap.read()
            if not ret:
                consecutive_failures += 1
                if consecutive_failures > 5:
                    socketio.emit('preview_error', {'message': f'Lost connection to camera: {source} after {consecutive_failures} failures'}, room=stream_id)
                    break
                time.sleep(0.1)
                continue
            
            consecutive_failures = 0  # Reset failure counter
            frame_count += 1
            
            # Resize frame for preview
            try:
                frame = cv2.resize(frame, (320, 240))
            except:
                # If resize fails, try to get frame dimensions
                height, width = frame.shape[:2]
                if width > 320 or height > 240:
                    frame = cv2.resize(frame, (320, 240))
            
            # Convert to base64 for transmission
            try:
                _, buffer = cv2.imencode('.jpg', frame, [cv2.IMWRITE_JPEG_QUALITY, 80])
                frame_base64 = base64.b64encode(buffer).decode('utf-8')
                
                # Send frame via WebSocket
                socketio.emit('preview_frame', {
                    'frame': frame_base64,
                    'timestamp': datetime.now().isoformat(),
                    'frame_count': frame_count
                }, room=stream_id)
                
            except Exception as e:
                socketio.emit('preview_error', {'message': f'Frame encoding error: {str(e)}'}, room=stream_id)
                break
            
            # Limit to 15 FPS for better performance
            time.sleep(1/15)
        
        # Cleanup
        if cap:
            cap.release()
            if source in camera_handles:
                del camera_handles[source]
        
    except Exception as e:
        error_msg = f'Camera preview error: {str(e)}. Please check camera permissions and availability.'
        socketio.emit('preview_error', {'message': error_msg}, room=stream_id)
        if cap:
            cap.release()
            if source in camera_handles:
                del camera_handles[source]

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/api/preview/start', methods=['POST'])
def start_preview():
    """Start preview stream for a source"""
    try:
        data = request.get_json()
        source = data.get('source', '')
        stream_id = data.get('stream_id', str(time.time()))
        
        if not source:
            return jsonify({"error": "Source is required"}), 400
        
        validation = validate_video_source(source)
        if not validation["valid"]:
            return jsonify({"error": f"Invalid source: {validation['description']}"}), 400
        
        # Start preview stream in background
        preview_streams[stream_id] = True
        thread = threading.Thread(target=stream_preview_frames, args=(source, stream_id))
        thread.daemon = True
        thread.start()
        
        return jsonify({
            "success": True,
            "stream_id": stream_id,
            "message": f"Preview started for {validation['description']}"
        })
        
    except Exception as e:
        return jsonify({"error": str(e)}), 500

@app.route('/api/preview/stop', methods=['POST'])
def stop_preview():
    """Stop preview stream"""
    try:
        data = request.get_json()
        stream_id = data.get('stream_id', '')
        source = data.get('source', '')
        
        if stream_id in preview_streams:
            del preview_streams[stream_id]
        
        # Release camera handle if provided
        if source and source in camera_handles:
            try:
                camera_handles[source].release()
                del camera_handles[source]
            except:
                pass
        
        return jsonify({"success": True, "message": "Preview stopped and camera released"})
        
    except Exception as e:
        return jsonify({"error": str(e)}), 500

@app.route('/api/preview/release-cameras', methods=['POST'])
def release_all_cameras():
    """Release all camera handles"""
    try:
        released_count = 0
        for source, cap in list(camera_handles.items()):
            try:
                cap.release()
                released_count += 1
            except:
                pass
        
        camera_handles.clear()
        
        return jsonify({
            "success": True, 
            "message": f"Released {released_count} camera handles"
        })
        
    except Exception as e:
        return jsonify({"error": str(e)}), 500

@app.route('/video_feed/<int:camera_index>')
def video_feed(camera_index):
    """Simple video streaming endpoint"""
    def generate_frames():
        cap = cv2.VideoCapture(camera_index)
        if not cap.isOpened():
            return
        
        # Set camera properties
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, 320)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 240)
        cap.set(cv2.CAP_PROP_FPS, 15)
        
        while True:
            ret, frame = cap.read()
            if not ret:
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
        
        cap.release()
    
    return Response(generate_frames(),
                   mimetype='multipart/x-mixed-replace; boundary=frame')

# SocketIO events
@socketio.on('connect')
def handle_connect():
    emit('connected', {'message': 'Connected to preview server'})

@socketio.on('disconnect')
def handle_disconnect():
    print('Client disconnected')

@socketio.on('join_preview')
def handle_join_preview(data):
    stream_id = data.get('stream_id')
    if stream_id:
        socketio.join_room(stream_id)
        emit('joined_preview', {'stream_id': stream_id})

@socketio.on('leave_preview')
def handle_leave_preview(data):
    stream_id = data.get('stream_id')
    if stream_id:
        socketio.leave_room(stream_id)
        if stream_id in preview_streams:
            del preview_streams[stream_id]

@app.route('/api/available-apis')
def get_available_apis():
    """Get list of available YOLO APIs"""
    return jsonify({
        "apis": list(AVAILABLE_APIS.keys()),
        "base_url": API_BASE_URL
    })

@app.route('/api/video-sources/validate', methods=['POST'])
def validate_video_source_endpoint():
    """Validate a video input source"""
    try:
        data = request.get_json()
        source = data.get('source', '')
        
        if not source:
            return jsonify({"error": "Source is required"}), 400
        
        result = validate_video_source(source)
        return jsonify(result)
        
    except Exception as e:
        return jsonify({"error": str(e)}), 500

@app.route('/api/video-sources/cameras')
def get_cameras():
    """Get list of available camera devices"""
    try:
        cameras = get_available_cameras()
        return jsonify({
            "cameras": cameras,
            "count": len(cameras)
        })
    except Exception as e:
        return jsonify({"error": str(e)}), 500

@app.route('/api/video-sources/camera-diagnostics')
def camera_diagnostics():
    """Run comprehensive camera diagnostics"""
    try:
        diagnostics = {
            "total_cameras": 0,
            "working_cameras": [],
            "problem_cameras": [],
            "system_info": {
                "opencv_version": cv2.__version__,
                "python_version": f"{sys.version_info.major}.{sys.version_info.minor}.{sys.version_info.micro}"
            }
        }
        
        # Test cameras up to MAX_CAMERA_INDEX
        for i in range(MAX_CAMERA_INDEX):
            try:
                cap = cv2.VideoCapture(i)
                if cap.isOpened():
                    diagnostics["total_cameras"] += 1
                    
                    # Try to read a frame
                    ret, frame = cap.read()
                    if ret:
                        # Get camera properties
                        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
                        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
                        fps = cap.get(cv2.CAP_PROP_FPS)
                        
                        camera_info = {
                            "index": i,
                            "name": f"Camera {i}",
                            "resolution": f"{width}x{height}",
                            "fps": fps,
                            "status": "working"
                        }
                        diagnostics["working_cameras"].append(camera_info)
                    else:
                        camera_info = {
                            "index": i,
                            "name": f"Camera {i}",
                            "status": "connected_but_no_frames",
                            "error": "Camera opened but cannot read frames"
                        }
                        diagnostics["problem_cameras"].append(camera_info)
                    
                    cap.release()
                else:
                    # Camera not available
                    camera_info = {
                        "index": i,
                        "name": f"Camera {i}",
                        "status": "not_available",
                        "error": "Camera not found or in use"
                    }
                    diagnostics["problem_cameras"].append(camera_info)
                    
            except Exception as e:
                camera_info = {
                    "index": i,
                    "name": f"Camera {i}",
                    "status": "error",
                    "error": str(e)
                }
                diagnostics["problem_cameras"].append(camera_info)
        
        return jsonify(diagnostics)
        
    except Exception as e:
        return jsonify({"error": str(e)}), 500

@app.route('/api/video-sources/examples')
def get_video_source_examples():
    """Get examples of different video sources"""
    try:
        examples_path = os.path.join(os.path.dirname(__file__), 'video_sources_examples.json')
        if os.path.exists(examples_path):
            with open(examples_path, 'r') as f:
                examples = json.load(f)
            return jsonify(examples)
        else:
            return jsonify({"error": "Examples file not found"}), 404
    except Exception as e:
        return jsonify({"error": str(e)}), 500

@app.route('/api/video-sources/test', methods=['POST'])
def test_video_source():
    """Test if a video source is accessible"""
    try:
        data = request.get_json()
        source = data.get('source', '')
        
        if not source:
            return jsonify({"error": "Source is required"}), 400
        
        validation = validate_video_source(source)
        if not validation["valid"]:
            return jsonify({
                "success": False,
                "message": f"Invalid source: {validation['description']}"
            })
        
        # Try to open the video source
        try:
            if validation["type"] == "camera":
                # Convert to int for camera index
                camera_index = int(validation["source"])
                cap = cv2.VideoCapture(camera_index)
                
                # Set camera properties for better compatibility
                cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
                cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
                cap.set(cv2.CAP_PROP_FPS, 30)
            else:
                cap = cv2.VideoCapture(validation["source"])
        except ValueError:
            return jsonify({
                "success": False,
                "message": f"Invalid camera index: {validation['source']}"
            })
        
        if not cap.isOpened():
            if validation["type"] == "camera":
                return jsonify({
                    "success": False,
                    "message": f"Cannot open camera {validation['source']}. Camera may be in use by another application or not available."
                })
            else:
                return jsonify({
                    "success": False,
                    "message": f"Cannot open video source: {validation['description']}"
                })
        
        # Try to read multiple frames to ensure stability
        frame_count = 0
        for i in range(5):  # Try to read 5 frames
            ret, frame = cap.read()
            if ret:
                frame_count += 1
            time.sleep(0.1)  # Small delay between reads
        
        cap.release()
        
        if frame_count == 0:
            return jsonify({
                "success": False,
                "message": f"Cannot read frames from source: {validation['description']}"
            })
        elif frame_count < 3:
            return jsonify({
                "success": True,
                "message": f"Video source accessible but unstable: {validation['description']} (read {frame_count}/5 frames)",
                "source_info": validation,
                "warning": "Source may have connectivity issues"
            })
        else:
            return jsonify({
                "success": True,
                "message": f"Video source is accessible and stable: {validation['description']} (read {frame_count}/5 frames)",
                "source_info": validation
            })
        
    except Exception as e:
        return jsonify({
            "success": False,
            "message": f"Error testing video source: {str(e)}"
        }), 500

@app.route('/api/test-connection')
def test_connection():
    """Test connection to YOLO API"""
    try:
        response = requests.get(f"{API_BASE_URL}/hello_world", timeout=API_TIMEOUT)
        if response.status_code == 200:
            return jsonify({"status": "success", "message": "API connection successful"})
        else:
            return jsonify({"status": "error", "message": "API connection failed"})
    except Exception as e:
        return jsonify({"status": "error", "message": f"Connection error: {str(e)}"})

@app.route('/api/detect', methods=['POST'])
def detect_objects():
    """Generic detection endpoint that routes to specific YOLO APIs"""
    try:
        data = request.get_json()
        api_type = data.get('api_type')
        threshold = data.get('threshold', DEFAULT_CONFIDENCE_THRESHOLD)
        
        if api_type not in AVAILABLE_APIS:
            return jsonify({"error": "Invalid API type"}), 400
        
        # Prepare request data
        api_data = {
            "threshold": threshold
        }
        
        # Add specific parameters based on API type
        if "image" in api_type:
            api_data["image_path"] = data.get('image_path')
        elif "video" in api_type:
            api_data["video_path"] = data.get('video_path')
        
        # Make request to YOLO API
        response = requests.post(
            f"{API_BASE_URL}{AVAILABLE_APIS[api_type]}",
            json=api_data,
            timeout=API_TIMEOUT
        )
        
        if response.status_code == 200:
            result = response.json()
            # Update people counter if it's a person detection
            if "person" in api_type and "person_count" in result:
                update_people_counter(result["person_count"])
            return jsonify(result)
        else:
            return jsonify({"error": f"API request failed: {response.text}"}), response.status_code
            
    except Exception as e:
        return jsonify({"error": str(e)}), 500

@app.route('/api/detect-stream', methods=['POST'])
def detect_stream():
    """Detect objects in a video stream with dynamic source"""
    try:
        data = request.get_json()
        api_type = data.get('api_type')
        video_source = data.get('video_source', '')
        threshold = data.get('threshold', DEFAULT_CONFIDENCE_THRESHOLD)
        
        if not video_source:
            return jsonify({"error": "Video source is required"}), 400
        
        if api_type not in AVAILABLE_APIS:
            return jsonify({"error": "Invalid API type"}), 400
        
        # Validate video source
        validation = validate_video_source(video_source)
        if not validation["valid"]:
            return jsonify({"error": f"Invalid video source: {validation['description']}"}), 400
        
        # Prepare request data for streaming API
        api_data = {
            "threshold": threshold,
            "video_path": video_source,  # Changed from video_source to video_path
            "source_type": validation["type"]
        }
        
        # Use the appropriate streaming API
        if "person" in api_type:
            stream_api = "/person/detect_video_stream"
        elif "car" in api_type:
            stream_api = "/car/detect_video_stream"
        else:
            return jsonify({"error": "Unsupported API type for streaming"}), 400
        
        # Make request to YOLO API
        response = requests.post(
            f"{API_BASE_URL}{stream_api}",
            json=api_data,
            timeout=API_TIMEOUT,
            stream=True
        )
        
        if response.status_code == 200:
            def generate():
                for line in response.iter_lines():
                    if line:
                        yield f"data: {line.decode('utf-8')}\n\n"
            
            return Response(stream_with_context(generate()), content_type='text/event-stream')
        else:
            return jsonify({"error": f"Streaming API request failed: {response.text}"}), response.status_code
            
    except Exception as e:
        return jsonify({"error": str(e)}), 500

@app.route('/api/people-counter/start')
def start_people_counting():
    """Start people counting mode"""
    people_counter["is_counting"] = True
    people_counter["total_count"] = 0
    people_counter["current_count"] = 0
    people_counter["last_update"] = datetime.now()
    return jsonify({"status": "success", "message": "People counting started"})

@app.route('/api/people-counter/stop')
def stop_people_counting():
    """Stop people counting mode"""
    people_counter["is_counting"] = False
    return jsonify({"status": "success", "message": "People counting stopped"})

@app.route('/api/people-counter/reset')
def reset_people_counter():
    """Reset people counter"""
    people_counter["total_count"] = 0
    people_counter["current_count"] = 0
    people_counter["last_update"] = datetime.now()
    return jsonify({"status": "success", "message": "Counter reset"})

@app.route('/api/people-counter/status')
def get_people_counter_status():
    """Get current people counter status"""
    return jsonify({
        "total_count": people_counter["total_count"],
        "current_count": people_counter["current_count"],
        "direction": people_counter["direction"],
        "line_position": people_counter["line_position"],
        "is_counting": people_counter["is_counting"],
        "last_update": people_counter["last_update"].isoformat()
    })

@app.route('/api/people-counter/set-line', methods=['POST'])
def set_counting_line():
    """Set the position of the counting line"""
    try:
        data = request.get_json()
        position = data.get('position', DEFAULT_LINE_POSITION)
        if 0 <= position <= 1:
            people_counter["line_position"] = position
            return jsonify({"status": "success", "line_position": position})
        else:
            return jsonify({"error": "Position must be between 0 and 1"}), 400
    except Exception as e:
        return jsonify({"error": str(e)}), 500

def update_people_counter(count):
    """Update people counter with new detection count"""
    if people_counter["is_counting"]:
        # Simple logic: if count increases, someone entered
        if count > people_counter["current_count"]:
            people_counter["total_count"] += (count - people_counter["current_count"])
        people_counter["current_count"] = count
        people_counter["last_update"] = datetime.now()

@app.route('/api/stream-detection')
def stream_detection():
    """Stream detection results for real-time updates"""
    def generate():
        while True:
            if people_counter["is_counting"]:
                data = {
                    "total_count": people_counter["total_count"],
                    "current_count": people_counter["current_count"],
                    "timestamp": datetime.now().isoformat()
                }
                yield f"data: {json.dumps(data)}\n\n"
            time.sleep(1)
    
    return Response(stream_with_context(generate()), content_type='text/event-stream')

@app.route('/api/detect-upload', methods=['POST'])
def detect_upload():
    print("[detect-upload] Endpoint called")
    try:
        api_type = request.form.get('api_type')
        threshold = float(request.form.get('threshold', 0.6))
        detection_type = request.form.get('detection_type', 'image')
        file = request.files.get('file')
        print("[detect-upload] File received")
        
        if not file or not api_type:
            print("[detect-upload] Missing file or api_type")
            return jsonify({'error': 'Missing file or api_type'}), 400

        # Save file to a temporary location
        import uuid
        import os
        temp_filename = f"temp_{uuid.uuid4().hex}{'.jpg' if detection_type=='image' else '.mp4'}"
        temp_path = os.path.join(tempfile.gettempdir(), temp_filename)
        file.save(temp_path)
        print(f"[detect-upload] File saved to {temp_path}")

        # Prepare payload for backend API
        print("[detect-upload] Building payload")
        payload = {
            'threshold': threshold
        }
        files = None
        if detection_type == 'image':
            payload['image_path'] = temp_path
            endpoint = '/person/detect_image' if 'person' in api_type else '/car/detect_image'
        elif detection_type == 'video':
            payload['video_path'] = temp_path
            # Add frame skip to payload only if provided
            frame_skip = request.form.get('frame_skip')
            if frame_skip:
                try:
                    payload['frame_skip'] = int(frame_skip)
                except ValueError:
                    payload['frame_skip'] = 10  # Default if invalid
            else:
                payload['frame_skip'] = 10  # Default if not provided
            endpoint = '/person/detect_video' if 'person' in api_type else '/car/detect_video'
        else:
            print("[detect-upload] Unsupported detection type")
            return jsonify({'error': 'Unsupported detection type'}), 400

        # Call the YOLO API backend
        api_url = f"{API_BASE_URL}{endpoint}"
        print(f"[detect-upload] About to POST to YOLO API: {api_url} with payload: {payload}")
        try:
            response = requests.post(api_url, json=payload, timeout=API_TIMEOUT)
            print(f"[detect-upload] Got response from YOLO API: {response.status_code}")
            result = response.json()
        except Exception as e:
            print(f"[detect-upload] Exception during POST to YOLO API: {e}")
            return jsonify({'error': f'Backend API error: {str(e)}'}), 500
        finally:
            # Schedule file deletion for later to avoid Windows file handle issues
            import threading
            def delete_file_later():
                import time
                time.sleep(2)  # Wait 2 seconds
                try:
                    if os.path.exists(temp_path):
                        os.remove(temp_path)
                except Exception:
                    pass  # Silently ignore file deletion errors
            
            # Start deletion in background thread
            threading.Thread(target=delete_file_later, daemon=True).start()

        return jsonify(result)
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/api/detect-camera', methods=['POST'])
def detect_camera():
    """Detect objects in a local camera"""
    try:
        data = request.get_json()
        api_type = data.get('api_type')
        camera_index = data.get('camera_index', 0)
        threshold = data.get('threshold', DEFAULT_CONFIDENCE_THRESHOLD)
        
        if api_type not in AVAILABLE_APIS:
            return jsonify({"error": "Invalid API type"}), 400
        
        # Prepare request data for camera API
        api_data = {
            "threshold": threshold,
            "camera_index": camera_index
        }
        
        # Use the appropriate camera API
        if "person" in api_type:
            camera_api = "/camera/detect_person"
        elif "car" in api_type:
            camera_api = "/camera/detect_car"
        else:
            return jsonify({"error": "Unsupported API type for camera detection"}), 400
        
        # Make request to YOLO API
        response = requests.post(
            f"{API_BASE_URL}{camera_api}",
            json=api_data,
            timeout=API_TIMEOUT
        )
        
        if response.status_code == 200:
            result = response.json()
            # Update people counter if it's a person detection
            if "person" in api_type and "person_count" in result:
                update_people_counter(result["person_count"])
            return jsonify(result)
        else:
            return jsonify({"error": f"Camera API request failed: {response.text}"}), response.status_code
            
    except Exception as e:
        return jsonify({"error": str(e)}), 500

@app.route('/camera/preview/<int:camera_index>')
def camera_preview_proxy(camera_index):
    """Proxy camera preview requests to YOLO API backend"""
    try:
        response = requests.get(
            f"{API_BASE_URL}/camera/preview/{camera_index}",
            stream=True,
            timeout=30
        )
        
        if response.status_code == 200:
            def generate():
                for chunk in response.iter_content(chunk_size=8192):
                    if chunk:
                        yield chunk
            
            return Response(
                generate(),
                content_type=response.headers.get('content-type', 'multipart/x-mixed-replace; boundary=frame')
            )
        else:
            return jsonify({"error": f"Camera preview failed: {response.text}"}), response.status_code
            
    except Exception as e:
        return jsonify({"error": f"Camera preview error: {str(e)}"}), 500

@app.route('/camera/detect_live/<int:camera_index>')
def camera_detect_live_proxy(camera_index):
    """Proxy live person detection requests to YOLO API backend"""
    try:
        # Forward query parameters to the API
        query_string = request.query_string.decode('utf-8')
        api_url = f"{API_BASE_URL}/camera/detect_live/{camera_index}"
        if query_string:
            api_url += f"?{query_string}"
        
        response = requests.get(
            api_url,
            stream=True,
            timeout=30
        )
        
        if response.status_code == 200:
            def generate():
                for chunk in response.iter_content(chunk_size=8192):
                    if chunk:
                        yield chunk
            
            return Response(
                generate(),
                content_type=response.headers.get('content-type', 'multipart/x-mixed-replace; boundary=frame')
            )
        else:
            return jsonify({"error": f"Live detection failed: {response.text}"}), response.status_code
            
    except Exception as e:
        return jsonify({"error": f"Live detection error: {str(e)}"}), 500

@app.route('/camera/detect_live_car/<int:camera_index>')
def camera_detect_live_car_proxy(camera_index):
    """Proxy live car detection requests to YOLO API backend"""
    try:
        # Forward query parameters to the API
        query_string = request.query_string.decode('utf-8')
        api_url = f"{API_BASE_URL}/camera/detect_live_car/{camera_index}"
        if query_string:
            api_url += f"?{query_string}"
        
        response = requests.get(
            api_url,
            stream=True,
            timeout=30
        )
        
        if response.status_code == 200:
            def generate():
                for chunk in response.iter_content(chunk_size=8192):
                    if chunk:
                        yield chunk
            
            return Response(
                generate(),
                content_type=response.headers.get('content-type', 'multipart/x-mixed-replace; boundary=frame')
            )
        else:
            return jsonify({"error": f"Live car detection failed: {response.text}"}), response.status_code
            
    except Exception as e:
        return jsonify({"error": f"Live car detection error: {str(e)}"}), 500

@app.route('/api/live-detection/counts')
def get_live_detection_counts_proxy():
    """Get live detection counts from YOLO API backend"""
    try:
        response = requests.get(
            f"{API_BASE_URL}/live-detection/counts",
            timeout=5
        )
        
        if response.status_code == 200:
            return jsonify(response.json())
        else:
            return jsonify({"error": f"Failed to get live detection counts: {response.text}"}), response.status_code
            
    except Exception as e:
        return jsonify({"error": f"Live detection counts error: {str(e)}"}), 500

if __name__ == '__main__':
    app.run(debug=False, host=WEBAPP_HOST, port=WEBAPP_PORT) 