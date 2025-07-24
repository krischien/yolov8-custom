from flask import Flask, jsonify, request, Response, stream_with_context
from ultralytics import YOLO  # Import YOLO
import cv2  # Import OpenCV
import yt_dlp  # Import yt_dlp for YouTube video handling
import numpy as np
from sort import Sort
import torch # Import torch for GPU optimization
from datetime import datetime # Import datetime for timestamp

app = Flask(__name__)

# Global model cache to avoid reloading models
model_cache = {}

# Global variables for live detection counts
live_detection_counts = {
    "person_count": 0,
    "car_count": 0,
    "last_update": None
}

def get_model(model_path):
    """Get model from cache or load it"""
    if model_path not in model_cache:
        try:
            print(f"Loading model: {model_path}")
            model_cache[model_path] = YOLO(model_path)
            print(f"Model loaded successfully: {model_path}")
        except Exception as e:
            print(f"Error loading model {model_path}: {e}")
            raise e
    return model_cache[model_path]

@app.route('/hello_world', methods=['GET'])
def hello_world():
    return jsonify({"message": "Hello, World!"})

#PERSONS API
@app.route('/person/detect_image', methods=['POST'])
def detect_image():
    try:
        # Extract the JSON data from the request        
        data = request.get_json()
        if not data or 'image_path' not in data:
            return jsonify({"error": "Image path is required"}), 400

        if not data or 'threshold' not in data:
            threshold = 0.60
        else:
            threshold = data['threshold']
        
        image_path = data['image_path']
    
        # Load YOLOv8 model using cache
        model = get_model('./API/yolo11m.pt')

        # Load image
        image = cv2.imread(image_path)
        if image is None:
            return jsonify({"error": "Image not found or unable to load"}), 400

        # Run prediction with a confidence threshold of 60% by default
        results = model.predict(image, conf=threshold)

        # Count persons in the image
        person_count = 0
        for result in results:
            for cls in result.boxes.cls:
                if cls == 0:  # Assuming class 0 is 'person'
                    person_count += 1

        return jsonify({"person_count": person_count})
    except Exception as e:
        return jsonify({"error": str(e)}), 500

@app.route('/person/detect_video', methods=['POST'])
def detect_video():
    try:
        # Extract the JSON data from the request 
        data = request.get_json()
        if not data or 'video_path' not in data:
            return jsonify({"error": "Video path is required"}), 400

        if not data or 'threshold' not in data:
            threshold = 0.60
        else:
            threshold = data['threshold']
        
        # Add frame sampling parameter (process every Nth frame)
        frame_skip = data.get('frame_skip', 10)  # Process every 10th frame by default

        video_path = data['video_path']

        # Load YOLOv8 model using cache
        model = get_model('./API/yolo11m.pt')

        # Open video capture
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            return jsonify({"error": "Cannot open video stream"}), 400

        # Get video properties
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        fps = cap.get(cv2.CAP_PROP_FPS)
        duration = total_frames / fps if fps > 0 else 0

        # Initialize counters
        total_person_count = 0
        processed_frames = 0
        frame_count = 0

        # Batch processing for better GPU utilization
        batch_size = 4
        frames_batch = []

        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break

            frame_count += 1
            
            # Only process every Nth frame
            if frame_count % frame_skip != 0:
                continue

            frames_batch.append(frame)
            processed_frames += 1

            print(f"[detect_video] Analyzing frame {frame_count}/{total_frames} (processed: {processed_frames})")

            # Process batch when it reaches batch_size or at the end
            if len(frames_batch) >= batch_size or not ret:
                if frames_batch:
                    # Run YOLOv8 inference on batch
                    results = model.predict(frames_batch, conf=threshold, verbose=False)

                    # Count persons in each frame of the batch
                    for result in results:
                        frame_person_count = 0
                        for cls in result.boxes.cls:
                            if cls == 0:  # Assuming class 0 is 'person'
                                frame_person_count += 1
                        total_person_count += frame_person_count

                    frames_batch = []

        cap.release()

        # Calculate average persons per frame and estimate total
        avg_persons_per_frame = total_person_count / processed_frames if processed_frames > 0 else 0
        estimated_total_persons = avg_persons_per_frame * (total_frames / frame_skip)

        return jsonify({
            "person_count": int(estimated_total_persons),
            "processed_frames": processed_frames,
            "total_frames": total_frames,
            "frame_skip": frame_skip,
            "video_duration_seconds": round(duration, 2),
            "avg_persons_per_processed_frame": round(avg_persons_per_frame, 2)
        })
    except Exception as e:
        return jsonify({"error": str(e)}), 500


def get_youtube_stream_url(url):
    """Extract the direct video stream URL from a YouTube URL."""
    ydl_opts = {
        'format': 'best',
        'quiet': True,
    }
    try:
        with yt_dlp.YoutubeDL(ydl_opts) as ydl:
            info_dict = ydl.extract_info(url, download=False)
            return info_dict['url'] if 'url' in info_dict else None
    except Exception as e:
        raise ValueError(f"Error retrieving YouTube stream URL: {str(e)}")

#Car API
@app.route('/car/detect_image', methods=['POST'])
def detect_car_image():
    try:
        # Extract the JSON data from the request        
        data = request.get_json()
        if not data or 'image_path' not in data:
            return jsonify({"error": "Image path is required"}), 400

        if not data or 'threshold' not in data:
            threshold = 0.60
        else:
            threshold = data['threshold']
        
        image_path = data['image_path']
    
        # Load YOLOv8 model using cache
        model = get_model('./models/best.pt')

        # Load image
        image = cv2.imread(image_path)
        if image is None:
            return jsonify({"error": "Image not found or unable to load"}), 400

        # Run prediction with a confidence threshold of 60% by default
        results = model.predict(image, conf=threshold)

        # Count car in the image
        car_count = 0
        for result in results:
            for cls in result.boxes.cls:
                if cls == 0:  # Assuming class 0 is 'car'
                    car_count += 1

        return jsonify({"car_count": car_count})
    except Exception as e:
        return jsonify({"error": str(e)}), 500

@app.route('/car/detect_video', methods=['POST'])
def detect_car_video():
    try:
        print("[detect_car_video] Called endpoint.")
        # Extract the JSON data from the request 
        data = request.get_json()
        if not data or 'video_path' not in data:
            print("[detect_car_video] No video_path provided.")
            return jsonify({"error": "Video path is required"}), 400

        if not data or 'threshold' not in data:
            threshold = 0.60
        else:
            threshold = data['threshold']
        
        # Add frame sampling parameter (process every Nth frame)
        frame_skip = data.get('frame_skip', 10)  # Process every 10th frame by default

        video_path = data['video_path']
        print(f"[detect_car_video] Video path: {video_path}")

        # Load YOLOv8 model using cache
        try:
            model = get_model('./models/best.pt')
        except Exception as e:
            print(f"[detect_car_video] Error loading model: {e}")
            return jsonify({"error": f"Model load failed: {e}"}), 500
        print("[detect_car_video] Model loaded.")

        # Open video capture
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            print(f"[detect_car_video] Cannot open video stream: {video_path}")
            return jsonify({"error": "Cannot open video stream"}), 400
        print("[detect_car_video] Video opened successfully.")

        # Get video properties
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        fps = cap.get(cv2.CAP_PROP_FPS)
        duration = total_frames / fps if fps > 0 else 0
        print(f"[detect_car_video] Video properties: total_frames={total_frames}, fps={fps}, duration={duration}")

        # Initialize counters
        total_car_count = 0
        processed_frames = 0
        frame_count = 0

        # Batch processing for better GPU utilization
        batch_size = 4
        frames_batch = []

        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                print(f"[detect_car_video] End of video or read error at frame {frame_count}.")
                break

            frame_count += 1
            
            # Only process every Nth frame
            if frame_count % frame_skip != 0:
                continue

            frames_batch.append(frame)
            processed_frames += 1
            print(f"[detect_car_video] Processing frame {frame_count}/{total_frames} (processed: {processed_frames})")

            # Process batch when it reaches batch_size or at the end
            if len(frames_batch) >= batch_size or not ret:
                if frames_batch:
                    print(f"[detect_car_video] About to run model.predict on batch of {len(frames_batch)} frames (device: cpu)")
                    results = model.predict(frames_batch, conf=threshold, verbose=False, device='cpu')
                    print(f"[detect_car_video] Finished model.predict on batch of {len(frames_batch)} frames")

                    # Count cars in each frame of the batch
                    for result in results:
                        frame_car_count = 0
                        for cls in result.boxes.cls:
                            if cls == 0:  # Assuming class 0 is 'car'
                                frame_car_count += 1
                        total_car_count += frame_car_count

                    frames_batch = []

        cap.release()
        print(f"[detect_car_video] Finished processing. Total cars: {total_car_count}, Processed frames: {processed_frames}")

        # Calculate average cars per frame and estimate total
        avg_cars_per_frame = total_car_count / processed_frames if processed_frames > 0 else 0
        estimated_total_cars = avg_cars_per_frame * (total_frames / frame_skip)

        return jsonify({
            "car_count": int(estimated_total_cars),
            "processed_frames": processed_frames,
            "total_frames": total_frames,
            "frame_skip": frame_skip,
            "video_duration_seconds": round(duration, 2),
            "avg_cars_per_processed_frame": round(avg_cars_per_frame, 2)
        })
    except Exception as e:
        return jsonify({"error": str(e)}), 500

@app.route('/car/detect_video_fast', methods=['POST'])
def detect_car_video_fast():
    """Optimized car video detection with aggressive performance improvements"""
    try:
        data = request.get_json()
        if not data or 'video_path' not in data:
            return jsonify({"error": "Video path is required"}), 400

        threshold = data.get('threshold', 0.60)
        frame_skip = data.get('frame_skip', 15)  # More aggressive default
        max_frames = data.get('max_frames', 300)  # Limit total frames to process

        video_path = data['video_path']

        # Use cached model
        model = get_model('./models/best.pt')

        # Open video capture
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            return jsonify({"error": "Cannot open video stream"}), 400

        # Get video properties
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        fps = cap.get(cv2.CAP_PROP_FPS)
        duration = total_frames / fps if fps > 0 else 0

        # Calculate frames to process
        frames_to_process = min(max_frames, total_frames // frame_skip)
        
        # Initialize counters
        total_car_count = 0
        processed_frames = 0
        frame_count = 0

        # Larger batch size for better GPU utilization
        batch_size = 8
        frames_batch = []

        while cap.isOpened() and processed_frames < frames_to_process:
            ret, frame = cap.read()
            if not ret:
                break

            frame_count += 1
            
            # Only process every Nth frame
            if frame_count % frame_skip != 0:
                continue

            # Resize frame for faster processing (optional)
            frame = cv2.resize(frame, (640, 480))  # Smaller size = faster processing
            
            frames_batch.append(frame)
            processed_frames += 1

            # Process batch when it reaches batch_size or at the end
            if len(frames_batch) >= batch_size or processed_frames >= frames_to_process:
                if frames_batch:
                    # Run YOLOv8 inference on batch with optimizations
                    results = model.predict(
                        frames_batch, 
                        conf=threshold, 
                        verbose=False,
                        device='0' if torch.cuda.is_available() else 'cpu'  # Use GPU if available
                    )

                    # Count cars in each frame of the batch
                    for result in results:
                        frame_car_count = 0
                        for cls in result.boxes.cls:
                            if cls == 0:  # Assuming class 0 is 'car'
                                frame_car_count += 1
                        total_car_count += frame_car_count

                    frames_batch = []

        cap.release()

        # Calculate average cars per frame and estimate total
        avg_cars_per_frame = total_car_count / processed_frames if processed_frames > 0 else 0
        estimated_total_cars = avg_cars_per_frame * (total_frames / frame_skip)

        return jsonify({
            "car_count": int(estimated_total_cars),
            "processed_frames": processed_frames,
            "total_frames": total_frames,
            "frame_skip": frame_skip,
            "max_frames_processed": max_frames,
            "video_duration_seconds": round(duration, 2),
            "avg_cars_per_processed_frame": round(avg_cars_per_frame, 2),
            "processing_speed": f"{processed_frames}/{total_frames} frames ({round(processed_frames/total_frames*100, 1)}%)"
        })
    except Exception as e:
        return jsonify({"error": str(e)}), 500

@app.route('/person/detect_video_fast', methods=['POST'])
def detect_person_video_fast():
    """Optimized person video detection with aggressive performance improvements"""
    try:
        data = request.get_json()
        if not data or 'video_path' not in data:
            return jsonify({"error": "Video path is required"}), 400

        threshold = data.get('threshold', 0.60)
        frame_skip = data.get('frame_skip', 15)  # More aggressive default
        max_frames = data.get('max_frames', 300)  # Limit total frames to process

        video_path = data['video_path']

        # Use cached model
        model = get_model('./API/yolo11m.pt')

        # Open video capture
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            return jsonify({"error": "Cannot open video stream"}), 400

        # Get video properties
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        fps = cap.get(cv2.CAP_PROP_FPS)
        duration = total_frames / fps if fps > 0 else 0

        # Calculate frames to process
        frames_to_process = min(max_frames, total_frames // frame_skip)
        
        # Initialize counters
        total_person_count = 0
        processed_frames = 0
        frame_count = 0

        # Larger batch size for better GPU utilization
        batch_size = 8
        frames_batch = []

        while cap.isOpened() and processed_frames < frames_to_process:
            ret, frame = cap.read()
            if not ret:
                break

            frame_count += 1
            
            # Only process every Nth frame
            if frame_count % frame_skip != 0:
                continue

            # Resize frame for faster processing (optional)
            frame = cv2.resize(frame, (640, 480))  # Smaller size = faster processing
            
            frames_batch.append(frame)
            processed_frames += 1

            # Process batch when it reaches batch_size or at the end
            if len(frames_batch) >= batch_size or processed_frames >= frames_to_process:
                if frames_batch:
                    # Run YOLOv8 inference on batch with optimizations
                    results = model.predict(
                        frames_batch, 
                        conf=threshold, 
                        verbose=False,
                        device='0' if torch.cuda.is_available() else 'cpu'  # Use GPU if available
                    )

                    # Count persons in each frame of the batch
                    for result in results:
                        frame_person_count = 0
                        for cls in result.boxes.cls:
                            if cls == 0:  # Assuming class 0 is 'person'
                                frame_person_count += 1
                        total_person_count += frame_person_count

                    frames_batch = []

        cap.release()

        # Calculate average persons per frame and estimate total
        avg_persons_per_frame = total_person_count / processed_frames if processed_frames > 0 else 0
        estimated_total_persons = avg_persons_per_frame * (total_frames / frame_skip)

        return jsonify({
            "person_count": int(estimated_total_persons),
            "processed_frames": processed_frames,
            "total_frames": total_frames,
            "frame_skip": frame_skip,
            "max_frames_processed": max_frames,
            "video_duration_seconds": round(duration, 2),
            "avg_persons_per_processed_frame": round(avg_persons_per_frame, 2),
            "processing_speed": f"{processed_frames}/{total_frames} frames ({round(processed_frames/total_frames*100, 1)}%)"
        })
    except Exception as e:
        return jsonify({"error": str(e)}), 500

@app.route('/car/detect_video_stream', methods=['POST'])
def detect_car_video_stream():
    try: 
        # Extract the JSON data from the request
        data = request.get_json()
        if not data or 'video_path' not in data:
            return jsonify({"error": "Video path or YouTube URL is required"}), 400

        if not data or 'threshold' not in data:
            threshold = 0.60
        else:
            threshold = data['threshold']

        video_path = data['video_path']

        # Check if the video_path is a YouTube URL
        if video_path.startswith("http"):
            video_path = get_youtube_stream_url(video_path)
            if not video_path:
                return jsonify({"error": "Cannot retrieve YouTube video stream"}), 400

        # Load YOLOv8 model using cache
        model = get_model('./yolo11m.pt')

        # Open video capture
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            return jsonify({"error": "Cannot open video stream"}), 400

        def generate_stream():
            # Initialize cumulative person counter
            total_person_count = 0

            try:
                while cap.isOpened():
                    ret, frame = cap.read()
                    if not ret:
                        break

                    # Run YOLOv8 inference with a confidence threshold of 60 by default%
                    results = model.predict(frame, conf=threshold)

                    # Count persons in the current frame
                    frame_person_count = 0
                    for result in results:
                        for cls in result.boxes.cls:
                            if cls == 0:  # Assuming class 0 is 'person'
                                frame_person_count += 1

                    # Add the current frame's person count to the total count
                    total_person_count += frame_person_count

                    # Send the current frame's person count to the client
                    yield f"data: Persons in frame: {frame_person_count}\n\n"

                # Send the total person count detected in the video
                yield f"data: Total persons detected: {total_person_count}\n\n"
            except Exception as e:
                # Handle exceptions during streaming
                yield f"data: Error: {str(e)}\n\n"
            finally:
                # Ensure the video capture is released
                cap.release()

        # Use Flask's Response object with stream_with_context for safe streaming
        return Response(stream_with_context(generate_stream()), content_type='text/event-stream')
    except Exception as e:
        return jsonify({"error": str(e)}), 500
 
@app.route('/live-detection/counts', methods=['GET'])
def get_live_detection_counts():
    """Get current live detection counts"""
    return jsonify(live_detection_counts)

@app.route('/car/track_and_count', methods=['POST'])
def car_track_and_count():
    """
    Track and count cars crossing a virtual line in a video or stream.
    POST JSON: {
        "video_path": "path/to/video.mp4" OR "stream_url": "rtsp://...",
        "line_position": 0.5,  # 0.0 (top/left) to 1.0 (bottom/right)
        "orientation": "horizontal"  # or "vertical"
    }
    """
    try:
        data = request.get_json()
        video_path = data.get('video_path') or data.get('stream_url')
        line_position = float(data.get('line_position', 0.5))
        orientation = data.get('orientation', 'horizontal')
        conf = float(data.get('confidence', 0.3))

        # Load YOLO model using cache
        model = get_model('./models/best.pt')  # or your car model path

        # Open video/stream
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            return jsonify({"error": f"Cannot open video/stream: {video_path}"}), 400

        # Get frame size
        ret, frame = cap.read()
        if not ret:
            return jsonify({"error": "Cannot read first frame"}), 400
        H, W = frame.shape[:2]

        # Define line
        if orientation == 'horizontal':
            y_line = int(H * line_position)
            line = ((0, y_line), (W, y_line))
        else:
            x_line = int(W * line_position)
            line = ((x_line, 0), (x_line, H))

        # Tracker
        tracker = Sort()
        counted_ids = set()
        track_last_pos = {}  # track_id -> (prev_cx, prev_cy)

        # For each frame
        cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
        while True:
            ret, frame = cap.read()
            if not ret:
                break

            # YOLO detection
            results = model(frame, conf=conf)
            detections = []
            for r in results:
                for box, cls in zip(r.boxes.xyxy.cpu().numpy(), r.boxes.cls.cpu().numpy()):
                    # Only count cars (class 2 for COCO, or adjust for your model)
                    if int(cls) == 2 or (hasattr(model, 'names') and model.names[int(cls)].lower() == 'car'):
                        x1, y1, x2, y2 = box
                        conf_score = float(r.boxes.conf.cpu().numpy()[0])
                        detections.append([x1, y1, x2, y2, conf_score])

            # Update tracker
            tracks = tracker.update(np.array(detections))

            # Check for line crossing using previous and current positions
            for track in tracks:
                x1, y1, x2, y2, track_id = track
                cx = int((x1 + x2) / 2)
                cy = int((y1 + y2) / 2)
                tid = int(track_id)

                if tid in track_last_pos:
                    prev_cx, prev_cy = track_last_pos[tid]
                    if orientation == 'horizontal':
                        # Check if crossed the line between previous and current frame
                        if (prev_cy < y_line and cy >= y_line) or (prev_cy > y_line and cy <= y_line):
                            if tid not in counted_ids:
                                counted_ids.add(tid)
                    else:
                        if (prev_cx < x_line and cx >= x_line) or (prev_cx > x_line and cx <= x_line):
                            if tid not in counted_ids:
                                counted_ids.add(tid)
                # Update last position
                track_last_pos[tid] = (cx, cy)

        cap.release()
        return jsonify({
            "car_count": len(counted_ids),
            "line_position": line_position,
            "orientation": orientation,
            "message": f"Total unique cars counted crossing the line: {len(counted_ids)}"
        })
    except Exception as e:
        return jsonify({"error": str(e)}), 500

@app.route('/camera/detect_person', methods=['POST'])
def detect_person_camera():
    """Detect persons in a local camera stream"""
    try:
        data = request.get_json()
        camera_index = int(data.get('camera_index', 0))
        threshold = data.get('threshold', 0.60)
        
        print(f"[detect_person_camera] Starting detection on camera {camera_index}")
        
        # Load YOLOv8 model using cache
        model = get_model('./yolo11m.pt')
        
        # Open camera capture
        cap = cv2.VideoCapture(camera_index)
        if not cap.isOpened():
            print(f"[detect_person_camera] Cannot open camera {camera_index}")
            return jsonify({"error": f"Cannot open camera {camera_index}"}), 400
        
        print(f"[detect_person_camera] Camera {camera_index} opened successfully")
        
        # Process a few frames for detection
        total_person_count = 0
        processed_frames = 0
        max_frames = 10  # Process 10 frames for detection
        
        while processed_frames < max_frames:
            ret, frame = cap.read()
            if not ret:
                break
            
            # Run YOLOv8 inference
            results = model.predict(frame, conf=threshold, verbose=False)
            
            # Count persons in the current frame
            frame_person_count = 0
            for result in results:
                for cls in result.boxes.cls:
                    if cls == 0:  # Assuming class 0 is 'person'
                        frame_person_count += 1
            
            total_person_count += frame_person_count
            processed_frames += 1
            
            print(f"[detect_person_camera] Frame {processed_frames}: {frame_person_count} persons")
        
        cap.release()
        
        # Calculate average persons per frame
        avg_persons = total_person_count / processed_frames if processed_frames > 0 else 0
        
        print(f"[detect_person_camera] Detection complete. Average persons: {avg_persons}")
        
        return jsonify({
            "person_count": int(avg_persons),
            "processed_frames": processed_frames,
            "camera_index": camera_index,
            "message": f"Detected {int(avg_persons)} persons on average in camera {camera_index}"
        })
        
    except Exception as e:
        print(f"[detect_person_camera] Error: {e}")
        return jsonify({"error": str(e)}), 500

@app.route('/camera/detect_car', methods=['POST'])
def detect_car_camera():
    """Detect cars in a local camera stream"""
    try:
        data = request.get_json()
        camera_index = int(data.get('camera_index', 0))
        threshold = data.get('threshold', 0.60)
        
        print(f"[detect_car_camera] Starting detection on camera {camera_index}")
        
        # Load YOLOv8 model using cache
        model = get_model('./models/best.pt')
        
        # Open camera capture
        cap = cv2.VideoCapture(camera_index)
        if not cap.isOpened():
            print(f"[detect_car_camera] Cannot open camera {camera_index}")
            return jsonify({"error": f"Cannot open camera {camera_index}"}), 400
        
        print(f"[detect_car_camera] Camera {camera_index} opened successfully")
        
        # Process a few frames for detection
        total_car_count = 0
        processed_frames = 0
        max_frames = 10  # Process 10 frames for detection
        
        while processed_frames < max_frames:
            ret, frame = cap.read()
            if not ret:
                break
            
            # Run YOLOv8 inference
            results = model.predict(frame, conf=threshold, verbose=False)
            
            # Count cars in the current frame
            frame_car_count = 0
            for result in results:
                for cls in result.boxes.cls:
                    if cls == 0:  # Assuming class 0 is 'car'
                        frame_car_count += 1
            
            total_car_count += frame_car_count
            processed_frames += 1
            
            print(f"[detect_car_camera] Frame {processed_frames}: {frame_car_count} cars")
        
        cap.release()
        
        # Calculate average cars per frame
        avg_cars = total_car_count / processed_frames if processed_frames > 0 else 0
        
        print(f"[detect_car_camera] Detection complete. Average cars: {avg_cars}")
        
        return jsonify({
            "car_count": int(avg_cars),
            "processed_frames": processed_frames,
            "camera_index": camera_index,
            "message": f"Detected {int(avg_cars)} cars on average in camera {camera_index}"
        })
        
    except Exception as e:
        print(f"[detect_car_camera] Error: {e}")
        return jsonify({"error": str(e)}), 500

@app.route('/camera/preview/<int:camera_index>')
def camera_preview(camera_index):
    """Stream live camera feed for preview"""
    def generate_frames():
        cap = cv2.VideoCapture(camera_index)
        if not cap.isOpened():
            return
        
        try:
            while True:
                ret, frame = cap.read()
                if not ret:
                    break
                
                # Resize frame for faster streaming
                frame = cv2.resize(frame, (640, 480))
                
                # Convert to JPEG
                ret, buffer = cv2.imencode('.jpg', frame, [cv2.IMWRITE_JPEG_QUALITY, 80])
                if not ret:
                    continue
                
                # Yield frame as multipart response
                yield (b'--frame\r\n'
                       b'Content-Type: image/jpeg\r\n\r\n' + buffer.tobytes() + b'\r\n')
                
        except Exception as e:
            print(f"[camera_preview] Error streaming camera {camera_index}: {e}")
        finally:
            cap.release()
    
    return Response(generate_frames(), 
                   mimetype='multipart/x-mixed-replace; boundary=frame')

@app.route('/camera/detect_live/<int:camera_index>')
def camera_detect_live(camera_index):
    """Stream live camera feed with real-time detection overlay"""
    def generate_frames_with_detection():
        cap = cv2.VideoCapture(camera_index)
        if not cap.isOpened():
            print(f"[detect_live] Cannot open camera {camera_index}")
            return
        
        # Try to read a test frame to verify camera works
        ret, test_frame = cap.read()
        if not ret:
            print(f"[detect_live] Camera {camera_index} cannot read frames")
            cap.release()
            return
        
        # Load YOLO model for detection
        model = get_model('./API/yolo11m.pt')  # Using person model, can be made configurable
        
        # Get confidence threshold from query parameter, default to 0.5
        confidence_threshold = float(request.args.get('confidence', 0.5))
        
        # Track detections across frames for persistence
        tracked_detections = []  # List of [x1, y1, x2, y2, conf, age]
        
        try:
            frame_count = 0
            while True:
                ret, frame = cap.read()
                if not ret:
                    break
                
                frame_count += 1
                
                # Process every 3rd frame for detection (to maintain performance)
                if frame_count % 3 == 0:
                    try:
                        # Run detection with configurable confidence threshold
                        results = model.predict(frame, conf=confidence_threshold, verbose=False)
                        
                        # Update tracked detections
                        current_detections = []
                        for result in results:
                            boxes = result.boxes
                            if boxes is not None:
                                for box in boxes:
                                    x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()
                                    conf = box.conf[0].cpu().numpy()
                                    current_detections.append([x1, y1, x2, y2, conf, 0])
                        
                        # Update tracked detections (simple persistence)
                        tracked_detections = current_detections.copy()
                        
                        # Update global count and log
                        new_count = len(current_detections)
                        if new_count != live_detection_counts["person_count"]:
                            live_detection_counts["person_count"] = new_count
                            live_detection_counts["last_update"] = datetime.now().isoformat()
                            print(f"[live_detection] Person count updated: {new_count} (confidence: {confidence_threshold})")
                        
                    except Exception as e:
                        print(f"[detect_live] Detection error: {e}")
                
                # Draw all tracked detections on frame
                for detection in tracked_detections:
                    x1, y1, x2, y2, conf, age = detection
                    
                    # Draw bounding box with persistence effect
                    color = (0, 255, 0)  # Green for persons
                    thickness = 2
                    
                    # Make boxes more visible with thicker lines for persistent detections
                    if age > 0:
                        thickness = 3
                        # Add slight glow effect for persistent detections
                        cv2.rectangle(frame, (int(x1-1), int(y1-1)), (int(x2+1), int(y2+1)), (0, 200, 0), thickness+1)
                    
                    cv2.rectangle(frame, (int(x1), int(y1)), (int(x2), int(y2)), color, thickness)
                    
                    # Add label with confidence
                    label = f"Person: {conf:.2f}"
                    cv2.putText(frame, label, (int(x1), int(y1)-10), 
                              cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
                
                # Age tracked detections (for persistence across frames)
                for detection in tracked_detections:
                    detection[5] += 1  # Increment age
                
                # Remove old detections (keep for 15 frames = ~0.5 seconds at 30fps)
                tracked_detections = [d for d in tracked_detections if d[5] < 15]
                
                # Resize frame for streaming
                frame = cv2.resize(frame, (640, 480))
                
                # Convert to JPEG
                ret, buffer = cv2.imencode('.jpg', frame, [cv2.IMWRITE_JPEG_QUALITY, 80])
                if not ret:
                    continue
                
                # Yield frame as multipart response
                yield (b'--frame\r\n'
                       b'Content-Type: image/jpeg\r\n\r\n' + buffer.tobytes() + b'\r\n')
                
        except Exception as e:
            print(f"[detect_live] Error streaming camera {camera_index}: {e}")
        finally:
            cap.release()
    
    return Response(generate_frames_with_detection(), 
                   mimetype='multipart/x-mixed-replace; boundary=frame')

@app.route('/camera/detect_live_car/<int:camera_index>')
def camera_detect_live_car(camera_index):
    """Stream live camera feed with real-time car detection overlay"""
    def generate_frames_with_detection():
        cap = cv2.VideoCapture(camera_index)
        if not cap.isOpened():
            print(f"[detect_live_car] Cannot open camera {camera_index}")
            return
        
        # Try to read a test frame to verify camera works
        ret, test_frame = cap.read()
        if not ret:
            print(f"[detect_live_car] Camera {camera_index} cannot read frames")
            cap.release()
            return
        
        # Load YOLO model for car detection
        model = get_model('./API/models/best.pt')
        
        # Get confidence threshold from query parameter, default to 0.5
        confidence_threshold = float(request.args.get('confidence', 0.5))
        
        # Track detections across frames for persistence
        tracked_detections = []  # List of [x1, y1, x2, y2, conf, age]
        
        try:
            frame_count = 0
            while True:
                ret, frame = cap.read()
                if not ret:
                    break
                
                frame_count += 1
                
                # Process every 3rd frame for detection (to maintain performance)
                if frame_count % 3 == 0:
                    try:
                        # Run detection with configurable confidence threshold
                        results = model.predict(frame, conf=confidence_threshold, verbose=False)
                        
                        # Update tracked detections
                        current_detections = []
                        for result in results:
                            boxes = result.boxes
                            if boxes is not None:
                                for box in boxes:
                                    x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()
                                    conf = box.conf[0].cpu().numpy()
                                    current_detections.append([x1, y1, x2, y2, conf, 0])
                        
                        # Update tracked detections (simple persistence)
                        tracked_detections = current_detections.copy()
                        
                        # Update global count and log
                        new_count = len(current_detections)
                        if new_count != live_detection_counts["car_count"]:
                            live_detection_counts["car_count"] = new_count
                            live_detection_counts["last_update"] = datetime.now().isoformat()
                            print(f"[live_detection] Car count updated: {new_count} (confidence: {confidence_threshold})")
                        
                    except Exception as e:
                        print(f"[detect_live_car] Detection error: {e}")
                
                # Draw all tracked detections on frame
                for detection in tracked_detections:
                    x1, y1, x2, y2, conf, age = detection
                    
                    # Draw bounding box with persistence effect
                    color = (255, 0, 0)  # Blue for cars
                    thickness = 2
                    
                    # Make boxes more visible with thicker lines for persistent detections
                    if age > 0:
                        thickness = 3
                        # Add slight glow effect for persistent detections
                        cv2.rectangle(frame, (int(x1-1), int(y1-1)), (int(x2+1), int(y2+1)), (200, 0, 0), thickness+1)
                    
                    cv2.rectangle(frame, (int(x1), int(y1)), (int(x2), int(y2)), color, thickness)
                    
                    # Add label with confidence
                    label = f"Car: {conf:.2f}"
                    cv2.putText(frame, label, (int(x1), int(y1)-10), 
                              cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
                
                # Age tracked detections (for persistence across frames)
                for detection in tracked_detections:
                    detection[5] += 1  # Increment age
                
                # Remove old detections (keep for 15 frames = ~0.5 seconds at 30fps)
                tracked_detections = [d for d in tracked_detections if d[5] < 15]
                
                # Resize frame for streaming
                frame = cv2.resize(frame, (640, 480))
                
                # Convert to JPEG
                ret, buffer = cv2.imencode('.jpg', frame, [cv2.IMWRITE_JPEG_QUALITY, 80])
                if not ret:
                    continue
                
                # Yield frame as multipart response
                yield (b'--frame\r\n'
                       b'Content-Type: image/jpeg\r\n\r\n' + buffer.tobytes() + b'\r\n')
                
        except Exception as e:
            print(f"[detect_live_car] Error streaming camera {camera_index}: {e}")
        finally:
            cap.release()
    
    return Response(generate_frames_with_detection(), 
                   mimetype='multipart/x-mixed-replace; boundary=frame')

# @app.route('/train_model', methods=['POST'])
# def train_model():
#     try:
#         # Extract the JSON data from the request
#         data = request.get_json()
#         if not data or 'data_path' not in data:
#             return jsonify({"error": "Data path is required"}), 400

#         if not data or 'epochs' not in data:
#             epochs = 100
#         else:   
#             epochs = data['epochs']
        
#         if not data or 'imgsz' not in data:
#             imgsz = 640
#         else:   
#             imgsz = data['imgsz']

#         if not data or 'workers' not in data:
#             workers = 1
#         else:   
#             workers = data['workers']

#         if not data or 'batch' not in data:
#             batch = 10
#         else:   
#             batch = data['batch']

#         data_path = data['data_path']

#         # Train YOLOv8 model
#         model = YOLO()
#         model.train(data_path)

#         return jsonify({"message": "Model training completed successfully"})
#     except Exception as e:
#         return jsonify({"error": str(e)}), 500

if __name__ == "__main__":
    app.run(debug=False, host='0.0.0.0', port=5000)
