import cv2
import torch
from ultralytics import YOLO
import yt_dlp
import subprocess
from collections import deque
from cv2 import CascadeClassifier

# Load YOLOv8 model
model = YOLO('./yolo11m.pt')

# YouTube Live Stream URL
YOUTUBE_URL = "https://www.youtube.com/watch?v=p0Qhe4vhYLQ"

# Extract direct video stream URL using yt-dlp
def get_youtube_stream_url(url):
    ydl_opts = {
        'format': 'best',
        'quiet': True,
    }
    with yt_dlp.YoutubeDL(ydl_opts) as ydl:
        info_dict = ydl.extract_info(url, download=False)
        return info_dict['url'] if 'url' in info_dict else None

stream_url = get_youtube_stream_url(YOUTUBE_URL)
if not stream_url:
    print("Error: Cannot retrieve stream URL.")
    exit()

# Open video capture
cap = cv2.VideoCapture(stream_url)

if not cap.isOpened():
    print("Error: Cannot open video stream.")
    exit()

# Initialize person counter and tracker
person_count = 0
unique_persons = set()
track_history = deque(maxlen=50)  # Store recent centroids for tracking

# Load Haar cascade for face detection
face_cascade = CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

def calculate_centroid(box):
    x1, y1, x2, y2 = box
    return (int((x1 + x2) / 2), int((y1 + y2) / 2))

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    # Run YOLOv8 inference with a confidence threshold of 95%
    results = model.predict(frame, conf=0.60)

    # Reset person counter for each frame
    person_count = 0
    current_frame_centroids = []

    # Display results
    for result in results:
        for box, cls in zip(result.boxes.xyxy, result.boxes.cls):
            if cls == 0:  # Assuming class 0 is 'person'
                person_count += 1
                x1, y1, x2, y2 = map(int, box)
                cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)

                # Calculate centroid and track unique persons
                centroid = calculate_centroid((x1, y1, x2, y2))
                current_frame_centroids.append(centroid)

                # Detect faces within the person's bounding box
                person_roi = frame[y1:y2, x1:x2]
                faces = face_cascade.detectMultiScale(person_roi, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))
                for (fx, fy, fw, fh) in faces:
                    # Draw red box around the face
                    cv2.rectangle(frame, (x1 + fx, y1 + fy), (x1 + fx + fw, y1 + fy + fh), (0, 0, 255), 2)

    # Update unique persons
    for centroid in current_frame_centroids:
        if all(
            abs(centroid[0] - prev[0]) > 50 or abs(centroid[1] - prev[1]) > 50
            for prev in track_history
        ):
            unique_persons.add(len(unique_persons) + 1)  # Assign a new unique ID
    track_history.extend(current_frame_centroids)

    # Log person count and unique persons to the console
    print(f"Persons in frame: {person_count}, Unique Persons: {len(unique_persons)}")

    # Display person count and unique persons on the frame
    cv2.putText(frame, f'Persons: {person_count}', (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
    cv2.putText(frame, f'Unique Persons: {len(unique_persons)}', (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 255), 2)

    cv2.imshow("YOLOv8 Detection", frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()