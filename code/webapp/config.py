"""
Configuration file for the YOLO People Counter Web Application
"""

import os
from pathlib import Path

# API Configuration
API_BASE_URL = os.getenv('YOLO_API_URL', 'http://localhost:5000')
API_TIMEOUT = 2400  # 40 minutes

# Web Application Configuration
WEBAPP_HOST = os.getenv('WEBAPP_HOST', '0.0.0.0')
WEBAPP_PORT = int(os.getenv('WEBAPP_PORT', 5001))
DEBUG_MODE = os.getenv('DEBUG_MODE', 'True').lower() == 'true'

# Available YOLO APIs
AVAILABLE_APIS = {
    "person_detect_image": "/person/detect_image",
    "person_detect_video": "/person/detect_video", 
    "person_detect_video_stream": "/person/detect_video_stream",
    "car_detect_image": "/car/detect_image",
    "car_detect_video": "/car/detect_video",
    "car_detect_video_stream": "/car/detect_video_stream",
    "person_detect_camera": "/camera/detect_person",
    "car_detect_camera": "/camera/detect_car"
}

# People Counter Configuration
DEFAULT_CONFIDENCE_THRESHOLD = float(os.getenv('DEFAULT_CONFIDENCE_THRESHOLD', 0.6))
DEFAULT_LINE_POSITION = float(os.getenv('DEFAULT_LINE_POSITION', 0.5))
COUNTING_UPDATE_INTERVAL = int(os.getenv('COUNTING_UPDATE_INTERVAL', 1000))  # milliseconds

# Logging Configuration
LOG_LEVEL = os.getenv('LOG_LEVEL', 'INFO')
MAX_LOG_ENTRIES = int(os.getenv('MAX_LOG_ENTRIES', 50))

# Security Configuration
ENABLE_CORS = os.getenv('ENABLE_CORS', 'True').lower() == 'true'
CORS_ORIGINS = os.getenv('CORS_ORIGINS', '*').split(',')

# File Paths
BASE_DIR = Path(__file__).parent
TEMPLATES_DIR = BASE_DIR / 'templates'
STATIC_DIR = BASE_DIR / 'static'

# Model Paths (relative to API directory)
DEFAULT_PERSON_MODEL = os.getenv('PERSON_MODEL_PATH', './yolo11m.pt')
DEFAULT_CAR_MODEL = os.getenv('CAR_MODEL_PATH', './models/best.pt')

# Detection Settings
DEFAULT_IMAGE_SIZE = int(os.getenv('DEFAULT_IMAGE_SIZE', 640))
DEFAULT_BATCH_SIZE = int(os.getenv('DEFAULT_BATCH_SIZE', 1))

# Streaming Configuration
STREAM_BUFFER_SIZE = int(os.getenv('STREAM_BUFFER_SIZE', 1024))
STREAM_TIMEOUT = int(os.getenv('STREAM_TIMEOUT', 30))

# Video Source Configuration
SUPPORTED_VIDEO_FORMATS = ['.mp4', '.avi', '.mov', '.mkv', '.flv', '.wmv', '.webm']
SUPPORTED_STREAM_PROTOCOLS = ['rtsp://', 'http://', 'https://', 'rtmp://', 'udp://']
MAX_CAMERA_INDEX = int(os.getenv('MAX_CAMERA_INDEX', 10))
CAMERA_TEST_TIMEOUT = int(os.getenv('CAMERA_TEST_TIMEOUT', 5))

# UI Configuration
UI_THEME = os.getenv('UI_THEME', 'default')  # 'default', 'dark', 'light'
REFRESH_INTERVAL = int(os.getenv('REFRESH_INTERVAL', 5000))  # milliseconds 