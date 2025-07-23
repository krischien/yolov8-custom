#!/usr/bin/env python3
"""
Test script for video detection debugging
"""

import requests
import time
import os
import tempfile
from PIL import Image
import cv2
import numpy as np

def create_test_video():
    """Create a simple test video file"""
    print("🎬 Creating test video...")
    
    # Create a simple video with moving shapes
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter('test_video.mp4', fourcc, 10.0, (320, 240))
    
    for i in range(50):  # 5 seconds at 10 fps
        # Create frame
        frame = np.zeros((240, 320, 3), dtype=np.uint8)
        
        # Draw moving rectangle
        x = int(50 + 20 * np.sin(i * 0.2))
        y = int(120 + 10 * np.cos(i * 0.3))
        cv2.rectangle(frame, (x, y), (x + 50, y + 30), (0, 255, 0), -1)
        
        # Draw moving circle
        cx = int(200 + 30 * np.sin(i * 0.1))
        cy = int(100 + 20 * np.cos(i * 0.15))
        cv2.circle(frame, (cx, cy), 20, (255, 0, 0), -1)
        
        out.write(frame)
    
    out.release()
    print("✅ Test video created: test_video.mp4")
    return 'test_video.mp4'

def test_video_detection_direct_api():
    """Test video detection directly with the API"""
    print("\n🔍 Testing video detection directly with API...")
    
    video_path = create_test_video()
    
    try:
        payload = {
            'video_path': os.path.abspath(video_path),
            'threshold': 0.6,
            'frame_skip': 5  # Process every 5th frame
        }
        
        print(f"   Calling: http://localhost:5000/person/detect_video")
        print(f"   Video path: {payload['video_path']}")
        print(f"   Payload: {payload}")
        
        start_time = time.time()
        response = requests.post(
            'http://localhost:5000/person/detect_video',
            json=payload,
            timeout=300  # 5 minute timeout for video processing
        )
        end_time = time.time()
        
        print(f"   Response time: {end_time - start_time:.2f} seconds")
        print(f"   Status code: {response.status_code}")
        
        if response.status_code == 200:
            result = response.json()
            print(f"   ✅ Success! Result: {result}")
        else:
            print(f"   ❌ Error: {response.text}")
            
    except requests.exceptions.Timeout:
        print("   ❌ Request timed out after 5 minutes")
    except requests.exceptions.ConnectionError as e:
        print(f"   ❌ Connection error: {e}")
    except Exception as e:
        print(f"   ❌ Unexpected error: {e}")
    finally:
        # Clean up test video
        try:
            if os.path.exists(video_path):
                os.remove(video_path)
        except:
            pass

def test_video_detection_webapp():
    """Test video detection through the webapp"""
    print("\n🔍 Testing video detection through webapp...")
    
    video_path = create_test_video()
    
    try:
        # Prepare form data
        with open(video_path, 'rb') as f:
            files = {'file': ('test_video.mp4', f, 'video/mp4')}
            data = {
                'api_type': 'person_detect_video',
                'threshold': '0.6',
                'detection_type': 'video',
                'frame_skip': '5'
            }
            
            print(f"   Uploading to: http://localhost:5001/api/detect-upload")
            print(f"   File size: {os.path.getsize(video_path)} bytes")
            print(f"   Form data: {data}")
            
            start_time = time.time()
            response = requests.post(
                'http://localhost:5001/api/detect-upload',
                files=files,
                data=data,
                timeout=300  # 5 minute timeout
            )
            end_time = time.time()
            
            print(f"   Response time: {end_time - start_time:.2f} seconds")
            print(f"   Status code: {response.status_code}")
            
            if response.status_code == 200:
                result = response.json()
                print(f"   ✅ Success! Result: {result}")
            else:
                print(f"   ❌ Error: {response.text}")
                
    except requests.exceptions.Timeout:
        print("   ❌ Request timed out after 5 minutes")
    except requests.exceptions.ConnectionError as e:
        print(f"   ❌ Connection error: {e}")
    except Exception as e:
        print(f"   ❌ Unexpected error: {e}")
    finally:
        # Clean up test video
        try:
            if os.path.exists(video_path):
                os.remove(video_path)
        except:
            pass

def test_car_video_detection():
    """Test car video detection"""
    print("\n🔍 Testing car video detection...")
    
    video_path = create_test_video()
    
    try:
        payload = {
            'video_path': os.path.abspath(video_path),
            'threshold': 0.6,
            'frame_skip': 5
        }
        
        print(f"   Calling: http://localhost:5000/car/detect_video")
        
        start_time = time.time()
        response = requests.post(
            'http://localhost:5000/car/detect_video',
            json=payload,
            timeout=300
        )
        end_time = time.time()
        
        print(f"   Response time: {end_time - start_time:.2f} seconds")
        print(f"   Status code: {response.status_code}")
        
        if response.status_code == 200:
            result = response.json()
            print(f"   ✅ Success! Result: {result}")
        else:
            print(f"   ❌ Error: {response.text}")
            
    except Exception as e:
        print(f"   ❌ Error: {e}")
    finally:
        try:
            if os.path.exists(video_path):
                os.remove(video_path)
        except:
            pass

def test_api_health():
    """Test API health and model loading"""
    print("\n🔍 Testing API health...")
    
    try:
        # Test basic endpoint
        response = requests.get('http://localhost:5000/hello_world', timeout=10)
        print(f"   Hello world: {response.status_code}")
        if response.status_code == 200:
            print(f"   Response: {response.text}")
            
        # Test model loading by calling image detection
        test_image = create_test_image()
        payload = {
            'image_path': test_image,
            'threshold': 0.6
        }
        
        response = requests.post(
            'http://localhost:5000/person/detect_image',
            json=payload,
            timeout=30
        )
        print(f"   Model test: {response.status_code}")
        if response.status_code == 200:
            print(f"   Model loaded successfully")
        else:
            print(f"   Model loading issue: {response.text}")
            
    except Exception as e:
        print(f"   ❌ API health check failed: {e}")
    finally:
        try:
            if os.path.exists(test_image):
                os.remove(test_image)
        except:
            pass

def create_test_image():
    """Create a simple test image"""
    img = Image.new('RGB', (100, 100), color='red')
    img_path = 'test_image.jpg'
    img.save(img_path)
    return img_path

def main():
    print("🚀 Video Detection Debug Test")
    print("=" * 60)
    
    # Test API health first
    test_api_health()
    
    # Test video detection directly
    test_video_detection_direct_api()
    
    # Test video detection through webapp
    test_video_detection_webapp()
    
    # Test car video detection
    test_car_video_detection()
    
    print("\n" + "=" * 60)
    print("📋 Video detection debug complete!")

if __name__ == "__main__":
    main() 