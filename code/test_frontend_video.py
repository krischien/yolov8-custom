#!/usr/bin/env python3
"""
Test frontend video detection improvements
"""

import requests
import time
import os
import cv2
import numpy as np

def create_test_video_with_person():
    """Create a test video that might trigger person detection"""
    print("🎬 Creating test video with person-like shapes...")
    
    # Create a video with person-like rectangles
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter('person_test.mp4', fourcc, 10.0, (640, 480))
    
    for i in range(30):  # 3 seconds at 10 fps
        # Create frame
        frame = np.zeros((480, 640, 3), dtype=np.uint8)
        
        # Draw person-like shape (head + body)
        # Head
        head_x = 320 + int(50 * np.sin(i * 0.3))
        head_y = 150 + int(20 * np.cos(i * 0.2))
        cv2.circle(frame, (head_x, head_y), 30, (255, 255, 255), -1)
        
        # Body
        body_x = head_x
        body_y = head_y + 60
        cv2.rectangle(frame, (body_x - 40, body_y), (body_x + 40, body_y + 120), (255, 255, 255), -1)
        
        # Arms
        cv2.rectangle(frame, (body_x - 60, body_y + 20), (body_x - 40, body_y + 80), (255, 255, 255), -1)
        cv2.rectangle(frame, (body_x + 40, body_y + 20), (body_x + 60, body_y + 80), (255, 255, 255), -1)
        
        # Legs
        cv2.rectangle(frame, (body_x - 20, body_y + 120), (body_x - 5, body_y + 200), (255, 255, 255), -1)
        cv2.rectangle(frame, (body_x + 5, body_y + 120), (body_x + 20, body_y + 200), (255, 255, 255), -1)
        
        out.write(frame)
    
    out.release()
    print("✅ Person test video created: person_test.mp4")
    return 'person_test.mp4'

def test_frontend_video_upload():
    """Test the frontend video upload with improved error handling"""
    print("\n🔍 Testing frontend video upload with improvements...")
    
    video_path = create_test_video_with_person()
    
    try:
        # Test with a shorter timeout to simulate frontend behavior
        with open(video_path, 'rb') as f:
            files = {'file': ('person_test.mp4', f, 'video/mp4')}
            data = {
                'api_type': 'person_detect_video',
                'threshold': '0.3',  # Lower threshold to catch more detections
                'detection_type': 'video',
                'frame_skip': '3'  # Process more frames
            }
            
            print(f"   Uploading to: http://localhost:5001/api/detect-upload")
            print(f"   File size: {os.path.getsize(video_path)} bytes")
            print(f"   Form data: {data}")
            
            start_time = time.time()
            response = requests.post(
                'http://localhost:5001/api/detect-upload',
                files=files,
                data=data,
                timeout=60  # 1 minute timeout to test timeout handling
            )
            end_time = time.time()
            
            print(f"   Response time: {end_time - start_time:.2f} seconds")
            print(f"   Status code: {response.status_code}")
            
            if response.status_code == 200:
                result = response.json()
                print(f"   ✅ Success! Result: {result}")
                
                # Check if we got meaningful results
                if 'person_count' in result:
                    print(f"   📊 Person count: {result['person_count']}")
                    print(f"   📊 Processed frames: {result['processed_frames']}")
                    print(f"   📊 Total frames: {result['total_frames']}")
                    print(f"   📊 Video duration: {result['video_duration_seconds']} seconds")
            else:
                print(f"   ❌ Error: {response.text}")
                
    except requests.exceptions.Timeout:
        print("   ⏰ Request timed out (expected for testing)")
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

def test_webapp_health():
    """Test webapp health"""
    print("\n🔍 Testing webapp health...")
    
    try:
        response = requests.get('http://localhost:5001/', timeout=10)
        print(f"   Main page: {response.status_code}")
        
        response = requests.get('http://localhost:5001/api/test-connection', timeout=10)
        print(f"   API connection: {response.status_code}")
        
    except Exception as e:
        print(f"   ❌ Error: {e}")

def main():
    print("🚀 Frontend Video Detection Test")
    print("=" * 60)
    
    # Test webapp health
    test_webapp_health()
    
    # Test frontend video upload
    test_frontend_video_upload()
    
    print("\n" + "=" * 60)
    print("📋 Frontend video detection test complete!")
    print("\n💡 If the test passes, the video detection should now work properly in the frontend.")
    print("💡 The improvements include:")
    print("   - Better timeout handling (5 minutes for videos)")
    print("   - Progress indicators and button states")
    print("   - Clear success/error messages")
    print("   - Proper error handling for different failure types")

if __name__ == "__main__":
    main() 