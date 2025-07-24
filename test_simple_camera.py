#!/usr/bin/env python3
"""
Simple camera test to check if camera is accessible
"""

import cv2
import requests
import time

def test_camera_access():
    """Test if camera is accessible"""
    
    print("Testing camera access directly with OpenCV...")
    try:
        cap = cv2.VideoCapture(0)
        if cap.isOpened():
            print("✓ Camera 0 is accessible")
            ret, frame = cap.read()
            if ret:
                print("✓ Camera 0 can read frames")
                print(f"  Frame shape: {frame.shape}")
            else:
                print("✗ Camera 0 cannot read frames")
            cap.release()
        else:
            print("✗ Camera 0 is not accessible")
    except Exception as e:
        print(f"✗ Camera 0 error: {e}")
    
    print("\nTesting camera preview endpoint...")
    try:
        response = requests.get(
            "http://localhost:5000/camera/preview/0",
            timeout=5
        )
        print(f"Camera preview: {response.status_code}")
        if response.status_code != 200:
            print(f"Error: {response.text}")
    except Exception as e:
        print(f"Camera preview failed: {e}")

if __name__ == "__main__":
    print("Simple Camera Test")
    print("Waiting 2 seconds for services to start...")
    time.sleep(2)
    test_camera_access() 