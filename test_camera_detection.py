#!/usr/bin/env python3
"""
Test script to verify camera detection endpoints work with query parameters
"""

import requests
import time

def test_camera_detection():
    """Test camera detection endpoints"""
    
    # Test the API directly
    print("Testing API directly...")
    try:
        # Test with confidence parameter
        response = requests.get(
            "http://localhost:5000/camera/detect_live/0?confidence=0.6",
            timeout=5
        )
        print(f"Direct API test: {response.status_code}")
        if response.status_code != 200:
            print(f"Error: {response.text}")
    except Exception as e:
        print(f"Direct API test failed: {e}")
    
    # Test through webapp proxy
    print("\nTesting through webapp proxy...")
    try:
        response = requests.get(
            "http://localhost:5001/camera/detect_live/0?confidence=0.6",
            timeout=5
        )
        print(f"Proxy test: {response.status_code}")
        if response.status_code != 200:
            print(f"Error: {response.text}")
    except Exception as e:
        print(f"Proxy test failed: {e}")

if __name__ == "__main__":
    print("Camera Detection Test")
    print("Waiting 2 seconds for services to start...")
    time.sleep(2)
    test_camera_detection() 