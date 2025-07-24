#!/usr/bin/env python3
"""
Test to find available camera indices
"""

import cv2

def test_camera_indices():
    """Test different camera indices"""
    
    print("Testing camera indices...")
    
    for i in range(5):  # Test indices 0-4
        try:
            cap = cv2.VideoCapture(i)
            if cap.isOpened():
                print(f"✓ Camera {i} is accessible")
                ret, frame = cap.read()
                if ret:
                    print(f"  ✓ Camera {i} can read frames - Shape: {frame.shape}")
                else:
                    print(f"  ✗ Camera {i} cannot read frames")
                cap.release()
            else:
                print(f"✗ Camera {i} is not accessible")
        except Exception as e:
            print(f"✗ Camera {i} error: {e}")

if __name__ == "__main__":
    test_camera_indices() 