#!/usr/bin/env python3
"""
Camera Test Script
This script helps diagnose camera access issues
"""

import cv2
import time
import sys

def test_camera(camera_index):
    """Test a specific camera index"""
    print(f"Testing camera index: {camera_index}")
    
    # Try to open camera
    cap = cv2.VideoCapture(camera_index)
    
    if not cap.isOpened():
        print(f"❌ Cannot open camera {camera_index}")
        return False
    
    print(f"✅ Camera {camera_index} opened successfully")
    
    # Get camera properties
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = cap.get(cv2.CAP_PROP_FPS)
    
    print(f"📷 Camera properties:")
    print(f"   Resolution: {width}x{height}")
    print(f"   FPS: {fps}")
    
    # Try to read frames
    frame_count = 0
    start_time = time.time()
    
    print(f"🎬 Testing frame capture...")
    
    for i in range(30):  # Try to read 30 frames
        ret, frame = cap.read()
        if ret:
            frame_count += 1
            if frame_count == 1:
                print(f"✅ First frame captured successfully")
                print(f"   Frame shape: {frame.shape}")
                print(f"   Frame type: {frame.dtype}")
        else:
            print(f"❌ Failed to read frame {i+1}")
            break
        
        time.sleep(0.1)  # Small delay
    
    cap.release()
    
    elapsed_time = time.time() - start_time
    actual_fps = frame_count / elapsed_time if elapsed_time > 0 else 0
    
    print(f"📊 Frame capture results:")
    print(f"   Frames captured: {frame_count}/30")
    print(f"   Actual FPS: {actual_fps:.2f}")
    
    if frame_count > 0:
        print(f"✅ Camera {camera_index} is working!")
        return True
    else:
        print(f"❌ Camera {camera_index} cannot capture frames")
        return False

def test_all_cameras():
    """Test all camera indices from 0 to 9"""
    print("🔍 Testing all available cameras...")
    print("=" * 50)
    
    working_cameras = []
    
    for i in range(10):  # Test cameras 0-9
        print(f"\n--- Testing Camera {i} ---")
        if test_camera(i):
            working_cameras.append(i)
        print("-" * 30)
        time.sleep(1)  # Wait between tests
    
    print(f"\n📋 Summary:")
    print(f"   Working cameras: {working_cameras}")
    print(f"   Total working: {len(working_cameras)}")
    
    if working_cameras:
        print(f"✅ Found {len(working_cameras)} working camera(s)")
        return working_cameras[0]  # Return first working camera
    else:
        print(f"❌ No working cameras found")
        return None

def test_camera_with_display(camera_index):
    """Test camera with live display"""
    print(f"🎥 Testing camera {camera_index} with live display...")
    print("Press 'q' to quit, 's' to save frame")
    
    cap = cv2.VideoCapture(camera_index)
    
    if not cap.isOpened():
        print(f"❌ Cannot open camera {camera_index}")
        return
    
    # Set camera properties
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
    cap.set(cv2.CAP_PROP_FPS, 30)
    
    frame_count = 0
    start_time = time.time()
    
    while True:
        ret, frame = cap.read()
        if not ret:
            print("❌ Failed to read frame")
            break
        
        frame_count += 1
        
        # Add text overlay
        cv2.putText(frame, f"Camera {camera_index} - Frame {frame_count}", 
                   (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
        
        # Calculate FPS
        elapsed_time = time.time() - start_time
        fps = frame_count / elapsed_time if elapsed_time > 0 else 0
        cv2.putText(frame, f"FPS: {fps:.1f}", 
                   (10, 70), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
        
        # Display frame
        cv2.imshow(f'Camera {camera_index} Test', frame)
        
        # Handle key presses
        key = cv2.waitKey(1) & 0xFF
        if key == ord('q'):
            break
        elif key == ord('s'):
            filename = f"camera_{camera_index}_test_frame_{frame_count}.jpg"
            cv2.imwrite(filename, frame)
            print(f"💾 Saved frame as {filename}")
    
    cap.release()
    cv2.destroyAllWindows()
    
    print(f"📊 Test completed:")
    print(f"   Total frames: {frame_count}")
    print(f"   Average FPS: {fps:.1f}")

def main():
    print("🎥 Camera Diagnostic Tool")
    print("=" * 50)
    
    # Check OpenCV version
    print(f"🔧 OpenCV version: {cv2.__version__}")
    print(f"🐍 Python version: {sys.version}")
    print()
    
    # Test all cameras first
    working_camera = test_all_cameras()
    
    if working_camera is not None:
        print(f"\n🎬 Would you like to test camera {working_camera} with live display? (y/n): ", end="")
        try:
            response = input().lower()
            if response == 'y':
                test_camera_with_display(working_camera)
        except KeyboardInterrupt:
            print("\n👋 Test interrupted by user")
    else:
        print("\n💡 Troubleshooting tips:")
        print("   1. Check if camera is in use by another application")
        print("   2. Try closing other camera applications (Zoom, Teams, etc.)")
        print("   3. Check camera permissions in Windows Settings")
        print("   4. Restart your computer")
        print("   5. Update camera drivers")

if __name__ == "__main__":
    main() 