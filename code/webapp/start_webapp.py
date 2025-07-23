#!/usr/bin/env python3
"""
Startup script for the YOLO People Counter Web Application
This script can start both the YOLO API and the web application
"""

import subprocess
import sys
import time
import os
import signal
import threading
from pathlib import Path

def check_dependencies():
    """Check if required packages are installed"""
    required_packages = ['flask', 'flask-cors', 'requests', 'opencv-python', 'numpy']
    missing_packages = []
    
    for package in required_packages:
        try:
            __import__(package.replace('-', '_'))
        except ImportError:
            missing_packages.append(package)
    
    if missing_packages:
        print(f"Missing required packages: {', '.join(missing_packages)}")
        print("Please install them using: pip install -r requirements.txt")
        return False
    
    return True

def start_api():
    """Start the YOLO API"""
    api_path = Path(__file__).parent.parent / "API" / "api.py"
    
    if not api_path.exists():
        print(f"Error: API file not found at {api_path}")
        return None
    
    print("Starting YOLO API...")
    try:
        # Change to API directory
        api_dir = api_path.parent
        process = subprocess.Popen(
            [sys.executable, str(api_path)],
            cwd=str(api_dir),
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE
        )
        
        # Wait a moment for the API to start
        time.sleep(3)
        
        if process.poll() is None:
            print("✅ YOLO API started successfully")
            return process
        else:
            stdout, stderr = process.communicate()
            print(f"❌ Failed to start YOLO API: {stderr.decode()}")
            return None
            
    except Exception as e:
        print(f"❌ Error starting YOLO API: {e}")
        return None

def start_webapp():
    """Start the web application"""
    webapp_path = Path(__file__).parent / "app.py"
    
    if not webapp_path.exists():
        print(f"Error: Web app file not found at {webapp_path}")
        return None
    
    print("Starting Web Application...")
    try:
        process = subprocess.Popen(
            [sys.executable, str(webapp_path)],
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE
        )
        
        # Wait a moment for the web app to start
        time.sleep(3)
        
        if process.poll() is None:
            print("✅ Web Application started successfully")
            print("🌐 Access the application at: http://localhost:5001")
            return process
        else:
            stdout, stderr = process.communicate()
            print(f"❌ Failed to start Web Application: {stderr.decode()}")
            return None
            
    except Exception as e:
        print(f"❌ Error starting Web Application: {e}")
        return None

def signal_handler(signum, frame):
    """Handle shutdown signals"""
    print("\n🛑 Shutting down...")
    sys.exit(0)

def main():
    """Main function"""
    print("🚀 YOLO People Counter Web Application")
    print("=" * 50)
    
    # Check dependencies
    if not check_dependencies():
        sys.exit(1)
    
    # Set up signal handlers
    signal.signal(signal.SIGINT, signal_handler)
    signal.signal(signal.SIGTERM, signal_handler)
    
    # Start API
    api_process = start_api()
    if not api_process:
        print("❌ Cannot start web application without API")
        sys.exit(1)
    
    # Start Web Application
    webapp_process = start_webapp()
    if not webapp_process:
        print("❌ Failed to start web application")
        api_process.terminate()
        sys.exit(1)
    
    print("\n🎉 Both services are running!")
    print("📊 YOLO API: http://localhost:5000")
    print("🌐 Web Application: http://localhost:5001")
    print("\nPress Ctrl+C to stop both services")
    
    try:
        # Keep the script running
        while True:
            time.sleep(1)
            
            # Check if processes are still running
            if api_process.poll() is not None:
                print("❌ YOLO API stopped unexpectedly")
                break
                
            if webapp_process.poll() is not None:
                print("❌ Web Application stopped unexpectedly")
                break
                
    except KeyboardInterrupt:
        print("\n🛑 Stopping services...")
    
    finally:
        # Clean up processes
        if api_process and api_process.poll() is None:
            api_process.terminate()
            print("✅ YOLO API stopped")
            
        if webapp_process and webapp_process.poll() is None:
            webapp_process.terminate()
            print("✅ Web Application stopped")

if __name__ == "__main__":
    main() 