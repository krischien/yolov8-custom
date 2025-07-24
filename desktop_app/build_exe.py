#!/usr/bin/env python3
"""
Build script for creating YOLOv8 Desktop Application executable
"""

import os
import sys
import subprocess
import shutil
from pathlib import Path

def build_executable():
    """Build the executable using PyInstaller"""
    
    print("🚀 Building YOLOv8 Desktop Application Executable...")
    print("=" * 60)
    
    # Get current directory
    current_dir = Path(__file__).parent
    api_dir = current_dir.parent / "code" / "API"
    
    # Check if models exist
    person_model = api_dir / "yolo11m.pt"
    car_model = api_dir / "models" / "best.pt"
    
    if not person_model.exists():
        print(f"❌ Person model not found: {person_model}")
        return False
    
    if not car_model.exists():
        print(f"❌ Car model not found: {car_model}")
        return False
    
    print(f"✅ Person model found: {person_model}")
    print(f"✅ Car model found: {car_model}")
    
    # Create PyInstaller command
    cmd = [
        "pyinstaller",
        "--onefile",                    # Single executable file
        "--windowed",                   # No console window
        "--name=YOLOv8_Desktop_App",    # Executable name
        "--icon=icon.ico",              # Icon (if exists)
        "--add-data", f"{api_dir};code/API",  # Include API directory
        "--hidden-import=cv2",
        "--hidden-import=torch",
        "--hidden-import=ultralytics",
        "--hidden-import=customtkinter",
        "--hidden-import=PIL",
        "--hidden-import=numpy",
        "--hidden-import=sort",
        "--hidden-import=kalman_filter",
        "--collect-all=ultralytics",
        "--collect-all=torch",
        "--collect-all=cv2",
        "main.py"
    ]
    
    # Remove icon if it doesn't exist
    if not (current_dir / "icon.ico").exists():
        cmd = [arg for arg in cmd if arg != "--icon=icon.ico"]
    
    print("\n📦 Running PyInstaller...")
    print(f"Command: {' '.join(cmd)}")
    
    try:
        # Run PyInstaller
        result = subprocess.run(cmd, capture_output=True, text=True, cwd=current_dir)
        
        if result.returncode == 0:
            print("\n✅ Build successful!")
            
            # Check if executable was created
            exe_path = current_dir / "dist" / "YOLOv8_Desktop_App.exe"
            if exe_path.exists():
                size_mb = exe_path.stat().st_size / (1024 * 1024)
                print(f"📁 Executable created: {exe_path}")
                print(f"📏 Size: {size_mb:.1f} MB")
                
                # Create a simple launcher script
                create_launcher_script(current_dir)
                
                print("\n🎉 Build complete! Your executable is ready for distribution.")
                print(f"📂 Location: {exe_path}")
                print("\n📋 Distribution Notes:")
                print("- The executable is self-contained (no Python installation needed)")
                print("- Models are included in the executable")
                print("- Users can run it on any Windows 10/11 machine")
                print("- No additional dependencies required")
                
                return True
            else:
                print("❌ Executable not found after build")
                return False
        else:
            print(f"❌ Build failed!")
            print(f"Error: {result.stderr}")
            return False
            
    except Exception as e:
        print(f"❌ Build error: {e}")
        return False

def create_launcher_script(current_dir):
    """Create a simple launcher script"""
    launcher_content = """@echo off
echo Starting YOLOv8 Desktop Application...
echo.
echo This is a standalone executable that includes:
echo - YOLOv8 object detection models
echo - All required dependencies
echo - Modern GUI interface
echo.
echo Features:
echo - Person and car detection
echo - Camera, IP camera, video, and image support
echo - Real-time bounding boxes and counting
echo.
echo Starting application...
echo.

REM Run the executable
"%~dp0YOLOv8_Desktop_App.exe"

echo.
echo Application closed.
pause
"""
    
    launcher_path = current_dir / "dist" / "Run_YOLOv8_App.bat"
    with open(launcher_path, 'w') as f:
        f.write(launcher_content)
    
    print(f"📄 Launcher script created: {launcher_path}")

def create_readme():
    """Create a README for the distribution"""
    readme_content = """# YOLOv8 Desktop Application - Standalone Executable

## 🎉 What's Included

This is a **standalone executable** that includes everything needed to run the YOLOv8 object detection application:

- ✅ **YOLOv8 Models**: Person and car detection models
- ✅ **All Dependencies**: OpenCV, PyTorch, CustomTkinter, etc.
- ✅ **Modern GUI**: Dark theme with intuitive controls
- ✅ **Multiple Input Sources**: Camera, IP Camera, Video, Image
- ✅ **Real-time Detection**: Live bounding boxes and counting

## 🚀 Quick Start

1. **Download** the executable files
2. **Extract** to any folder
3. **Double-click** `YOLOv8_Desktop_App.exe` or `Run_YOLOv8_App.bat`
4. **Start detecting** objects!

## 📋 System Requirements

- **OS**: Windows 10 or Windows 11
- **RAM**: 4GB minimum, 8GB recommended
- **Storage**: 500MB free space
- **Camera**: Optional (for live detection)
- **Network**: Optional (for IP cameras)

## 🎯 Features

### Detection Types
- **Person Detection**: Count and track people
- **Car Detection**: Count and track vehicles

### Input Sources
- **Local Camera**: Built-in or USB webcam
- **IP Camera**: RTSP, HTTP, HTTPS, RTMP, UDP streams
- **Video Files**: MP4, AVI, MOV, MKV, etc.
- **Image Files**: JPG, PNG, BMP, etc.

### Settings
- **Confidence Threshold**: Adjust detection sensitivity (0.1 - 1.0)
- **Camera Index**: Select camera device
- **Real-time Results**: Live object counting and bounding boxes

## 🔧 Troubleshooting

### Camera Issues
- Try different camera indices (0, 1, 2, etc.)
- Check camera permissions in Windows
- Ensure camera is not used by other applications

### IP Camera Issues
- Verify the stream URL format
- Test connection before starting detection
- Check network connectivity

### Performance Issues
- Lower confidence threshold for faster detection
- Close other applications to free up memory
- Use smaller video files for testing

## 📞 Support

This is an MVP (Minimum Viable Product) for testing and demonstration purposes.

## 📄 License

This application uses the same license as the parent YOLOv8 project.
"""
    
    readme_path = Path(__file__).parent / "dist" / "README.txt"
    with open(readme_path, 'w') as f:
        f.write(readme_content)
    
    print(f"📖 README created: {readme_path}")

if __name__ == "__main__":
    print("YOLOv8 Desktop App - Executable Builder")
    print("=" * 50)
    
    if build_executable():
        create_readme()
        print("\n🎊 All done! Your MVP is ready for distribution!")
    else:
        print("\n❌ Build failed. Please check the errors above.")
        sys.exit(1) 