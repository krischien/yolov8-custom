# YOLOv8 Desktop Application

A modern desktop GUI application for YOLOv8 object detection with support for camera, video files, and images.

## Features

- **Real-time Camera Detection**: Live object detection from webcam
- **Video File Processing**: Process video files with detection overlay
- **Image File Processing**: Single image detection and analysis
- **Dual Model Support**: Person detection and car detection models
- **Adjustable Confidence**: Real-time confidence threshold adjustment
- **Modern GUI**: Dark theme with intuitive controls
- **Performance Optimized**: Frame skipping and efficient processing

## Installation

1. **Install Dependencies**:
   ```bash
   pip install -r requirements.txt
   ```

2. **Ensure Models are Available**:
   - Person detection model: `../code/API/yolo11m.pt`
   - Car detection model: `../code/API/models/best.pt`

## Usage

### Running the Application

```bash
python main.py
```

### Controls

#### Detection Type
- **Person Detection**: Uses YOLOv8 person detection model
- **Car Detection**: Uses custom car detection model

#### Input Source
- **Camera**: Live webcam feed (select camera index)
- **Video File**: Process video files (.mp4, .avi, .mov, etc.)
- **Image File**: Process single images (.jpg, .png, .bmp, etc.)

#### Settings
- **Confidence Threshold**: Adjust detection sensitivity (0.1 - 1.0)
- **Camera Index**: Select camera device (usually 0 for built-in webcam)

### Workflow

1. **Select Detection Type**: Choose between person or car detection
2. **Choose Input Source**: Camera, video file, or image file
3. **Configure Settings**: Adjust confidence threshold and camera index
4. **Start Detection**: Click "Start Detection" to begin processing
5. **View Results**: Real-time count updates and bounding box overlays
6. **Stop Detection**: Click "Stop Detection" when finished

## File Structure

```
desktop_app/
├── main.py              # Main application file
├── requirements.txt     # Python dependencies
├── README.md           # This file
└── __pycache__/        # Python cache (auto-generated)
```

## Troubleshooting

### Camera Issues
- **Camera not accessible**: Try different camera indices (0, 1, 2, etc.)
- **No video feed**: Check camera permissions and drivers
- **Poor performance**: Lower confidence threshold or use video files

### Model Issues
- **Model not found**: Ensure model files exist in the correct paths
- **Detection errors**: Check model compatibility and file integrity

### Performance Issues
- **Slow processing**: Reduce confidence threshold or use frame skipping
- **High memory usage**: Close other applications or use smaller video files

## Dependencies

- **customtkinter**: Modern GUI framework
- **opencv-python**: Computer vision and camera handling
- **Pillow**: Image processing
- **ultralytics**: YOLOv8 model inference
- **torch**: PyTorch backend
- **numpy**: Numerical computing

## System Requirements

- **OS**: Windows 10/11, macOS, or Linux
- **Python**: 3.8 or higher
- **RAM**: 4GB minimum, 8GB recommended
- **GPU**: Optional (CUDA support for faster inference)
- **Camera**: Webcam for live detection (optional)

## License

This project uses the same license as the parent YOLOv8 project. 