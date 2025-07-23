# YOLO People Counter Web Application

A modern web application that integrates with your YOLO API to count people passing through a specific area. The application provides a user-friendly interface for selecting different YOLO APIs, configuring detection parameters, and monitoring people counting in real-time.

## Features

- **Multiple API Support**: Select from different YOLO APIs (person detection, car detection, image/video/stream)
- **Real-time People Counting**: Track and count people passing through a defined area
- **Modern UI**: Responsive design with real-time updates and status indicators
- **Configuration Options**: Adjustable confidence thresholds and counting line positions
- **Activity Logging**: Comprehensive logging of all operations and events
- **Connection Testing**: Built-in API connection testing functionality

## Prerequisites

- Python 3.8 or higher
- Your YOLO API running on `http://localhost:5000`
- Required Python packages (see requirements.txt)

## Installation

1. **Navigate to the webapp directory**:
   ```bash
   cd code/webapp
   ```

2. **Install dependencies**:
   ```bash
   pip install -r requirements.txt
   ```

3. **Start your YOLO API** (if not already running):
   ```bash
   cd ../API
   python api.py
   ```

4. **Start the web application**:
   ```bash
   python app.py
   ```

5. **Access the application**:
   Open your browser and go to `http://localhost:5001`

## Usage

### 1. API Configuration
- Select the desired YOLO API from the dropdown menu
- Adjust the confidence threshold using the slider (0.1 to 1.0)
- Test the connection to ensure your YOLO API is accessible

### 2. People Counting
- **Start Counting**: Begin the people counting process
- **Stop Counting**: Pause the counting process
- **Reset Counter**: Reset the total count to zero
- **Real-time Updates**: View current and total counts in real-time

### 3. Dynamic Video Source Detection
- **Camera Detection**: Select from available camera devices
- **Video File Detection**: Use local video files (MP4, AVI, MOV, etc.)
- **IP Camera Detection**: Connect to RTSP, HTTP, RTMP streams
- **YouTube Detection**: Process YouTube videos and live streams
- **Custom Sources**: Support for custom video input sources
- **Real-time Streaming**: Live detection with dynamic source switching
- **Source Validation**: Built-in validation and testing for video sources

### 4. Monitoring
- **Statistics**: View current counting status and line position
- **Activity Log**: Monitor all operations and events
- **Status Indicators**: Visual indicators for connection and counting status

## API Endpoints

The web application provides the following API endpoints:

### Core Endpoints
- `GET /` - Main application interface
- `GET /api/available-apis` - Get list of available YOLO APIs
- `GET /api/test-connection` - Test connection to YOLO API

### Detection Endpoints
- `POST /api/detect` - Generic detection endpoint that routes to specific YOLO APIs
- `POST /api/detect-stream` - Stream detection with dynamic video sources

### Video Source Endpoints
- `GET /api/video-sources/cameras` - Get list of available camera devices
- `POST /api/video-sources/validate` - Validate a video input source
- `POST /api/video-sources/test` - Test if a video source is accessible
- `GET /api/video-sources/examples` - Get examples of different video sources

### People Counter Endpoints
- `GET /api/people-counter/start` - Start people counting
- `GET /api/people-counter/stop` - Stop people counting
- `GET /api/people-counter/reset` - Reset counter
- `GET /api/people-counter/status` - Get current counter status
- `POST /api/people-counter/set-line` - Set counting line position
- `GET /api/stream-detection` - Real-time detection stream

## Configuration

### API Base URL
Edit the `API_BASE_URL` variable in `app.py` to point to your YOLO API:
```python
API_BASE_URL = "http://localhost:5000"  # Change this to your API URL
```

### Available APIs
The application supports the following YOLO APIs:
- `person_detect_image` - Person detection in images
- `person_detect_video` - Person detection in videos
- `person_detect_video_stream` - Person detection in video streams
- `car_detect_image` - Car detection in images
- `car_detect_video` - Car detection in videos
- `car_detect_video_stream` - Car detection in video streams

### Supported Video Sources
- **Cameras**: USB cameras, built-in webcams (indices 0, 1, 2, etc.)
- **Video Files**: MP4, AVI, MOV, MKV, FLV, WMV, WebM
- **IP Cameras**: RTSP, HTTP, HTTPS, RTMP, UDP streams
- **YouTube**: Videos and live streams
- **Custom Sources**: Any OpenCV-compatible video source

## Troubleshooting

### Common Issues

1. **Connection Failed**
   - Ensure your YOLO API is running on the correct port
   - Check if the API_BASE_URL is correct
   - Verify network connectivity

2. **Detection Errors**
   - Check if the image/video path is correct and accessible
   - Ensure the selected API supports the input type
   - Verify the confidence threshold is appropriate

3. **People Counting Not Working**
   - Make sure you've started the counting process
   - Check if the selected API is for person detection
   - Verify the detection is returning person counts

### Logs
Check the activity log in the web interface for detailed error messages and operation status.

## Customization

### Adding New APIs
To add support for new YOLO APIs:

1. Add the API endpoint to the `AVAILABLE_APIS` dictionary in `app.py`
2. Update the detection logic in the `detect_objects()` function
3. Add corresponding UI elements in `index.html`

### Modifying the UI
The application uses Bootstrap 5 and custom CSS. You can modify the styling by editing the `<style>` section in `index.html`.

### Changing the Counting Logic
Modify the `update_people_counter()` function in `app.py` to implement different counting algorithms or tracking methods.

## Security Considerations

- The application currently runs without authentication
- Consider adding authentication for production use
- Validate and sanitize all user inputs
- Implement rate limiting for API endpoints
- Use HTTPS in production environments

## Performance Optimization

- The application uses Server-Sent Events for real-time updates
- Consider implementing caching for frequently accessed data
- Optimize image/video processing for large files
- Implement connection pooling for API requests

## License

This project is part of your YOLO custom object detection system. 