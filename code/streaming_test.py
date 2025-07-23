import subprocess
import cv2

# Set RTSP streaming address (local machine)
RTSP_URL = "rtsp://localhost:8554/live"

# Start the RTSP server using FFmpeg
def start_rtsp_stream(video_path):
    cmd = [
        "ffmpeg",
        "-re",  # Read video in real-time
        "-i", video_path,  # Input video file
        "-vcodec", "libx264",  # Encode with H.264
        "-preset", "ultrafast",  # Low latency streaming
        "-tune", "zerolatency",
        "-f", "rtsp",  # Output format
        RTSP_URL  # Destination
    ]
    return subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE)

# Start the RTSP stream
video_path = "test_video.mp4"  # Change this to your video file
stream_process = start_rtsp_stream(video_path)

# Wait a few seconds for the stream to initialize
import time
time.sleep(5)

# OpenCV to read the RTSP stream
cap = cv2.VideoCapture(RTSP_URL)

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        print("Stream ended or error occurred.")
        break

    cv2.imshow("RTSP Stream (Simulated Drone Feed)", frame)

    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

cap.release()
cv2.destroyAllWindows()

# Stop the FFmpeg process when done
stream_process.terminate()
