
import cv2

def main():
    # Replace with your RTSP URL
    rtsp_url = "rtsp://184.72.239.149/vod/mp4:BigBuckBunny_115k.mov"

    # Open a connection to the RTSP stream
    cap = cv2.VideoCapture(rtsp_url)

    if not cap.isOpened():
        print("Error: Could not open video stream")
        return

    while True:
        ret, frame = cap.read()
        if not ret:
            print("Error: Could not read frame")
            break

        # Display the frame
        cv2.imshow('RTSP Stream', frame)

        # Exit on 'q' key press
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    # Release the video capture object and close all OpenCV windows
    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
