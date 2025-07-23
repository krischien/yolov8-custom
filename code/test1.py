import cv2

# Use a test RTSP stream
stream_url = ""

cap = cv2.VideoCapture(stream_url)

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        print("Stream disconnected")
        break

    cv2.imshow("RTSP Stream", frame)
    
    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

cap.release()
cv2.destroyAllWindows()