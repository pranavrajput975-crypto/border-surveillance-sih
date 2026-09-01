from ultralytics import YOLO
import cv2

# Load YOLO model
model = YOLO("../yolov8n.pt")

# Open video
video_path = "yolo/test_video_3.mp4"
cap = cv2.VideoCapture(video_path)

if not cap.isOpened():
    print("Error: Could not open video.")
    exit()

print("Border surveillance tracking started...")

while True:
    ret, frame = cap.read()

    if not ret:
        break

    # YOLO tracking
    results = model.track(
        frame,
        persist=True,
        classes=[0],
        tracker="bytetrack.yaml"
    )

    # Draw boxes + tracking IDs
    annotated_frame = results[0].plot()

    # Display
    cv2.imshow("Border Surveillance - YOLO Tracking", annotated_frame)

    # Press Q to quit
    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

cap.release()
cv2.destroyAllWindows()

print("Tracking stopped.")