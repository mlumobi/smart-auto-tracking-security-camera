import cv2
from ultralytics import YOLO

# Initialize the YOLO model (YOLOv8n for lightweight; you can use v8s/v8m for higher accuracy)
model = YOLO("yolov8n.pt")  # Make sure yolov8n.pt is downloaded

# Open the IMX708 camera
cap = cv2.VideoCapture(0)  # cam0

# Set camera resolution (adjust as per IMX708 capability)
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)

if not cap.isOpened():
    print("Cannot open camera")
    exit()

while True:
    ret, frame = cap.read()
    if not ret:
        print("Failed to grab frame")
        break

    # Run YOLO detection
    results = model.predict(frame, conf=0.3, verbose=False)  # adjust confidence threshold

    # Draw boxes and coordinates
    for result in results:
        boxes = result.boxes.xyxy  # x1, y1, x2, y2
        scores = result.boxes.conf
        class_ids = result.boxes.cls
        for i, box in enumerate(boxes):
            x1, y1, x2, y2 = map(int, box)
            conf = float(scores[i])
            cls_id = int(class_ids[i])
            label = f"{model.names[cls_id]} {conf:.2f}"

            # Draw bounding box
            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
            # Show coordinates
            cv2.putText(frame, f"{label} ({x1},{y1})", (x1, y1-10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

    # Show frame
    cv2.imshow("YOLO Object Detection", frame)

    # Press 'q' to quit
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()