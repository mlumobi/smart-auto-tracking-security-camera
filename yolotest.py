import cv2
from ultralytics import YOLO

# Initialize YOLOv8 model (nano for RPi 5)
model = YOLO("yolov8n.pt")  # Download yolov8n.pt if not present

# Open IMX708 camera (video0)
cap = cv2.VideoCapture("/dev/video0")

# Set camera resolution
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

    # YOLOv8 detection
    results = model.predict(frame, conf=0.3, verbose=False)

    for result in results:
        boxes = result.boxes.xyxy  # x1, y1, x2, y2
        scores = result.boxes.conf
        class_ids = result.boxes.cls

        for i, box in enumerate(boxes):
            x1, y1, x2, y2 = map(int, box)
            conf = float(scores[i])
            cls_id = int(class_ids[i])
            label = f"{model.names[cls_id]} {conf:.2f}"

            # Calculate center coordinates
            cx = (x1 + x2) // 2
            cy = (y1 + y2) // 2

            # Draw rectangle
            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
            # Draw center point
            cv2.circle(frame, (cx, cy), 5, (0, 0, 255), -1)
            # Display label with center coordinates
            cv2.putText(frame, f"{label} ({cx},{cy})", (x1, y1 - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

    # Show frame
    cv2.imshow("YOLOv8 Object Detection", frame)

    # Press 'q' to quit
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()