from picamera2 import Picamera2
import cv2
from ultralytics import YOLO
import numpy as np

# Initialize YOLOv8 (nano model for RPi)
model = YOLO("yolov8n.pt")  # Downloaded automatically if not present

# Initialize Picamera2
picam2 = Picamera2()
# Configure preview/resolution
camera_config = picam2.create_preview_configuration(main={"size": (1280, 720)})
picam2.configure(camera_config)
picam2.start()

while True:
    # Capture frame
    frame = picam2.capture_array()  # Returns numpy array in RGB
    frame_bgr = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)  # Convert to BGR for OpenCV

    # YOLO detection
    results = model.predict(frame_bgr, conf=0.3, verbose=False)

    for result in results:
        boxes = result.boxes.xyxy
        scores = result.boxes.conf
        class_ids = result.boxes.cls

        for i, box in enumerate(boxes):
            x1, y1, x2, y2 = map(int, box)
            conf = float(scores[i])
            cls_id = int(class_ids[i])
            label = f"{model.names[cls_id]} {conf:.2f}"

            # Center coordinates
            cx = (x1 + x2) // 2
            cy = (y1 + y2) // 2

            # Draw bounding box
            cv2.rectangle(frame_bgr, (x1, y1), (x2, y2), (0, 255, 0), 2)
            # Draw center point
            cv2.circle(frame_bgr, (cx, cy), 5, (0, 0, 255), -1)
            # Show label and coordinates
            cv2.putText(frame_bgr, f"{label} ({cx},{cy})", (x1, y1-10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

    # Display frame
    cv2.imshow("YOLOv8 Detection with Picamera2", frame_bgr)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

picam2.stop()
cv2.destroyAllWindows()