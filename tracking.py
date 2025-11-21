import cv2
from picamera2 import Picamera2
from ultralytics import YOLO
import serial
import time

# -----------------------------
# UART Setup
# -----------------------------
ser = serial.Serial('/dev/serial0', 115200, timeout=1)
time.sleep(2)

# -----------------------------
# Camera Setup
# -----------------------------
picam2 = Picamera2()
picam2.preview_configuration.main.size = (1280, 720)  # full 720p
picam2.preview_configuration.main.format = "RGB888"
picam2.preview_configuration.align()
picam2.configure("preview")
picam2.start()

frame_width = 1280
frame_height = 720
center_x = frame_width // 2
center_y = frame_height // 2

# -----------------------------
# Servo Parameters
# -----------------------------
PAN_MIN, PAN_MAX = 0, 180
TILT_MIN, TILT_MAX = 0, 180
current_pan = 90
current_tilt = 90

CAMERA_H_FOV = 120  # degrees horizontal FOV
CAMERA_V_FOV = 90   # degrees vertical FOV
MAX_STEP = 5        # max degrees per frame

# -----------------------------
# Load YOLO
# -----------------------------
model = YOLO("yolov8n_ncnn_model")
TARGET_CLASS = 67  # cell phone

while True:
    frame = picam2.capture_array()  # full 720p frame
    results = model(frame)
    annotated_frame = results[0].plot()

    target_found = False
    closest_distance = float('inf')

    for box, cls in zip(results[0].boxes.xyxy, results[0].boxes.cls):
        if int(cls) == TARGET_CLASS:
            x1, y1, x2, y2 = box
            obj_x = int((x1 + x2) / 2)
            obj_y = int((y1 + y2) / 2)

            distance = ((obj_x - center_x)**2 + (obj_y - center_y)**2)**0.5
            if distance < closest_distance:
                closest_distance = distance
                best_x, best_y = obj_x, obj_y
                target_found = True

    if target_found:
        # Compute horizontal and vertical error as ratio (-1 to 1)
        error_x = (best_x - center_x) / (frame_width / 2)   # right = positive
        error_y = (best_y - center_y) / (frame_height / 2)  # down = positive

        # Map screen ratio to degrees (proportional control)
        pan_step = -error_x * (CAMERA_H_FOV / 2)  # invert pan direction
        tilt_step = error_y * (CAMERA_V_FOV / 2)

        # Limit max step for smooth movement
        pan_step = max(min(pan_step, MAX_STEP), -MAX_STEP)
        tilt_step = max(min(tilt_step, MAX_STEP), -MAX_STEP)

        # Update servo angles
        current_pan += pan_step
        current_tilt += tilt_step

        current_pan = max(min(current_pan, PAN_MAX), PAN_MIN)
        current_tilt = max(min(current_tilt, TILT_MAX), TILT_MIN)

        # Send angles via UART
        ser.write(f"pan={int(current_pan)}\n".encode())
        ser.write(f"tilt={int(current_tilt)}\n".encode())

        # Draw object center
        cv2.circle(annotated_frame, (best_x, best_y), 10, (0, 0, 255), -1)

        # Display current servo angles
        cv2.putText(annotated_frame, f'Pan: {int(current_pan)}', (10, 60),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 0), 2, cv2.LINE_AA)
        cv2.putText(annotated_frame, f'Tilt: {int(current_tilt)}', (10, 90),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 0), 2, cv2.LINE_AA)

    # Display FPS
    inference_time = results[0].speed['inference']
    fps = 1000 / inference_time
    cv2.putText(annotated_frame, f'FPS: {fps:.1f}', (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2, cv2.LINE_AA)

    # Show frame
    cv2.imshow("Camera", annotated_frame)
    if cv2.waitKey(1) == ord("q"):
        break

# Cleanup
cv2.destroyAllWindows()
ser.close()