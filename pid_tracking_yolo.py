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
camera_config = picam2.create_preview_configuration(
    main={"size": (4608, 2592), "format": "RGB888"},
    controls={"FrameRate": 15}
)
picam2.configure(camera_config)
picam2.start()

frame_width = 4608
frame_height = 2592
center_x = frame_width // 2
center_y = frame_height // 2

# -----------------------------
# Servo Parameters
# -----------------------------
PAN_MIN, PAN_MAX = 0, 180
TILT_MIN, TILT_MAX = 0, 180
current_pan = 90.0
current_tilt = 90.0

MAX_STEP = 15.0     # max degrees per frame
SMOOTHING = 0.6     # smoothing factor (0=no move, 1=full move)

# -----------------------------
# ZONE SETTINGS
# -----------------------------
INNER_DEAD_ZONE = 25     # Completely stable zone (camera does not move)
OUTER_TRIGGER_ZONE = 60  # Only move when phone exceeds this distance

# -----------------------------
# PID Control Parameters (pixel-based control)
# -----------------------------
Kp = 0.05   # Aggressive response to pixel errors
Ki = 0.0    # Keep disabled
Kd = 0.02   # Strong damping to prevent overshoot
pan_integral = 0.0
tilt_integral = 0.0
pan_last_error = 0.0
tilt_last_error = 0.0

# -----------------------------
# Load YOLO
# -----------------------------
model = YOLO("yolov8n_ncnn_model")
TARGET_CLASS = 67  # cell phone (COCO dataset)

# Frame skipping for performance (increased for full resolution)
frame_count = 0
SKIP_FRAMES = 3
last_results = None
last_target_found = False
last_best_x, last_best_y = 0, 0

while True:
    frame = picam2.capture_array()
    
    # Check if this frame should be processed
    is_processing = frame_count % SKIP_FRAMES == 0
    
    # Process YOLO only every SKIP_FRAMES frames
    if is_processing:
        results = model(frame)
        last_results = results
    else:
        results = last_results
    
    frame_count += 1
    
    if results is not None:
        annotated_frame = results[0].plot()
    else:
        annotated_frame = frame.copy()

    target_found = False
    closest_distance = float('inf')

    # Find closest phone to center
    if results is not None and results[0].boxes is not None and len(results[0].boxes) > 0:
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
        
        # Update last known position only when processing new frame
        if is_processing:
            if target_found:
                last_target_found = True
                last_best_x, last_best_y = best_x, best_y
            else:
                last_target_found = False
    
    # Use last known position if target was found before (for skipped frames)
    if not target_found and last_target_found and not is_processing:
        target_found = True
        best_x, best_y = last_best_x, last_best_y

    # Debug info
    num_detections = len(results[0].boxes) if (results is not None and results[0].boxes is not None) else 0
    # Show if frame was skipped
    skip_text = "PROCESSING" if is_processing else "SKIPPED"
    skip_color = (0, 255, 0) if is_processing else (128, 128, 128)
    cv2.putText(annotated_frame, f'{skip_text}', (10, 120),
                cv2.FONT_HERSHEY_SIMPLEX, 0.8, skip_color, 2, cv2.LINE_AA)
    
    cv2.putText(annotated_frame, f'Detections: {num_detections}', (10, 150),
                cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2, cv2.LINE_AA)
    
    if num_detections > 0 and results is not None:
        detected_classes = [int(c) for c in results[0].boxes.cls]
        cv2.putText(annotated_frame, f'Classes: {detected_classes}', (10, 180),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2, cv2.LINE_AA)
    
    cv2.putText(annotated_frame, f'Target Found: {target_found}', (10, 210),
                cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 255), 2, cv2.LINE_AA)

    if target_found:
        error_x_pixels = best_x - center_x
        error_y_pixels = best_y - center_y

        pan_step = 0.0
        tilt_step = 0.0

        # -----------------------------
        # PAN (X-axis)
        # -----------------------------
        if abs(error_x_pixels) < INNER_DEAD_ZONE:
            # Full stability zone → do nothing
            pan_step = 0.0
            pan_integral = 0.0
            pan_last_error = 0.0

        elif abs(error_x_pixels) >= OUTER_TRIGGER_ZONE:
            # PID correction only outside trigger zone
            pan_integral += error_x_pixels
            pan_derivative = error_x_pixels - pan_last_error

            pan_step = -(Kp * error_x_pixels +
                         Ki * pan_integral +
                         Kd * pan_derivative)

            pan_step = max(min(pan_step, MAX_STEP), -MAX_STEP)
            pan_last_error = error_x_pixels

        # -----------------------------
        # TILT (Y-axis)
        # -----------------------------
        if abs(error_y_pixels) < INNER_DEAD_ZONE:
            tilt_step = 0.0
            tilt_integral = 0.0
            tilt_last_error = 0.0

        elif abs(error_y_pixels) >= OUTER_TRIGGER_ZONE:
            tilt_integral += error_y_pixels
            tilt_derivative = error_y_pixels - tilt_last_error

            tilt_step = (Kp * error_y_pixels +
                         Ki * tilt_integral +
                         Kd * tilt_derivative)

            tilt_step = max(min(tilt_step, MAX_STEP), -MAX_STEP)
            tilt_last_error = error_y_pixels

        # -----------------------------
        # Apply smoothing filter
        # -----------------------------
        current_pan = current_pan * (1 - SMOOTHING) + (current_pan + pan_step) * SMOOTHING
        current_tilt = current_tilt * (1 - SMOOTHING) + (current_tilt + tilt_step) * SMOOTHING

        # Clamp
        current_pan = max(min(current_pan, PAN_MAX), PAN_MIN)
        current_tilt = max(min(current_tilt, TILT_MAX), TILT_MIN)

        # UART Send
        ser.write(f"pan={int(current_pan)}\n".encode())
        ser.write(f"tilt={int(current_tilt)}\n".encode())

        # Draw tracking dot
        cv2.circle(annotated_frame, (best_x, best_y), 10, (0, 0, 255), -1)

        cv2.putText(annotated_frame, f'Pan: {current_pan:.1f}', (10, 60),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 0), 2, cv2.LINE_AA)
        cv2.putText(annotated_frame, f'Tilt: {current_tilt:.1f}', (10, 90),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 0), 2, cv2.LINE_AA)

    # FPS (only update when processing)
    if results is not None and is_processing:
        inference_time = results[0].speed['inference']
        fps = 1000 / inference_time
        cv2.putText(annotated_frame, f'FPS: {fps:.1f}', (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2, cv2.LINE_AA)

    cv2.imshow("Camera", annotated_frame)
    if cv2.waitKey(1) == ord("q"):
        break

# Cleanup
cv2.destroyAllWindows()
ser.close()
