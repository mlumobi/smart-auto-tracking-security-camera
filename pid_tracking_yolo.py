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
    main={"size": (1920, 1080), "format": "RGB888"},
    controls={"FrameRate": 30}
)
picam2.configure(camera_config)
picam2.start()

frame_width = 1920
frame_height = 1080
center_x = frame_width // 2
center_y = frame_height // 2

# Detection resolution (downscaled for faster YOLO processing)
# Lower = faster but less accurate. 416x234 is good balance
detect_width = 416
detect_height = 234
scale_x = frame_width / detect_width
scale_y = frame_height / detect_height

# YOLO inference settings
CONF_THRESHOLD = 0.4  # Minimum confidence
IOU_THRESHOLD = 0.5   # NMS IoU threshold

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
INNER_DEAD_ZONE = 70     # Stable zone - no movement (increased from 35)
OUTER_TRIGGER_ZONE = 150  # Trigger zone - movement starts here (increased from 80)

# -----------------------------
# PID Control Parameters (pixel-based control)
# -----------------------------
Kp = 0.03   # Reduced to prevent overshoot
Ki = 0.001  # Small integral gain to eliminate steady-state error
Kd = 0.02   # Damping to prevent overshoot
MAX_INTEGRAL = 500  # Anti-windup: clamp integral accumulation
pan_integral = 0.0
tilt_integral = 0.0
pan_last_error = 0.0
tilt_last_error = 0.0

# -----------------------------
# Load YOLO
# -----------------------------
model = YOLO("yolov8n.pt")
TARGET_CLASS = 67  # cell phone (COCO dataset)

# Frame skipping for performance (increased for full resolution)
frame_count = 0
SKIP_FRAMES = 2  # Detect every 2nd frame for balance of FPS and tracking
last_results = None
last_target_found = False
last_best_x, last_best_y = 0, 0

while True:
    # Capture full resolution frame
    frame_full = picam2.capture_array()
    
    # Check if this frame should be processed
    is_processing = frame_count % SKIP_FRAMES == 0
    
    # Process YOLO only every SKIP_FRAMES frames
    if is_processing:
        # Downscale for faster YOLO detection
        frame_small = cv2.resize(frame_full, (detect_width, detect_height), interpolation=cv2.INTER_LINEAR)
        results = model(frame_small, conf=CONF_THRESHOLD, iou=IOU_THRESHOLD, verbose=False)
        last_results = results
    else:
        results = last_results
    
    frame_count += 1
    
    # Use full resolution frame for display
    annotated_frame = frame_full.copy()

    target_found = False
    closest_distance = float('inf')

    # Find closest phone to center
    if results is not None and results[0].boxes is not None and len(results[0].boxes) > 0:
        for box, cls, conf in zip(results[0].boxes.xyxy, results[0].boxes.cls, results[0].boxes.conf):
            if int(cls) == TARGET_CLASS:
                x1, y1, x2, y2 = box
                confidence = float(conf)
                # Scale coordinates back to full resolution
                obj_x = int((x1 + x2) / 2 * scale_x)
                
                # For person class, target the top of the box (head area)
                if int(cls) == 0:  # person class
                    # Use top 20% of the box (head/upper body area)
                    obj_y = int((y1 + (y2 - y1) * 0.2) * scale_y)
                else:
                    # For other objects, use center
                    obj_y = int((y1 + y2) / 2 * scale_y)

                distance = ((obj_x - center_x)**2 + (obj_y - center_y)**2)**0.5
                if distance < closest_distance:
                    closest_distance = distance
                    best_x, best_y = obj_x, obj_y
                    target_found = True
                    # Draw bounding box on full resolution frame
                    box_x1 = int(x1 * scale_x)
                    box_y1 = int(y1 * scale_y)
                    box_x2 = int(x2 * scale_x)
                    box_y2 = int(y2 * scale_y)
                    cv2.rectangle(annotated_frame, (box_x1, box_y1), (box_x2, box_y2), (0, 255, 0), 3)
                    
                    # Draw confidence on top of the box
                    conf_text = f"{confidence:.2f}"
                    cv2.putText(annotated_frame, conf_text, (box_x1, box_y1 - 10),
                               cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 255, 0), 2, cv2.LINE_AA)
        
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
    cv2.putText(annotated_frame, f'{skip_text}', (10, 140),
                cv2.FONT_HERSHEY_SIMPLEX, 1, skip_color, 2, cv2.LINE_AA)
    
    cv2.putText(annotated_frame, f'Detections: {num_detections}', (10, 180),
                cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2, cv2.LINE_AA)
    
    if num_detections > 0 and results is not None:
        detected_classes = [int(c) for c in results[0].boxes.cls]
        cv2.putText(annotated_frame, f'Classes: {detected_classes}', (10, 220),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2, cv2.LINE_AA)
    
    cv2.putText(annotated_frame, f'Target Found: {target_found}', (10, 260),
                cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 255), 2, cv2.LINE_AA)

    if target_found:
        error_x_pixels = best_x - center_x
        error_y_pixels = best_y - center_y
        error_distance = int(((error_x_pixels)**2 + (error_y_pixels)**2)**0.5)
        
        # Show error and tracking status
        cv2.putText(annotated_frame, f'Error: X={error_x_pixels} Y={error_y_pixels} Dist={error_distance}', 
                   (10, 300), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2, cv2.LINE_AA)
        
        # Determine tracking status
        if abs(error_x_pixels) < INNER_DEAD_ZONE and abs(error_y_pixels) < INNER_DEAD_ZONE:
            status = "IN DEAD ZONE"
            status_color = (0, 255, 0)
        elif abs(error_x_pixels) >= OUTER_TRIGGER_ZONE or abs(error_y_pixels) >= OUTER_TRIGGER_ZONE:
            status = "TRACKING"
            status_color = (0, 255, 255)
        else:
            status = "IN NEUTRAL ZONE"
            status_color = (255, 165, 0)
        
        cv2.putText(annotated_frame, f'Status: {status}', 
                   (10, 340), cv2.FONT_HERSHEY_SIMPLEX, 0.8, status_color, 2, cv2.LINE_AA)

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
            pan_integral = max(min(pan_integral, MAX_INTEGRAL), -MAX_INTEGRAL)  # Anti-windup
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
            tilt_integral = max(min(tilt_integral, MAX_INTEGRAL), -MAX_INTEGRAL)  # Anti-windup
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
        
        # Debug output
        print(f"UART发送: pan={int(current_pan)}, tilt={int(current_tilt)}, step=({pan_step:.2f}, {tilt_step:.2f})")

        # Draw tracking dot
        cv2.circle(annotated_frame, (best_x, best_y), 12, (0, 0, 255), -1)
        # Add tracking mode indicator
        track_mode = "HEAD" if TARGET_CLASS == 0 else "CENTER"
        cv2.putText(annotated_frame, track_mode, (best_x - 30, best_y - 20),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2, cv2.LINE_AA)

        cv2.putText(annotated_frame, f'Pan: {current_pan:.1f}', (10, 60),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 0), 2, cv2.LINE_AA)
        cv2.putText(annotated_frame, f'Tilt: {current_tilt:.1f}', (10, 100),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 0), 2, cv2.LINE_AA)

    # FPS (only update when processing)
    if results is not None and is_processing:
        inference_time = results[0].speed['inference']
        fps = 1000 / inference_time
        cv2.putText(annotated_frame, f'FPS: {fps:.1f}', (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2, cv2.LINE_AA)
    
    # Draw dead zones (inner and outer trigger zones)
    # Inner dead zone (green rectangle) - no movement inside this zone
    inner_x1 = center_x - INNER_DEAD_ZONE
    inner_y1 = center_y - INNER_DEAD_ZONE
    inner_x2 = center_x + INNER_DEAD_ZONE
    inner_y2 = center_y + INNER_DEAD_ZONE
    cv2.rectangle(annotated_frame, (inner_x1, inner_y1), (inner_x2, inner_y2), (0, 255, 0), 2)
    cv2.putText(annotated_frame, "DEAD ZONE", (inner_x1, inner_y1 - 5),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1, cv2.LINE_AA)
    
    # Outer trigger zone (yellow rectangle) - movement triggers outside this zone
    outer_x1 = center_x - OUTER_TRIGGER_ZONE
    outer_y1 = center_y - OUTER_TRIGGER_ZONE
    outer_x2 = center_x + OUTER_TRIGGER_ZONE
    outer_y2 = center_y + OUTER_TRIGGER_ZONE
    cv2.rectangle(annotated_frame, (outer_x1, outer_y1), (outer_x2, outer_y2), (0, 255, 255), 2)
    cv2.putText(annotated_frame, "TRIGGER ZONE", (outer_x1, outer_y1 - 5),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 1, cv2.LINE_AA)
    
    # Draw center crosshair
    cv2.line(annotated_frame, (center_x - 20, center_y), (center_x + 20, center_y), (255, 255, 255), 1)
    cv2.line(annotated_frame, (center_x, center_y - 20), (center_x, center_y + 20), (255, 255, 255), 1)

    cv2.imshow("Camera", annotated_frame)
    if cv2.waitKey(1) == ord("q"):
        break

# Cleanup
cv2.destroyAllWindows()
ser.close()
