from flask import Flask, render_template, Response, jsonify
import cv2
from picamera2 import Picamera2
from ultralytics import YOLO
import serial
import time
import threading
import webbrowser

app = Flask(__name__)

# Global variables
target_detected = False
current_pan = 90.0
current_tilt = 50.0
fps_value = 0.0
manual_mode = False  # Toggle for manual control
lock = threading.Lock()

# UART Setup
try:
    ser = serial.Serial('/dev/serial0', 115200, timeout=1)
    time.sleep(2)
    uart_enabled = True
    # set servo to 90 50
    ser.write("pan=90\n".encode())
    ser.write("tilt=50\n".encode())
    time.sleep(1)
    print(f"camera initialized")
except:
    uart_enabled = False
    print("Warning: UART not available")

# Camera Setup
picam2 = Picamera2()
camera_config = picam2.create_preview_configuration(
    main={"size": (1920, 1080), "format": "RGB888"},
    controls={"FrameRate": 30}
)
picam2.configure(camera_config)
picam2.start()
time.sleep(2)

frame_width = 1920
frame_height = 1080
center_x = frame_width // 2
center_y = frame_height // 2

# Detection resolution (downscaled for faster YOLO processing)
# Lower = faster but less accurate. 320x180 is ~4x faster than 640x360
detect_width = 416
detect_height = 234
scale_x = frame_width / detect_width
scale_y = frame_height / detect_height

# Servo Parameters
PAN_MIN, PAN_MAX = 0, 180
TILT_MIN, TILT_MAX = 0, 180

MAX_STEP = 45.0 # max degrees per frame
SMOOTHING = 0.6  # 0=no smoothing (instant jump), 1=full smoothing (slow)

# Zone Settings
INNER_DEAD_ZONE = 70  # Stable zone - no movement (increased from 35)
OUTER_TRIGGER_ZONE = 150  # Trigger zone - movement starts here (increased from 80)

# PID Parameters (pixel-based control)
Kp = 0.03   # Reduced to prevent overshoot
Ki = 0.00 # no integral gain
Kd = 0.05   # Damping to prevent overshoot
MAX_INTEGRAL = 500  # Anti-windup: clamp integral accumulation
pan_integral = 0.0
tilt_integral = 0.0
pan_last_error = 0.0
tilt_last_error = 0.0

# Load YOLO with optimized settings
model = YOLO("yolov8n_ncnn_model")
CONF_THRESHOLD = 0.4  # Minimum confidence to process (higher = faster)
IOU_THRESHOLD = 0.5   # NMS IoU threshold
TARGET_CLASS = 67 # default target class is cell phone

# COCO class names
COCO_CLASSES = {
    0: "person", 1: "bicycle", 2: "car", 3: "motorcycle", 4: "airplane",
    5: "bus", 6: "train", 7: "truck", 8: "boat", 9: "traffic light",
    10: "fire hydrant", 11: "stop sign", 12: "parking meter", 13: "bench", 14: "bird",
    15: "cat", 16: "dog", 17: "horse", 18: "sheep", 19: "cow",
    20: "elephant", 21: "bear", 22: "zebra", 23: "giraffe", 24: "backpack",
    25: "umbrella", 26: "handbag", 27: "tie", 28: "suitcase", 29: "frisbee",
    30: "skis", 31: "snowboard", 32: "sports ball", 33: "kite", 34: "baseball bat",
    35: "baseball glove", 36: "skateboard", 37: "surfboard", 38: "tennis racket", 39: "bottle",
    40: "wine glass", 41: "cup", 42: "fork", 43: "knife", 44: "spoon",
    45: "bowl", 46: "banana", 47: "apple", 48: "sandwich", 49: "orange",
    50: "broccoli", 51: "carrot", 52: "hot dog", 53: "pizza", 54: "donut",
    55: "cake", 56: "chair", 57: "couch", 58: "potted plant", 59: "bed",
    60: "dining table", 61: "toilet", 62: "tv", 63: "laptop", 64: "mouse",
    65: "remote", 66: "keyboard", 67: "cell phone", 68: "microwave", 69: "oven",
    70: "toaster", 71: "sink", 72: "refrigerator", 73: "book", 74: "clock",
    75: "vase", 76: "scissors", 77: "teddy bear", 78: "hair drier", 79: "toothbrush"
}

# Frame skipping (skip detection on some frames for higher FPS)
# SKIP_FRAMES=2 means detect every 2nd frame, doubles streaming FPS
frame_count = 0
SKIP_FRAMES = 2
last_results = None
last_target_found = False
last_best_x, last_best_y = 0, 0

def process_frame():
    global target_detected, current_pan, current_tilt, fps_value
    global frame_count, last_results, last_target_found, last_best_x, last_best_y
    global pan_integral, tilt_integral, pan_last_error, tilt_last_error
    
    # Capture full resolution frame
    frame_full = picam2.capture_array()
    is_processing = frame_count % SKIP_FRAMES == 0
    
    if is_processing:
        # Downscale for faster YOLO detection
        frame_small = cv2.resize(frame_full, (detect_width, detect_height), interpolation=cv2.INTER_LINEAR)
        results = model(frame_small, conf=CONF_THRESHOLD, iou=IOU_THRESHOLD, verbose=False)
        last_results = results
    else:
        results = last_results
    
    frame_count += 1
    
    # Use full resolution frame for display (don't use results[0].plot() as it's wrong size)
    annotated_frame = frame_full.copy()
    
    target_found = False
    closest_distance = float('inf')
    best_box_width = 0
    best_box_height = 0
    
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
                    # Save box size for dynamic dead zone
                    best_box_width = box_x2 - box_x1
                    best_box_height = box_y2 - box_y1
                    
                    cv2.rectangle(annotated_frame, (box_x1, box_y1), (box_x2, box_y2), (0, 255, 0), 3)
                    
                    # Draw confidence on top of the box
                    conf_text = f"{confidence:.2f}"
                    cv2.putText(annotated_frame, conf_text, (box_x1, box_y1 - 10),
                               cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 255, 0), 2, cv2.LINE_AA)
        
        if is_processing:
            if target_found:
                last_target_found = True
                last_best_x, last_best_y = best_x, best_y
            else:
                last_target_found = False
    
    if not target_found and last_target_found and not is_processing:
        target_found = True
        best_x, best_y = last_best_x, last_best_y
    
    with lock:
        target_detected = target_found
    
    # Only track if not in manual mode
    if target_found and not manual_mode:
        error_x_pixels = best_x - center_x
        error_y_pixels = best_y - center_y
        
        # Track zone: instead of a single point, use a range (20% of object size)
        # This means: if any part of the track zone is in dead zone, consider it centered
        TRACK_ZONE_RATIO = 0.20  # 20% of object size
        track_zone_x = int(best_box_width * TRACK_ZONE_RATIO / 2)
        track_zone_y = int(best_box_height * TRACK_ZONE_RATIO / 2)
        
        # Effective error: reduce error by track zone size (more forgiving for large objects)
        effective_error_x = max(0, abs(error_x_pixels) - track_zone_x) * (1 if error_x_pixels >= 0 else -1)
        effective_error_y = max(0, abs(error_y_pixels) - track_zone_y) * (1 if error_y_pixels >= 0 else -1)
        
        # Dynamic dead zone based on object size
        # LARGER objects (closer) = LARGER dead zone (more tolerance)
        # Because small servo movements cause big changes when object is close
        object_size_ratio = (best_box_width * best_box_height) / (frame_width * frame_height)
        # Smaller base, still scales with size: 5% = 0.58x, 10% = 0.65x, 20% = 0.8x, 30% = 0.95x
        size_factor = 0.5 + object_size_ratio * 1.5
        
        dynamic_inner_dead = int(INNER_DEAD_ZONE * size_factor)
        dynamic_outer_trigger = int(OUTER_TRIGGER_ZONE * size_factor)
        
        pan_step = 0.0
        tilt_step = 0.0
        
        # Use effective error (reduced by track zone) for dead zone comparison
        if abs(effective_error_x) < dynamic_inner_dead:
            pan_step = 0.0
            pan_integral = 0.0
            pan_last_error = 0.0
        elif abs(effective_error_x) >= dynamic_outer_trigger:
            pan_integral += effective_error_x
            pan_integral = max(min(pan_integral, MAX_INTEGRAL), -MAX_INTEGRAL)  # Anti-windup
            pan_derivative = effective_error_x - pan_last_error
            pan_step = -(Kp * effective_error_x + Ki * pan_integral + Kd * pan_derivative)
            pan_step = max(min(pan_step, MAX_STEP), -MAX_STEP)
            pan_last_error = effective_error_x
        
        if abs(effective_error_y) < dynamic_inner_dead:
            tilt_step = 0.0
            tilt_integral = 0.0
            tilt_last_error = 0.0
        elif abs(effective_error_y) >= dynamic_outer_trigger:
            tilt_integral += effective_error_y
            tilt_integral = max(min(tilt_integral, MAX_INTEGRAL), -MAX_INTEGRAL)  # Anti-windup
            tilt_derivative = effective_error_y - tilt_last_error
            tilt_step = (Kp * effective_error_y + Ki * tilt_integral + Kd * tilt_derivative)
            tilt_step = max(min(tilt_step, MAX_STEP), -MAX_STEP)
            tilt_last_error = effective_error_y
        
        with lock:
            current_pan = current_pan * (1 - SMOOTHING) + (current_pan + pan_step) * SMOOTHING
            current_tilt = current_tilt * (1 - SMOOTHING) + (current_tilt + tilt_step) * SMOOTHING
            current_pan = max(min(current_pan, PAN_MAX), PAN_MIN)
            current_tilt = max(min(current_tilt, TILT_MAX), TILT_MIN)
        
        # Debug: Show PID calculation
        cv2.putText(annotated_frame, f"Step: pan={pan_step:.2f} tilt={tilt_step:.2f}", 
                   (10, 140), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 0), 2, cv2.LINE_AA)
        cv2.putText(annotated_frame, f"Servo: pan={int(current_pan)} tilt={int(current_tilt)}", 
                   (10, 170), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 0), 2, cv2.LINE_AA)
        
        if uart_enabled:
            ser.write(f"pan={int(current_pan)}\n".encode())
            ser.write(f"tilt={int(current_tilt)}\n".encode())
            print(f"UART发送: pan={int(current_pan)}, tilt={int(current_tilt)}, step=({pan_step:.2f}, {tilt_step:.2f})")
    
    if target_found:
        cv2.circle(annotated_frame, (best_x, best_y), 12, (0, 0, 255), -1)
        # Add tracking mode indicator
        track_mode = "HEAD" if TARGET_CLASS == 0 else "CENTER"
        cv2.putText(annotated_frame, track_mode, (best_x - 30, best_y - 20),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2, cv2.LINE_AA)
    
    # Add status indicator
    status_color = (0, 255, 0) if target_found else (0, 0, 255)
    status_text = "TARGET DETECTED" if target_found else "SEARCHING..."
    cv2.putText(annotated_frame, status_text, (10, 40),
                cv2.FONT_HERSHEY_SIMPLEX, 1.2, status_color, 2, cv2.LINE_AA)
    
    # Debug info: Show error distance and tracking status
    if target_found:
        error_x = best_x - center_x
        error_y = best_y - center_y
        error_distance = int(((error_x)**2 + (error_y)**2)**0.5)
        
        # Track zone calculation for display
        TRACK_ZONE_RATIO = 0.20
        track_zone_x = int(best_box_width * TRACK_ZONE_RATIO / 2)
        track_zone_y = int(best_box_height * TRACK_ZONE_RATIO / 2)
        effective_error_x = max(0, abs(error_x) - track_zone_x) * (1 if error_x >= 0 else -1)
        effective_error_y = max(0, abs(error_y) - track_zone_y) * (1 if error_y >= 0 else -1)
        
        # Calculate dynamic dead zones for display (larger object = larger dead zone)
        object_size_ratio = (best_box_width * best_box_height) / (frame_width * frame_height)
        size_factor = 0.5 + object_size_ratio * 1.5
        dynamic_inner_dead = int(INNER_DEAD_ZONE * size_factor)
        dynamic_outer_trigger = int(OUTER_TRIGGER_ZONE * size_factor)
        
        # Determine tracking status using effective error and dynamic zones
        if manual_mode:
            status = "MANUAL MODE"
            debug_color = (255, 255, 0)
        elif abs(effective_error_x) < dynamic_inner_dead and abs(effective_error_y) < dynamic_inner_dead:
            status = "IN DEAD ZONE"
            debug_color = (0, 255, 0)
        elif abs(effective_error_x) >= dynamic_outer_trigger or abs(effective_error_y) >= dynamic_outer_trigger:
            status = "TRACKING"
            debug_color = (0, 255, 255)
        else:
            status = "IN NEUTRAL ZONE"
            debug_color = (255, 165, 0)
        
        cv2.putText(annotated_frame, f"Raw Error: X={error_x} Y={error_y}", 
                   (10, 80), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2, cv2.LINE_AA)
        cv2.putText(annotated_frame, f"Effective: X={int(effective_error_x)} Y={int(effective_error_y)} (zone={track_zone_x},{track_zone_y})", 
                   (10, 110), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 200, 0), 2, cv2.LINE_AA)
        cv2.putText(annotated_frame, f"Status: {status}", 
                   (10, 140), cv2.FONT_HERSHEY_SIMPLEX, 0.8, debug_color, 2, cv2.LINE_AA)
        cv2.putText(annotated_frame, f"Size: {best_box_width}x{best_box_height} ({object_size_ratio*100:.1f}%)", 
                   (10, 200), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 0), 2, cv2.LINE_AA)
        cv2.putText(annotated_frame, f"Dead Zones: Inner={dynamic_inner_dead} Outer={dynamic_outer_trigger}", 
                   (10, 230), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 0), 2, cv2.LINE_AA)
        
        # Draw track zone rectangle around the tracking point (cyan rectangle)
        cv2.rectangle(annotated_frame, 
                     (best_x - track_zone_x, best_y - track_zone_y),
                     (best_x + track_zone_x, best_y + track_zone_y),
                     (255, 255, 0), 2)  # Cyan rectangle for track zone
    
    # Draw dead zones (dynamic if target found, static otherwise)
    if target_found:
        # Use dynamic zones based on object size (larger object = larger dead zone)
        object_size_ratio = (best_box_width * best_box_height) / (frame_width * frame_height)
        size_factor = 0.5 + object_size_ratio * 1.5
        display_inner = int(INNER_DEAD_ZONE * size_factor)
        display_outer = int(OUTER_TRIGGER_ZONE * size_factor)
    else:
        # Use static zones when no target
        display_inner = INNER_DEAD_ZONE
        display_outer = OUTER_TRIGGER_ZONE
    
    # Inner dead zone (green rectangle) - no movement inside this zone
    inner_x1 = center_x - display_inner
    inner_y1 = center_y - display_inner
    inner_x2 = center_x + display_inner
    inner_y2 = center_y + display_inner
    cv2.rectangle(annotated_frame, (inner_x1, inner_y1), (inner_x2, inner_y2), (0, 255, 0), 2)
    cv2.putText(annotated_frame, f"DEAD {display_inner}px", (inner_x1, inner_y1 - 5),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1, cv2.LINE_AA)
    
    # Outer trigger zone (yellow rectangle) - movement triggers outside this zone
    outer_x1 = center_x - display_outer
    outer_y1 = center_y - display_outer
    outer_x2 = center_x + display_outer
    outer_y2 = center_y + display_outer
    cv2.rectangle(annotated_frame, (outer_x1, outer_y1), (outer_x2, outer_y2), (0, 255, 255), 2)
    cv2.putText(annotated_frame, f"TRIGGER {display_outer}px", (outer_x1, outer_y1 - 5),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 1, cv2.LINE_AA)
    
    # Draw center crosshair
    cv2.line(annotated_frame, (center_x - 20, center_y), (center_x + 20, center_y), (255, 255, 255), 1)
    cv2.line(annotated_frame, (center_x, center_y - 20), (center_x, center_y + 20), (255, 255, 255), 1)
    
    if results is not None and is_processing:
        inference_time = results[0].speed['inference']
        with lock:
            fps_value = 1000 / inference_time if inference_time > 0 else 0
    
    return annotated_frame

def generate_frames():
    # JPEG encoding params: quality 70 (default 95) for faster encoding
    encode_param = [int(cv2.IMWRITE_JPEG_QUALITY), 70]
    while True:
        frame = process_frame()
        ret, buffer = cv2.imencode('.jpg', frame, encode_param)
        frame_bytes = buffer.tobytes()
        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + frame_bytes + b'\r\n')

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/video_feed')
def video_feed():
    return Response(generate_frames(),
                    mimetype='multipart/x-mixed-replace; boundary=frame')

@app.route('/status')
def status():
    with lock:
        return jsonify({
            'target_detected': target_detected,
            'pan': round(current_pan, 1),
            'tilt': round(current_tilt, 1),
            'fps': round(fps_value, 1),
            'target_class': TARGET_CLASS,
            'target_name': COCO_CLASSES.get(TARGET_CLASS, "Unknown"),
            'manual_mode': manual_mode
        })

@app.route('/classes')
def get_classes():
    return jsonify(COCO_CLASSES)

@app.route('/set_target/<int:class_id>', methods=['POST'])
def set_target(class_id):
    global TARGET_CLASS
    if class_id in COCO_CLASSES:
        TARGET_CLASS = class_id
        return jsonify({'success': True, 'class_id': class_id, 'class_name': COCO_CLASSES[class_id]})
    return jsonify({'success': False, 'error': 'Invalid class ID'}), 400

@app.route('/toggle_manual', methods=['POST'])
def toggle_manual():
    global manual_mode
    manual_mode = not manual_mode
    return jsonify({'success': True, 'manual_mode': manual_mode})

@app.route('/manual_control', methods=['POST'])
def manual_control():
    global current_pan, current_tilt
    from flask import request
    data = request.get_json()
    
    if not manual_mode:
        return jsonify({'success': False, 'error': 'Not in manual mode'}), 400
    
    direction = data.get('direction')
    step = 5  # degrees per button press
    
    with lock:
        if direction == 'left':
            current_pan = min(current_pan + step, PAN_MAX)
        elif direction == 'right':
            current_pan = max(current_pan - step, PAN_MIN)
        elif direction == 'up':
            current_tilt = max(current_tilt - step, TILT_MIN)
        elif direction == 'down':
            current_tilt = min(current_tilt + step, TILT_MAX)
        elif direction == 'center':
            current_pan = 90.0
            current_tilt = 50.0
        
        if uart_enabled:
            ser.write(f"pan={int(current_pan)}\n".encode())
            ser.write(f"tilt={int(current_tilt)}\n".encode())
    
    return jsonify({'success': True, 'pan': current_pan, 'tilt': current_tilt})

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000, threaded=True)
