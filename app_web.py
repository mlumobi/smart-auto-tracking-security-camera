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
detect_width = 640
detect_height = 360
scale_x = frame_width / detect_width
scale_y = frame_height / detect_height

# Servo Parameters
PAN_MIN, PAN_MAX = 0, 180
TILT_MIN, TILT_MAX = 0, 180

MAX_STEP = 15.0
SMOOTHING = 0.6

# Zone Settings
INNER_DEAD_ZONE = 25
OUTER_TRIGGER_ZONE = 60

# PID Parameters (pixel-based control)
Kp = 0.03   # Reduced to prevent overshoot
Ki = 0.0    # Keep disabled
Kd = 0.03   # Damping to prevent overshoot
pan_integral = 0.0
tilt_integral = 0.0
pan_last_error = 0.0
tilt_last_error = 0.0

# Load YOLO
model = YOLO("yolov8n_ncnn_model")
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

# Frame skipping (increased for full resolution performance)
frame_count = 0
SKIP_FRAMES = 3
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
        frame_small = cv2.resize(frame_full, (detect_width, detect_height))
        results = model(frame_small)
        last_results = results
    else:
        results = last_results
    
    frame_count += 1
    
    # Use full resolution frame for display (don't use results[0].plot() as it's wrong size)
    annotated_frame = frame_full.copy()
    
    target_found = False
    closest_distance = float('inf')
    
    if results is not None and results[0].boxes is not None and len(results[0].boxes) > 0:
        for box, cls in zip(results[0].boxes.xyxy, results[0].boxes.cls):
            if int(cls) == TARGET_CLASS:
                x1, y1, x2, y2 = box
                # Scale coordinates back to full resolution
                obj_x = int((x1 + x2) / 2 * scale_x)
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
    
    if target_found:
        error_x_pixels = best_x - center_x
        error_y_pixels = best_y - center_y
        
        pan_step = 0.0
        tilt_step = 0.0
        
        if abs(error_x_pixels) < INNER_DEAD_ZONE:
            pan_step = 0.0
            pan_integral = 0.0
            pan_last_error = 0.0
        elif abs(error_x_pixels) >= OUTER_TRIGGER_ZONE:
            pan_integral += error_x_pixels
            pan_derivative = error_x_pixels - pan_last_error
            pan_step = -(Kp * error_x_pixels + Ki * pan_integral + Kd * pan_derivative)
            pan_step = max(min(pan_step, MAX_STEP), -MAX_STEP)
            pan_last_error = error_x_pixels
        
        if abs(error_y_pixels) < INNER_DEAD_ZONE:
            tilt_step = 0.0
            tilt_integral = 0.0
            tilt_last_error = 0.0
        elif abs(error_y_pixels) >= OUTER_TRIGGER_ZONE:
            tilt_integral += error_y_pixels
            tilt_derivative = error_y_pixels - tilt_last_error
            tilt_step = (Kp * error_y_pixels + Ki * tilt_integral + Kd * tilt_derivative)
            tilt_step = max(min(tilt_step, MAX_STEP), -MAX_STEP)
            tilt_last_error = error_y_pixels
        
        with lock:
            current_pan = current_pan * (1 - SMOOTHING) + (current_pan + pan_step) * SMOOTHING
            current_tilt = current_tilt * (1 - SMOOTHING) + (current_tilt + tilt_step) * SMOOTHING
            current_pan = max(min(current_pan, PAN_MAX), PAN_MIN)
            current_tilt = max(min(current_tilt, TILT_MAX), TILT_MIN)
        
        if uart_enabled:
            ser.write(f"pan={int(current_pan)}\n".encode())
            ser.write(f"tilt={int(current_tilt)}\n".encode())
        
        cv2.circle(annotated_frame, (best_x, best_y), 12, (0, 0, 255), -1)
    
    # Add status indicator
    status_color = (0, 255, 0) if target_found else (0, 0, 255)
    status_text = "TARGET DETECTED" if target_found else "SEARCHING..."
    cv2.putText(annotated_frame, status_text, (10, 40),
                cv2.FONT_HERSHEY_SIMPLEX, 1.2, status_color, 2, cv2.LINE_AA)
    
    if results is not None and is_processing:
        inference_time = results[0].speed['inference']
        with lock:
            fps_value = 1000 / inference_time if inference_time > 0 else 0
    
    return annotated_frame

def generate_frames():
    while True:
        frame = process_frame()
        ret, buffer = cv2.imencode('.jpg', frame)
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
            'target_name': COCO_CLASSES.get(TARGET_CLASS, "Unknown")
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

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000, threaded=True)
