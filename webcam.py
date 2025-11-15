from picamera2 import Picamera2
from http.server import BaseHTTPRequestHandler, HTTPServer
import threading
import time

picam2 = Picamera2()
picam2.configure(picam2.create_video_configuration(main={"size": (1280, 720)}))
picam2.start()

frame_lock = threading.Lock()
current_frame = None

# Capture frames continuously
def capture_frames():
    global current_frame
    while True:
        frame = picam2.capture_buffer("main")
        with frame_lock:
            current_frame = frame
        time.sleep(0.01)

class MJPEGHandler(BaseHTTPRequestHandler):
    def do_GET(self):
        if self.path != "/":
            self.send_error(404)
            return

        self.send_response(200)
        self.send_header("Content-Type", "multipart/x-mixed-replace; boundary=FRAME")
        self.end_headers()

        try:
            while True:
                with frame_lock:
                    frame = current_frame

                if frame is None:
                    continue

                self.wfile.write(b"--FRAME\r\n")
                self.send_header("Content-Type", "image/jpeg")
                self.send_header("Content-Length", len(frame))
                self.end_headers()
                self.wfile.write(frame)
                self.wfile.write(b"\r\n")

        except (BrokenPipeError, ConnectionResetError):
            print("Client disconnected")
            return

    def log_message(self, format, *args):
        return  # Disable logging

# Start capture thread
threading.Thread(target=capture_frames, daemon=True).start()

# Start server
server = HTTPServer(("0.0.0.0", 8000), MJPEGHandler)
print("MJPEG Stream ready at: http://<your_pi_ip>:8000")
print("View on Pi: http://localhost:8000")

server.serve_forever()