from picamera2 import Picamera2
from http.server import BaseHTTPRequestHandler, HTTPServer
import io
import threading
import time

picam2 = Picamera2()
picam2.configure(picam2.create_video_configuration(main={"size": (1280, 720)}))
picam2.start()

frame_lock = threading.Lock()
frame = None

def capture_frames():
    global frame
    while True:
        with frame_lock:
            frame = picam2.capture_buffer("main")
        time.sleep(0.01)

class StreamHandler(BaseHTTPRequestHandler):
    def do_GET(self):
        if self.path != "/":
            self.send_error(404)
            return

        self.send_response(200)
        self.send_header("Content-Type", "multipart/x-mixed-replace; boundary=FRAME")
        self.end_headers()

        while True:
            with frame_lock:
                output = frame

            if output is None:
                continue

            self.wfile.write(b"--FRAME\r\n")
            self.send_header("Content-Type", "image/jpeg")
            self.send_header("Content-Length", len(output))
            self.end_headers()
            self.wfile.write(output)
            self.wfile.write(b"\r\n")

# Start capture thread
threading.Thread(target=capture_frames, daemon=True).start()

# Start HTTP server
server = HTTPServer(("0.0.0.0", 8000), StreamHandler)
print("Streaming at http://<your_pi_ip>:8000")
server.serve_forever()