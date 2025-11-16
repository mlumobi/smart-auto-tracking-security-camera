# camera.py
import cv2
from picamera2 import Picamera2
import time

class Camera:
    def __init__(self):
        self.picam2 = Picamera2()

        # Request 1920x1080 30fps
        config = self.picam2.create_video_configuration(
            main={"size": (1920, 1080)},
            controls={"FrameDurationLimits": (33333, 33333)}  # 30fps
        )

        self.picam2.configure(config)
        self.picam2.start()

        time.sleep(1)  # warm up

    def frames(self):
        while True:
            frame = self.picam2.capture_array()

            # Fix color (Picamera2 gives RGB → convert to BGR for JPEG encoding)
            frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)

            # Rotate 180° (upside down)
            frame = cv2.rotate(frame, cv2.ROTATE_180)
            # (remove above line if not needed)

            # Encode JPEG
            ret, jpeg = cv2.imencode(".jpg", frame, [cv2.IMWRITE_JPEG_QUALITY, 80])

            if ret:
                yield (
                    b"--frame\r\n"
                    b"Content-Type: image/jpeg\r\n\r\n" +
                    jpeg.tobytes() +
                    b"\r\n"
                )