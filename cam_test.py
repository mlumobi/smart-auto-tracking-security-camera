from picamera2 import Picamera2
import cv2
import time

def main():
    picam2 = Picamera2()

    # Configure video stream (1280x720 @ ~30 FPS)
    config = picam2.create_video_configuration(
        main={"size": (1280, 720)},
        controls={"FrameDurationLimits": (33333, 33333)}  # ~30 FPS
    )

    picam2.configure(config)
    picam2.start()
    time.sleep(0.1)  # Small delay to let camera stabilize

    while True:
        frame = picam2.capture_array()
        cv2.imshow("Arducam Live Stream", frame)

        if cv2.waitKey(1) == ord("q"):
            break

    picam2.stop()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()