from picamera2 import Picamera2
import cv2

def main():
    picam2 = Picamera2()

    # Configure preview stream
    config = picam2.create_video_configuration(
        main={"size": (1280, 720)},  # Change resolution if you want
        controls={"FrameDurationLimits": (33333, 33333)}  # ~30 FPS
    )
    picam2.configure(config)

    picam2.start()

    while True:
        frame = picam2.capture_array()
        cv2.imshow("Arducam Live Stream", frame)

        # Press 'q' to exit
        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

    cv2.destroyAllWindows()
    picam2.stop()

if __name__ == "__main__":
    main()