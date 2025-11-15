from picamera2 import Picamera2
import cv2
import time

def main():
    picam2 = Picamera2()

    config = picam2.create_video_configuration(
        main={"size": (1280, 720)},
        controls={"FrameDurationLimits": (33333, 33333)}
    )

    picam2.configure(config)

    # Fix blue tint
    picam2.set_controls({
        "AwbMode": "auto",
        "AwbEnable": True
    })

    picam2.start()
    time.sleep(0.2)

    while True:
        frame = picam2.capture_array()
        cv2.imshow("Arducam Live Stream", frame)

        if cv2.waitKey(1) == ord("q"):
            break

    picam2.stop()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()