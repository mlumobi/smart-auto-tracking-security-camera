from gpiozero import AngularServo
from time import sleep

servo = AngularServo(17, min_angle=0, max_angle=180)

try:
    while True:
        for angle in [0, 90, 180, 90]:
            servo.angle = angle
            print(f"Moved to {angle}°")
            sleep(0.5)
except KeyboardInterrupt:
    print("Exiting")