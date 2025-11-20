from gpiozero import AngularServo
from time import sleep

servo_1 = AngularServo(12, min_angle=-90, max_angle=90, min_pulse_width=0.5/1000, max_pulse_width=2.5/1000)

def goto_angle(angle):
    print(f"Moving to angle: {angle}")
    servo_1.angle = angle
    sleep(1)

try:
    while True:
        angle = input("Which angle (-90 to 90)?:")
        set_angle = int(angle)
except KeyboardInterrupt:
    print("\nExiting program.")
