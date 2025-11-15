import RPi.GPIO as GPIO
import time

# GPIO pin where the servo is connected
servo_pin = 18

# Set up GPIO
GPIO.setmode(GPIO.BCM)
GPIO.setup(servo_pin, GPIO.OUT)

# Set up PWM on the servo pin, 50Hz (standard for servos)
pwm = GPIO.PWM(servo_pin, 50)
pwm.start(0)

def set_angle(angle):
    # Convert angle (0-180) to duty cycle (2-12 for MG90S)
    duty = 2 + (angle / 18)
    GPIO.output(servo_pin, True)
    pwm.ChangeDutyCycle(duty)
    time.sleep(0.5)
    GPIO.output(servo_pin, False)
    pwm.ChangeDutyCycle(0)

try:
    while True:
        # Move servo to 0 degrees
        set_angle(0)
        time.sleep(1)

        # Move servo to 90 degrees
        set_angle(90)
        time.sleep(1)

        # Move servo to 180 degrees
        set_angle(180)
        time.sleep(1)

except KeyboardInterrupt:
    print("Exiting program")

finally:
    pwm.stop()
    GPIO.cleanup()