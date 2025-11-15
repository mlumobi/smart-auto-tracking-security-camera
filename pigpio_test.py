import pigpio
import time

# GPIO pin your servo is connected to
servo_pin = 18  # change if needed

# Connect to local Pi GPIO daemon
pi = pigpio.pi()
if not pi.connected:
    exit("Failed to connect to pigpio daemon")

# Function to set servo angle
def set_angle(angle):
    # MG90S pulse width: ~500-2500 microseconds
    pulse_width = 500 + (angle / 180.0) * 2000
    pi.set_servo_pulsewidth(servo_pin, pulse_width)

try:
    while True:
        for angle in range(0, 181, 10):
            set_angle(angle)
            print(f"Moved to {angle}°")
            time.sleep(0.5)
        for angle in range(180, -1, -10):
            set_angle(angle)
            print(f"Moved to {angle}°")
            time.sleep(0.5)

except KeyboardInterrupt:
    print("Exiting...")

finally:
    pi.set_servo_pulsewidth(servo_pin, 0)  # stop PWM
    pi.stop()