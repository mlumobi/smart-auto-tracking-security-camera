import time
import pigpio

SERVO_PIN = 17  # signal pin only

pi = pigpio.pi()
if not pi.connected:
    print("pigpio daemon not running")
    exit()

def angle_to_pulse(angle):
    # 0–180° → 500–2500 μs pulse
    return int(500 + (angle / 180.0) * 2000)

try:
    while True:
        # Sweep 0 → 180
        for angle in range(0, 181, 5):
            pi.set_servo_pulsewidth(SERVO_PIN, angle_to_pulse(angle))
            time.sleep(0.02)

        # Sweep 180 → 0
        for angle in range(180, -1, -5):
            pi.set_servo_pulsewidth(SERVO_PIN, angle_to_pulse(angle))
            time.sleep(0.02)

except KeyboardInterrupt:
    pass

# stop servo output
pi.set_servo_pulsewidth(SERVO_PIN, 0)
pi.stop()