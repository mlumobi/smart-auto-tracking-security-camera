import lgpio
import time

PWM_PIN = 12       # Hardware PWM pins only: 12, 13, 18, 19
FREQ = 50          # Servo frequency

chip = lgpio.gpiochip_open(0)

try:
    while True:
        lgpio.tx_servo(chip, 12, 1500, servo_frequency=50, pulse_offset=0, pulse_cycles=0)
        time.sleep(2)


except KeyboardInterrupt:
    pass

lgpio.tx_pwm(chip, PWM_PIN, 0, 0)   # Stop PWM
lgpio.gpiochip_close(chip)