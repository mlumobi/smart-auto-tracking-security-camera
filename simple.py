import time
import lgpio

chip = lgpio.gpiochip_open(0)

SERVO = 18   # use a hardware PWM pin: 12/13/18/19
FREQ = 50    # servo frequency

def set_angle(angle):
    # MG90S recommended range 2.5 12.5
    duty = 2.5 + (angle / 180) * 10  # percent
    lgpio.tx_pwm(chip, SERVO, FREQ, duty)

try:
    while True:
        print("Forward (simulate) ? rotate to 180")
        set_angle(180)
        time.sleep(2)

        print("Stop ? hold at 90")
        set_angle(90)
        time.sleep(2)

        print("Backward (simulate) ? rotate to 0")
        set_angle(0)
        time.sleep(2)

except KeyboardInterrupt:
    pass

lgpio.gpio_free(chip, SERVO)
lgpio.gpiochip_close(chip)