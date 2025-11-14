import time
import lgpio

chip = lgpio.gpiochip_open(0)

SERVO = 17  # GPIO pin for signal
lgpio.gpio_claim_output(chip, SERVO)

def set_pulse(angle):
    # convert angle to 1–2 ms pulse width
    pulse = 1000 + (angle / 180.0) * 1000  # microseconds
    lgpio.gpio_wave_clear(chip)

    # Create a single PWM pulse using waves
    waves = [
        (SERVO, 1, pulse),
        (SERVO, 0, 20000 - pulse)  # 20 ms period
    ]

    lgpio.gpio_wave_add_generic(chip, waves)
    wid = lgpio.gpio_wave_create(chip)
    lgpio.gpio_wave_send_repeat(chip, wid)

try:
    while True:
        for a in range(0, 181, 5):
            set_pulse(a)
            time.sleep(0.02)

        for a in range(180, -1, -5):
            set_pulse(a)
            time.sleep(0.02)

except KeyboardInterrupt:
    pass

lgpio.gpiochip_close(chip)