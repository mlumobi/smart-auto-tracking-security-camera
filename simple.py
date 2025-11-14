from gpiozero import Servo
from time import sleep

s = Servo(18)

while True:
    s.min()
    sleep(1)
    s.mid()
    sleep(1)
    s.max()
    sleep(1)