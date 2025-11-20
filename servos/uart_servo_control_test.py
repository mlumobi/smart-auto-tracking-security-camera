import serial
import time

ser = serial.Serial('/dev/serial0', 115200)

ser.write("tilt=0\n".encode())
ser.write("tilt=60\n".encode())
time.sleep(1)

while True:
    for panning in range(0, 181, 45):
        cmd = f"pan={panning}\n"
        ser.write(cmd.encode())
        print(f"Sent: {cmd.strip()}")
        time.sleep(0.6)

    for panning in range(180, -1, -45):
        cmd = f"pan={panning}\n"
        ser.write(cmd.encode())
        print(f"Sent: {cmd.strip()}")
        time.sleep(0.6)
    
    
