import serial

# Open serial port
ser = serial.Serial('/dev/serial0', 115200, timeout=1)
print("Connected to /dev/serial0 at 115200 baud.")
print("Type 'exit' to quit.")

try:
    while True:
        # Read input from user
        cmd = input("Enter command to send over UART: ")
        if cmd.lower() == "exit":
            print("Exiting...")
            break
        
        # Send command over UART
        ser.write((cmd + "\n").encode())
        print(f"Sent: {cmd}")

        # Optional: read response from device
        if ser.in_waiting:
            response = ser.readline().decode().strip()
            if response:
                print(f"Received: {response}")

except KeyboardInterrupt:
    print("\nInterrupted by user.")

finally:
    ser.close()
    print("Serial port closed.")