# Web UI Quick Start

## Installation

```bash
pip3 install -r requirements_web.txt
```

## Running the Web UI

```bash
./start_web.sh
```

Or manually:

```bash
python3 app_web.py
```

## Access

Open browser and go to:
```
http://raspberry-pi-ip:5000
```

Replace `raspberry-pi-ip` with your Raspberry Pi's IP address.

To find your Pi's IP:
```bash
hostname -I
```

## Features

- Live video stream with object detection
- Real-time target detection status
- Pan/Tilt servo angles display
- FPS counter
- Minimal, responsive design

## Stopping the Server

Press `Ctrl+C` in the terminal



