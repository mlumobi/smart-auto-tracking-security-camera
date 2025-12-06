import cv2
info = cv2.getBuildInformation().split("\n")
for line in info:
    if "GStreamer" in line or "FFmpeg" in line:
        print(line)