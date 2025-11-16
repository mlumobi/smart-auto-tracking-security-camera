import cv2
import gradio as gr
from picamera2 import Picamera2
import time

# Initialize PiCamera2
picam2 = Picamera2()
config = picam2.create_video_configuration(main={"size": (640, 480)})
picam2.configure(config)
picam2.start()
time.sleep(1)

def stream_generator():
    while True:
        frame = picam2.capture_array()

        # Convert BGR → RGB
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        yield frame

# Gradio live video streaming using generator
with gr.Blocks() as demo:
    gr.Markdown("## 📷 Raspberry Pi 5 Camera – Real-time Stream")
    video = gr.Image(label="Live Camera", streaming=True)
    video.stream(stream_generator)

demo.queue()
demo.launch(server_name="0.0.0.0", server_port=7860)