from picamera2 import Picamera2
from picamera2.encoders import JpegEncoder
from picamera2.outputs import FileOutput
from picamera2.streaming import StreamingHttpServer

picam2 = Picamera2()
picam2.configure(picam2.create_video_configuration(main={"size": (1280, 720)}))

output = FileOutput()
encoder = JpegEncoder(q=85)

server = StreamingHttpServer(bind_address="0.0.0.0", port=8000)
server.set_encoder(encoder)
server.set_output(output)

picam2.start_recording(encoder, output)

print("Streaming at http://0.0.0.0:8000")
server.serve_forever()