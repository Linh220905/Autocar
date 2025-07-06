# stream_video_api.py
from fastapi import FastAPI, Response
from fastapi.responses import StreamingResponse
import cv2
import uvicorn

app = FastAPI()

cap = cv2.VideoCapture("../Data/left_lane.mp4")

def generate_frames():
    while True:
        success, frame = cap.read()
        if not success:
            break

        # Encode frame thành JPEG
        ret, buffer = cv2.imencode('.jpg', frame)
        if not ret:
            continue

        frame_bytes = buffer.tobytes()

        # Trả về dạng multipart MJPEG
        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + frame_bytes + b'\r\n')

@app.get("/")
def root():
    return {"message": "Video Streaming API"}

@app.get("/video_feed")
def video_feed():
    return StreamingResponse(generate_frames(),
                             media_type="multipart/x-mixed-replace; boundary=frame")
if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)