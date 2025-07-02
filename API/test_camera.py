from fastapi import FastAPI, Request
from fastapi.responses import StreamingResponse, HTMLResponse
from fastapi.templating import Jinja2Templates
from picamera2 import Picamera2
import cv2
import requests
import time
import uvicorn

API_URL = "http://127.0.0.1:8000/predict/"

app = FastAPI()
templates = Jinja2Templates(directory="templates")

picam2 = Picamera2()
picam2.configure(picam2.create_preview_configuration(main={"size": (640, 480)}))
picam2.start()
time.sleep(1)

def generate_frames():
    while True:
        frame = picam2.capture_array()
        frame_height, frame_width = frame.shape[:2]

        resized = cv2.resize(frame, (640, 480))
        _, img_encoded = cv2.imencode('.jpg', resized)

        try:
            response = requests.post(
                API_URL,
                files={"file": ("frame.jpg", img_encoded.tobytes(), "image/jpeg")},
                timeout=2
            )

            if response.status_code == 200:
                data = response.json()
                offset = data.get("offset", None)

                if offset is not None:
                    center_x = frame_width // 2
                    predicted_x = int(center_x + offset)

                    cv2.line(frame, (center_x, 0), (center_x, frame_height), (255, 255, 255), 2)
                    cv2.line(frame, (predicted_x, 0), (predicted_x, frame_height), (0, 0, 255), 2)
                    cv2.putText(frame, f"Offset: {offset:.1f}", (10, 30),
                                cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
                else:
                    cv2.putText(frame, "No lane detected", (10, 30),
                                cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
            else:
                cv2.putText(frame, f"API Error: {response.status_code}", (10, 30),
                            cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)

        except Exception as e:
            cv2.putText(frame, "Request Failed", (10, 30),
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
            print("Exception:", e)

        # Encode lại ảnh để stream
        _, jpeg = cv2.imencode('.jpg', frame)
        frame_bytes = jpeg.tobytes()

        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + frame_bytes + b'\r\n')

@app.get("/", response_class=HTMLResponse)
def index(request: Request):
    return templates.TemplateResponse("index.html", {"request": request})

@app.get("/video_feed")
def video_feed():
    return StreamingResponse(generate_frames(),
                             media_type="multipart/x-mixed-replace; boundary=frame")

if __name__ == '__main__':
    uvicorn.run(app, host="0.0.0.0", port=8000)