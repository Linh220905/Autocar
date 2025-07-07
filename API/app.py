from fastapi import FastAPI, WebSocket
from fastapi.middleware.cors import CORSMiddleware
from ultralytics import YOLO
import numpy as np
from PIL import Image
import io
from collections import deque
import cv2
import uvicorn

app = FastAPI()
model = YOLO("../models/newmodel.pt")
offset_buffer = deque(maxlen=5)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.get("/")
def index():
    return {"message": "Lane Detection API (WebSocket Ready)"}

def preprocess_image(img_bytes: bytes) -> np.ndarray:
    image = Image.open(io.BytesIO(img_bytes)).convert("RGB")
    frame = np.array(image)
    return cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)

def get_lane_centers(masks: np.ndarray, height: int) -> list:
    centers = []
    for mask in masks:
        ys, xs = np.where(mask > 0.5)
        if xs.size == 0 or ys.size == 0:
            continue
        cx, cy = np.mean(xs), np.mean(ys)
        if cy > height * 0.3:
            centers.append((cx, cy))
    return centers

@app.websocket("/ws/predict")
async def ws_predict(websocket: WebSocket):
    await websocket.accept()
    try:
        while True:
            img_bytes = await websocket.receive_bytes()
            frame = preprocess_image(img_bytes)
            height, width = frame.shape[:2]
            center_frame = width / 2
            results = model.predict(frame, conf=0.7)[0]
            masks = results.masks.data.cpu().numpy() if results.masks is not None else None
            lane_centers = get_lane_centers(masks, height) if masks is not None else []

            if not lane_centers:
                await websocket.send_json({
                    "offset": None,
                    "direction": "NO LANE",
                    "centers": []
                })
                continue

            nearest = min(lane_centers, key=lambda pt: abs(pt[0] - center_frame))
            offset = nearest[0] - center_frame
            offset_buffer.append(offset)
            smoothed_offset = float(np.mean(offset_buffer))

            await websocket.send_json({
                "offset": smoothed_offset,
                "centers": lane_centers
            })

    except Exception as e:
        print("WebSocket Closed:", e)
        await websocket.close()

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)
