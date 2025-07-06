# === SERVER: main.py ===
from fastapi import FastAPI, WebSocket
from fastapi.middleware.cors import CORSMiddleware
from ultralytics import YOLO
import numpy as np
from PIL import Image
import io
from collections import deque
import torch
import uvicorn

app = FastAPI()
model = YOLO("../models/detect_lane_retrain.pt", verbose=False)
offset_buffer = deque(maxlen=3)

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

@app.websocket("/ws/predict")
async def ws_predict(websocket: WebSocket):
    await websocket.accept()
    while True:
        try:
            img_bytes = await websocket.receive_bytes()
            image = Image.open(io.BytesIO(img_bytes)).convert("RGB")
            frame = np.array(image)
            frame_height, frame_width = frame.shape[:2]
            center_frame = frame_width / 2

            results = model.predict(frame, conf=0.7, verbose=False)
            lane_centers = []
            masks = results[0].masks.data if results[0].masks is not None else []

            for mask in masks:
                ys, xs = torch.where(mask > 0.5)
                if ys.numel() == 0 or xs.numel() == 0:
                    continue
                cx = np.mean(xs.cpu().numpy())
                cy = np.mean(ys.cpu().numpy())
                if cy > frame_height * 0.3:
                    lane_centers.append((cx, cy))

            if not lane_centers:
                await websocket.send_json({"offset": None, "direction": "NO LANE", "centers": []})
                continue

            nearest = min(lane_centers, key=lambda pt: abs(pt[0] - center_frame))
            offset = float(nearest[0] - center_frame)
            offset_buffer.append(offset)
            smoothed_offset = float(np.mean(offset_buffer))

            await websocket.send_json({
                "offset": smoothed_offset,
                "direction": "TRACKING",
                "centers": lane_centers
            })

        except Exception as e:
            print("WebSocket Closed:", e)
            await websocket.close()
            break

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)