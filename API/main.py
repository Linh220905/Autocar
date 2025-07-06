from fastapi import FastAPI, File, UploadFile
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
model.to('cpu')

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

offset_buffer = deque(maxlen=5)

@app.get("/")
def index():
    return "Hello World!!!"

@app.post("/predict/")
async def predict_lane(file: UploadFile = File(...)):
    contents = await file.read()
    image = Image.open(io.BytesIO(contents)).convert("RGB").resize((640, 480))
    frame = np.array(image)
    frame_height, frame_width = frame.shape[:2]
    center_frame = frame_width / 2

    results = model.predict(frame, conf=0.7, device='cpu')
    lane_centers = []

    if results[0].masks is not None:
        masks = results[0].masks.data
        for mask in masks:
            ys, xs = torch.where(mask > 0.5)
            if ys.numel() == 0 or xs.numel() == 0:
                continue

            cx = torch.mean(xs.float()).item()
            cy = torch.mean(ys.float()).item()

            if cy > frame_height * 0.3:
                lane_centers.append((cx, cy))

    if not lane_centers:
        return {"offset": None, "direction": "NO LANE", "centers": []}

    nearest = min(lane_centers, key=lambda pt: abs(pt[0] - center_frame))
    offset = float(nearest[0] - center_frame)
    offset_buffer.append(offset)
    smoothed_offset = float(np.mean(offset_buffer))

    return {
        "offset": smoothed_offset,
        "direction": "TRACKING",
        "centers": lane_centers
    }

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)
