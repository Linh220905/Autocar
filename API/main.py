from fastapi import FastAPI, File, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from ultralytics import YOLO
from PIL import Image
import io
import numpy as np
from collections import deque
import uvicorn

app = FastAPI()
model = YOLO('../models/detect_lane_retrain.pt')

offset_buffer = deque(maxlen=5)
last_offset = None
max_delta = 30

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"]
)

@app.post("/predict/")
async def predict_lane(file: UploadFile = File(...)):
    global last_offset

    contents = await file.read()
    image = Image.open(io.BytesIO(contents)).convert("RGB")
    frame = np.array(image)

    frame_height, frame_width = frame.shape[:2]
    center_frame = frame_width / 2

    results = model.predict(frame, conf=0.7, verbose=False)
    lane_centers = []

    if results[0].masks is not None:
        masks = results[0].masks.data.cpu().numpy()

        for mask in masks:
            ys, xs = np.where(mask > 0.5)

            if len(xs) == 0 or len(ys) == 0:
                continue

            cx = np.mean(xs)
            cy = np.mean(ys)

            if cy > frame_height * 0.3:
                lane_centers.append((cx, cy))

    if not lane_centers:
        return {"offset": None, "direction": "NO LANE"}
    nearest = min(lane_centers, key=lambda pt: abs(pt[0] - center_frame))
    offset = float(nearest[0] - center_frame)
    offset_buffer.append(offset)
    smoothed_offset = float(np.mean(offset_buffer))

    return {
        "offset": smoothed_offset
    }

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)
