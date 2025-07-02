from fastapi import FastAPI, File, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from ultralytics import YOLO
from PIL import Image
import io
import numpy as np
import uvicorn

app = FastAPI()
model = YOLO('../models/detect_lane.pt')

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"]
)

@app.post("/predict/")
async def predict_lane(file: UploadFile = File(...)):
    contents = await file.read()
    image = Image.open(io.BytesIO(contents)).convert("RGB")
    frame = np.array(image)

    frame_width = frame.shape[1]
    results = model.predict(frame, conf=0.4, verbose=False)
    boxes = results[0].boxes

    lane_centers = []

    for box in boxes:
        x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()
        cx = (x1 + x2) / 2
        lane_centers.append(cx)

    if len(lane_centers) >= 2:
        mid_x = (lane_centers[0] + lane_centers[1]) / 2
    elif len(lane_centers) == 1:
        mid_x = lane_centers[0]
    else:
        return {"offset": None}

    offset = float(mid_x - (frame_width / 2))
    return {"offset": offset}

if __name__ == '__main__':
    uvicorn.run(app, host="0.0.0.0", port=8000)