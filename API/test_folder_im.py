import cv2
import numpy as np
import os
from collections import deque
from ultralytics import YOLO
import time

# Tải model YOLO
model = YOLO('../models/newmodel.pt')

# Thư mục ảnh đầu vào và đầu ra
input_folder = 'Frames'
output_folder = 'output_folder'
os.makedirs(output_folder, exist_ok=True)

# Đọc danh sách file ảnh
image_files = [f for f in os.listdir(input_folder) if f.lower().endswith(('.jpg', '.jpeg', '.png'))]

# Tính toán thông số
offset_buffer = deque(maxlen=5)

for image_file in image_files:
    image_path = os.path.join(input_folder, image_file)
    frame = cv2.imread(image_path)

    if frame is None:
        print(f"Không đọc được ảnh: {image_file}")
        continue

    height, width = frame.shape[:2]
    center_frame = width / 2

    # Thời gian bắt đầu (để đo FPS nếu cần)
    prev_time = time.time()

    # Dự đoán
    results = model.predict(frame, conf=0.7, verbose=False)
    boxes = results[0].boxes
    frame = results[0].plot()

    lane_centers = []
    if results[0].masks is not None:
        masks = results[0].masks.data.cpu().numpy()
        for mask in masks:
            ys, xs = np.where(mask > 0.5)
            if len(xs) == 0 or len(ys) == 0:
                continue
            cx = np.mean(xs)
            cy = np.mean(ys)
            if cy > height * 0.3:
                lane_centers.append((cx, cy))
                cv2.circle(frame, (int(cx), int(cy)), 5, (0, 255, 0), -1)

    # Kẻ vùng điều hướng
    cx_center = int(center_frame)
    lane_widths = [70, 45, 15]
    colors = {
        "HARD": (0, 0, 255),
        "SLIGHT": (0, 165, 255),
        "STRAIGHT": (0, 255, 0)
    }

    cv2.rectangle(frame, (cx_center - lane_widths[2], 0), (cx_center + lane_widths[2], height), colors["STRAIGHT"], 1)
    cv2.rectangle(frame, (cx_center - lane_widths[1], 0), (cx_center - lane_widths[2], height), colors["SLIGHT"], 1)
    cv2.rectangle(frame, (cx_center + lane_widths[2], 0), (cx_center + lane_widths[1], height), colors["SLIGHT"], 1)
    cv2.rectangle(frame, (0, 0), (cx_center - lane_widths[1], height), colors["HARD"], 1)
    cv2.rectangle(frame, (cx_center + lane_widths[1], 0), (width, height), colors["HARD"], 1)
    cv2.line(frame, (cx_center, 0), (cx_center, height), (255, 255, 255), 1)

    # Tính offset
    if lane_centers:
        nearest = min(lane_centers, key=lambda pt: abs(pt[0] - center_frame))
        offset = nearest[0] - center_frame

        if abs(offset) <= 15:
            direction = "GO STRAIGHT"
        elif 15 < offset <= 45:
            direction = "SLIGHT RIGHT"
        elif -45 <= offset < -15:
            direction = "SLIGHT LEFT"
        elif offset > 45:
            direction = "HARD RIGHT"
        elif offset < -45:
            direction = "HARD LEFT"
        else:
            direction = "UNKNOWN"

        cv2.line(frame, (int(nearest[0]), height // 2), (int(center_frame), height // 2), (255, 0, 0), 2)
        cv2.putText(frame, f"Offset: {offset:.2f}", (30, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
        cv2.putText(frame, f"Direction: {direction}", (30, 90), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 0), 2)
    else:
        cv2.putText(frame, "No lanes detected", (30, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)

    # Tính và hiển thị FPS
    curr_time = time.time()
    fps_text = 1 / (curr_time - prev_time)
    cv2.putText(frame, f"FPS: {fps_text:.2f}", (30, 130), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 255), 2)

    # Lưu ảnh đầu ra
    output_path = os.path.join(output_folder, image_file)
    cv2.imwrite(output_path, frame)
    print(f"Đã xử lý: {image_file}")

print("✅ Xử lý hoàn tất!")
