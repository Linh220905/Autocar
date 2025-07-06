import requests
import cv2
import time

API_URL = "http://localhost:8000/predict/"
# API_URL = "http://103.161.96.76:8000/predict/"
VIDEO_PATH = "../Data/left_lane_low.mp4"

cap = cv2.VideoCapture(VIDEO_PATH)

if not cap.isOpened():
    print("Could not open video")
    exit()

prev_time = time.time()
frame_count = 0
fps = 0

while True:
    ret, frame = cap.read()
    if not ret:
        break

    start_time = time.time()

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

    # ==== FPS calculation ====
    frame_count += 1
    current_time = time.time()
    elapsed = current_time - prev_time
    if elapsed >= 1.0:
        fps = frame_count / elapsed
        frame_count = 0
        prev_time = current_time

    cv2.putText(frame, f"FPS: {fps:.2f}", (10, frame_height - 10),
                cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 0), 2)

    cv2.imshow("YOLO Lane Detection", frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
