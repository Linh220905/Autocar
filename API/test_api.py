import cv2
import httpx
import asyncio
import time

API_URL = "http://100.64.0.4:8000/predict/"
# API_URL = "http://localhost:8000/predict/"
VIDEO_PATH = "../Data/left_lane_low.mp4"

async def main():
    cap = cv2.VideoCapture(VIDEO_PATH)
    if not cap.isOpened():
        print("❌ Could not open video.")
        return

    frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    center_x = frame_width // 2

    frame_count = 0
    prev_time = time.time()
    fps = 0
    infer_time = -1

    async with httpx.AsyncClient(http2=True, timeout=5.0) as client:
        while True:
            ret, frame = cap.read()
            if not ret:
                break

            resized = cv2.resize(frame, (640, 480))
            _, img_encoded = cv2.imencode('.jpg', resized)
            files = {"file": ("frame.jpg", img_encoded.tobytes(), "image/jpeg")}

            start_time = time.time()
            try:
                response = await client.post(API_URL, files=files)
                infer_time = (time.time() - start_time) * 1000

                if response.status_code == 200:
                    data = response.json()
                    offset = data.get("offset", None)

                    if offset is not None:
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

            # === FPS calc ===
            frame_count += 1
            now = time.time()
            elapsed = now - prev_time
            if elapsed >= 1.0:
                fps = frame_count / elapsed
                frame_count = 0
                prev_time = now

            cv2.putText(frame, f"FPS: {fps:.2f}", (10, frame_height - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 0), 2)
            cv2.putText(frame, f"Infer: {infer_time:.1f}ms", (10, 60),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)

            cv2.imshow("YOLO Lane Detection (httpx async)", frame)
            print(f"FPS: {fps:.2f}, Infer time: {infer_time:.2f}")
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

    cap.release()
    cv2.destroyAllWindows()
    print("✅ Done.")

if __name__ == "__main__":
    asyncio.run(main())