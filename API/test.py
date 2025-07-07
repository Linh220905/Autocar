import cv2
import asyncio
import websockets
import json
import time

VIDEO_PATH = "../Data/left_lane_low.mp4"
WS_URL = "ws://100.64.0.4:8000/ws/predict"

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

    async with websockets.connect(WS_URL) as ws:
        i = 0
        while True:
            ret, frame = cap.read()
            if not ret:
                break

            _, img_encoded = cv2.imencode('.jpg', frame, [int(cv2.IMWRITE_JPEG_QUALITY), 60])
            img_bytes = img_encoded.tobytes()

            start_time = time.time()
            try:
                await ws.send(img_bytes)
                res = await ws.recv()
                infer_time = (time.time() - start_time) * 1000

                data = json.loads(res)
                offset = data.get("offset", None)

                if offset is not None:
                    predicted_x = int(center_x + offset)
                    cv2.line(frame, (center_x, 0), (center_x, frame_height), (255, 255, 255), 2)
                    cv2.line(frame, (predicted_x, 0), (predicted_x, frame_height), (0, 0, 255), 2)
                    cv2.putText(frame, f"Offset: {offset:.1f}", (10, 30),
                                cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
                else:
                    cv2.imwrite(f"Frames/Frame_no_lane_{i}.png", frame)
                    i+=1
                    cv2.putText(frame, "No lane detected", (10, 30),
                                cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)

            except Exception as e:
                print("❌ Exception:", e)
                break

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

            cv2.imshow("YOLO Lane Detection (WebSocket)", frame)
            print(f"FPS: {fps:.2f}, Infer time: {infer_time:.2f}")
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

    cap.release()
    cv2.destroyAllWindows()
    print("✅ Done.")

if __name__ == "__main__":
    asyncio.run(main())
