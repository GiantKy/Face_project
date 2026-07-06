import sys
import os
import time

sys.path.append(
    os.path.abspath(
        os.path.join(os.path.dirname(__file__), "..")
    )
)

import cv2
from ultralytics import YOLO

# =========================
# MODEL PATH
# =========================
BASE_DIR = os.path.dirname(
    os.path.dirname(os.path.abspath(__file__))
)

MODEL_PATH = os.path.join(
    BASE_DIR, "model", "Face_Detection.pt"
)

print(f"[INFO] Loading model: {MODEL_PATH}")

model = YOLO(MODEL_PATH)

print("[INFO] Model loaded successfully!")

# =========================
# CAMERA
# =========================
cap = cv2.VideoCapture(0)

if not cap.isOpened():
    print("[ERROR] Cannot open camera!")
    sys.exit(1)

print("[INFO] Camera opened. Press ESC to exit.")

# =========================
# FPS
# =========================
prev_time = time.time()

# =========================
# MAIN LOOP
# =========================
while True:
    ret, frame = cap.read()

    if not ret:
        break

    frame = cv2.flip(frame, 1)

    # ---- Inference ----
    results = model(frame, verbose=False)

    # ---- Draw results ----
    for result in results:
        for box in result.boxes:

            x1, y1, x2, y2 = map(int, box.xyxy[0])

            conf = float(box.conf[0])

            cls = int(box.cls[0])

            # class name
            label = model.names[cls] if cls in model.names else f"class_{cls}"

            # color: green for high conf, yellow for low
            if conf > 0.7:
                color = (0, 255, 0)
            elif conf > 0.4:
                color = (0, 255, 255)
            else:
                color = (0, 0, 255)

            # bounding box
            cv2.rectangle(
                frame,
                (x1, y1),
                (x2, y2),
                color,
                2
            )

            # label + confidence
            text = f"{label} {conf:.2f}"

            (tw, th), _ = cv2.getTextSize(
                text,
                cv2.FONT_HERSHEY_SIMPLEX,
                0.6,
                1
            )

            cv2.rectangle(
                frame,
                (x1, y1 - th - 10),
                (x1 + tw, y1),
                color,
                -1
            )

            cv2.putText(
                frame,
                text,
                (x1, y1 - 5),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.6,
                (0, 0, 0),
                1
            )

    # ---- FPS ----
    curr_time = time.time()
    fps = 1.0 / (curr_time - prev_time)
    prev_time = curr_time

    cv2.putText(
        frame,
        f"FPS: {fps:.1f}",
        (10, 30),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.8,
        (255, 255, 255),
        2
    )

    # ---- Detection count ----
    total = sum(
        len(r.boxes) for r in results
    )

    cv2.putText(
        frame,
        f"Detected: {total}",
        (10, 60),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.7,
        (255, 255, 0),
        2
    )

    # ---- Display ----
    cv2.imshow(
        "Test Best(8) Model - Face Detection",
        frame
    )

    if cv2.waitKey(1) == 27:
        break

cap.release()
cv2.destroyAllWindows()

print("[INFO] Done.")
