# -*- coding: utf-8 -*-
"""
=============================================================================
Test Anti-Spoofing: Roboflow Inference Local (Giải pháp 2A - Không cần Docker)
=============================================================================
Sử dụng thư viện `inference.get_model` để chạy mô hình Roboflow trực tiếp trên máy:
  - Tự động tải và cache trọng số mô hình về máy cục bộ.
  - Chạy offline hoàn toàn trên CPU/GPU, không cần Docker, không cần bật Server riêng.
  - Hỗ trợ:
      + Test trên 1 ảnh tĩnh bất kỳ (`--image`)
      + Test trực tiếp trên Webcam thời gian thực (`--cam 0`)

Mô hình Roboflow:
  - Model ID: face-spoof-detection-liika-qopyy/1
  - API Key : LiYT7osRW01duX3ao91S

Cách chạy:
  # 1. Test trên ảnh tĩnh (mặc định lấy ảnh trong data_raw/0.jpg):
  python tests/test_anti_spoof_roboflow.py --image data_raw/0.jpg

  # 2. Test trực tiếp bằng Webcam thời gian thực:
  python tests/test_anti_spoof_roboflow.py --cam 0
=============================================================================
"""

import sys
import os
import time
import argparse
import warnings

# Tắt các cảnh báo không cần thiết từ thư viện
os.environ["CORE_MODEL_GAZE_ENABLED"] = "False"
os.environ["CORE_MODEL_SAM_ENABLED"] = "False"
warnings.filterwarnings("ignore")

if hasattr(sys.stdout, "reconfigure"):
    try:
        sys.stdout.reconfigure(encoding="utf-8")
        sys.stderr.reconfigure(encoding="utf-8")
    except Exception:
        pass

import cv2
import numpy as np
from inference import get_model

# Cấu hình đường dẫn
CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
BASE_DIR = os.path.dirname(CURRENT_DIR)
DATA_RAW_DIR = os.path.join(BASE_DIR, "data_raw")
OUTPUT_DIR = os.path.join(CURRENT_DIR, "output", "roboflow_test")
os.makedirs(OUTPUT_DIR, exist_ok=True)

MODEL_ID = "face-spoof-detection-liika-qopyy/1"
API_KEY = "LiYT7osRW01duX3ao91S"


def parse_predictions(response, img_w, img_h):
    """
    Trích xuất danh sách bounding box từ kết quả trả về của `inference`.
    Tọa độ trả về từ Roboflow dạng center (x, y, width, height) -> chuyển thành (x1, y1, x2, y2).
    """
    dets = []
    # Response có thể là list hoặc object đơn lẻ
    preds = []
    if isinstance(response, list) and len(response) > 0:
        if hasattr(response[0], "predictions"):
            preds = response[0].predictions
    elif hasattr(response, "predictions"):
        preds = response.predictions

    for p in preds:
        cx = float(getattr(p, "x", 0.0))
        cy = float(getattr(p, "y", 0.0))
        pw = float(getattr(p, "width", 0.0))
        ph = float(getattr(p, "height", 0.0))
        conf = float(getattr(p, "confidence", 0.0))
        cls_name = str(getattr(p, "class_name", "")).lower()

        # Tính toán góc trên-trái và dưới-phải
        x1 = max(0, int(cx - pw / 2.0))
        y1 = max(0, int(cy - ph / 2.0))
        x2 = min(img_w, int(cx + pw / 2.0))
        y2 = min(img_h, int(cy + ph / 2.0))

        is_real = ("real" in cls_name)
        dets.append({
            "bbox": [x1, y1, x2, y2],
            "confidence": conf,
            "class_name": cls_name,
            "is_real": is_real,
            "label": "REAL" if is_real else "SPOOF"
        })

    return dets


def draw_detection_results(image, detections, fps=None):
    """Vẽ bounding box và thông số lên ảnh"""
    vis = image.copy()
    h, w = vis.shape[:2]

    for d in detections:
        x1, y1, x2, y2 = d["bbox"]
        is_real = d["is_real"]
        color = (0, 255, 127) if is_real else (0, 0, 255)  # Xanh nếu Thật, Đỏ nếu Giả
        label_text = f"{d['label']} {d['confidence']*100:.1f}%"

        # Vẽ khung viền
        cv2.rectangle(vis, (x1, y1), (x2, y2), color, 2)

        # Vẽ nhãn phía trên box
        (tw, th), _ = cv2.getTextSize(label_text, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)
        bg_y1 = max(0, y1 - th - 10)
        bg_y2 = y1
        cv2.rectangle(vis, (x1, bg_y1), (x1 + tw + 10, bg_y2), (20, 20, 25), -1)
        cv2.rectangle(vis, (x1, bg_y1), (x1 + tw + 10, bg_y2), color, 1)
        cv2.putText(vis, label_text, (x1 + 5, y1 - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.55, color, 2, cv2.LINE_AA)

    # Vẽ thông tin tiêu đề góc trên bên trái
    overlay = vis.copy()
    cv2.rectangle(overlay, (10, 10), (min(w - 10, 480), 75), (15, 18, 24), -1)
    cv2.addWeighted(overlay, 0.85, vis, 0.15, 0, vis)
    cv2.rectangle(vis, (10, 10), (min(w - 10, 480), 75), (60, 70, 90), 1)

    cv2.putText(vis, "ROBOFLOW ANTI-SPOOF (LOCAL INFERENCE)", (20, 34),
                cv2.FONT_HERSHEY_SIMPLEX, 0.52, (0, 230, 255), 2, cv2.LINE_AA)
    info_str = f"Model: {MODEL_ID} | Detections: {len(detections)}"
    cv2.putText(vis, info_str, (20, 58),
                cv2.FONT_HERSHEY_SIMPLEX, 0.40, (200, 210, 220), 1, cv2.LINE_AA)

    if fps is not None:
        cv2.putText(vis, f"FPS: {fps:.1f}", (w - 110, 35),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 255, 255), 2, cv2.LINE_AA)

    return vis


def test_on_image(model, image_path):
    """Kiểm tra mô hình trên 1 ảnh tĩnh"""
    if not os.path.exists(image_path):
        print(f"[ERROR] Không tìm thấy file ảnh tại: {image_path}")
        return

    print(f"\n[INFO] Đang đọc ảnh: {image_path}")
    image = cv2.imread(image_path)
    if image is None:
        print(f"[ERROR] Không thể đọc định dạng ảnh từ: {image_path}")
        return

    h, w = image.shape[:2]
    print(f"[INFO] Kích thước ảnh: {w}x{h}")

    # Chạy suy luận cục bộ
    t0 = time.time()
    response = model.infer(image=image)
    infer_time = (time.time() - t0) * 1000.0

    # Trích xuất kết quả
    detections = parse_predictions(response, w, h)
    print(f"[OK] Thời gian suy luận: {infer_time:.1f}ms")
    print(f"[OK] Phát hiện {len(detections)} vùng:")

    for idx, d in enumerate(detections, 1):
        status = "THẬT (REAL)" if d["is_real"] else "GIẢ MẠO (SPOOF)"
        print(f"  [{idx}] BBox: {d['bbox']} | Confidence: {d['confidence']*100:.1f}% -> {status}")

    # Vẽ kết quả
    vis = draw_detection_results(image, detections)

    # Lưu ảnh kết quả
    out_filename = f"result_{os.path.basename(image_path)}"
    out_path = os.path.join(OUTPUT_DIR, out_filename)
    cv2.imwrite(out_path, vis)
    print(f"[INFO] Đã lưu ảnh kết quả vào: {out_path}")

    # Hiển thị cửa sổ
    cv2.imshow("Roboflow Anti-Spoof Test (Solution 2A)", vis)
    print("\n[HƯỚNG DẪN] Nhấn phím bất kỳ trên cửa sổ ảnh để đóng.")
    cv2.waitKey(0)
    cv2.destroyAllWindows()


def test_on_webcam(model, cam_id=0):
    """Kiểm tra mô hình trực tiếp qua Webcam theo thời gian thực"""
    print(f"\n[INFO] Đang mở Camera ID: {cam_id}...")
    cap = cv2.VideoCapture(cam_id)
    if not cap.isOpened():
        print(f"[ERROR] Không thể mở Camera ID {cam_id}!")
        return

    print("[OK] Đã kết nối Camera thành công!")
    print("  * Nhấn phím [q] hoặc [ESC] để thoát.")
    print("  * Nhấn phím [s] để lưu ảnh chụp hiện tại.\n")

    prev_time = time.time()
    shot_count = 0

    while True:
        ret, frame = cap.read()
        if not ret:
            print("[WARN] Mất tín hiệu Webcam.")
            break

        frame = cv2.flip(frame, 1)
        h, w = frame.shape[:2]

        # Dự đoán trực tiếp trên frame
        curr_time = time.time()
        fps = 1.0 / (curr_time - prev_time) if curr_time > prev_time else 0.0
        prev_time = curr_time

        try:
            response = model.infer(image=frame)
            detections = parse_predictions(response, w, h)
        except Exception as e:
            detections = []
            print(f"[WARN] Lỗi khi infer: {e}")

        # Vẽ kết quả
        display = draw_detection_results(frame, detections, fps=fps)

        cv2.imshow("Roboflow Anti-Spoof Test (Solution 2A)", display)
        key = cv2.waitKey(1) & 0xFF

        if key in (27, ord('q'), ord('Q')):
            break
        elif key in (ord('s'), ord('S')):
            shot_count += 1
            save_path = os.path.join(OUTPUT_DIR, f"webcam_snap_{shot_count}.jpg")
            cv2.imwrite(save_path, display)
            print(f"[LƯU ẢNH] Đã lưu snapshot tại: {save_path}")

    cap.release()
    cv2.destroyAllWindows()
    print("[INFO] Đã đóng Webcam an toàn.")


def main():
    parser = argparse.ArgumentParser(description="Test Anti-Spoofing Roboflow Local (Solution 2A)")
    parser.add_argument("--image", type=str, default="", help="Đường dẫn file ảnh để kiểm tra (VD: data_raw/0.jpg)")
    parser.add_argument("--cam", type=int, default=None, help="Camera device index để kiểm tra thời gian thực (VD: 0)")
    args = parser.parse_args()

    print("\n" + "=" * 70)
    print("     TEST MÔ HÌNH ROBOFLOW LOCAL INFERENCE (GIẢI PHÁP 2A)")
    print("=" * 70)
    print(f"  * Model ID : {MODEL_ID}")
    print(f"  * Nền tảng : inference (Local Python Cache, KHÔNG CẦN DOCKER)")
    print("=" * 70 + "\n")

    print("[INFO] Đang nạp mô hình vào bộ nhớ RAM...")
    t_start = time.time()
    try:
        model = get_model(model_id=MODEL_ID, api_key=API_KEY)
        load_sec = time.time() - t_start
        print(f"[OK] Đã nạp xong mô hình thành công trong {load_sec:.2f} giây!\n")
    except Exception as e:
        print(f"[ERROR] Không thể nạp mô hình Roboflow: {e}")
        return

    # Xác định chế độ test: Webcam hay Ảnh
    if args.cam is not None:
        test_on_webcam(model, cam_id=args.cam)
    else:
        # Nếu truyền ảnh hoặc mặc định kiểm tra data_raw/0.jpg
        img_target = args.image
        if not img_target:
            default_img = os.path.join(DATA_RAW_DIR, "0.jpg")
            if os.path.exists(default_img):
                img_target = default_img
            else:
                # Tìm ảnh bất kỳ trong data_raw/
                jpgs = [f for f in os.listdir(DATA_RAW_DIR) if f.lower().endswith((".jpg", ".png"))]
                if jpgs:
                    img_target = os.path.join(DATA_RAW_DIR, jpgs[0])

        if img_target and os.path.exists(img_target):
            test_on_image(model, img_target)
        else:
            print("[INFO] Không tìm thấy ảnh trong data_raw/ để test mặc định.")
            print("[INFO] Đang chuyển sang mở Webcam ID 0 để test...")
            test_on_webcam(model, cam_id=0)


if __name__ == "__main__":
    main()
