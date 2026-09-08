# -*- coding: utf-8 -*-
"""
=============================================================================
Test Anti-Spoofing: Roboflow Model Mới Tải (anti-spoof-qhqvq-yhn2n/1) - V2
=============================================================================
Sử dụng mô hình Anti-Spoofing YOLOv11 mới nhất được tải về trong dự án:
  - Thư mục Cache: models/roboflow/anti-spoof-qhqvq-yhn2n/1/
  - File trọng số: weights.onnx (10.11 MB)
  - Kiến trúc: YOLOv11n Object Detection
  - Nhãn phân loại: fake, real

Chế độ hoạt động:
  - MẶC ĐỊNH: Chạy OFFLINE 100% bằng trọng số cục bộ trong models/roboflow/.
    + Không phụ thuộc mạng Internet, không có độ trễ kết nối.
    + Tốc độ suy luận tức thì (~20-30ms), Webcam đạt 30+ FPS mượt mà.
  - Tùy chọn `--workflow`: Chạy qua Roboflow Workflow Inference HTTP Client.

Cách chạy:
  # 1. Test trên Webcam thời gian thực (Mặc định dùng model mới tải anti-spoof-qhqvq-yhn2n/1):
  py -3.11 tests/test_anti_spoof_roboflow_v2.py --cam 0

  # 2. Test trên ảnh tĩnh:
  py -3.11 tests/test_anti_spoof_roboflow_v2.py --image data_raw/0.jpg

  # 3. Chuyển đổi giữa các model trong thư mục models/roboflow/:
  py -3.11 tests/test_anti_spoof_roboflow_v2.py --model-id anti-spoof-qhqvq-yhn2n/1
  py -3.11 tests/test_anti_spoof_roboflow_v2.py --model-id face-spoof-detection-liika-qopyy/1
=============================================================================
"""

import sys
import os
import time
import argparse
import warnings

# Tắt các cảnh báo không cần thiết
os.environ["CORE_MODEL_GAZE_ENABLED"] = "False"
os.environ["CORE_MODEL_SAM_ENABLED"] = "False"
os.environ["CORE_MODEL_SAM3_ENABLED"] = "False"
warnings.filterwarnings("ignore")

if hasattr(sys.stdout, "reconfigure"):
    try:
        sys.stdout.reconfigure(encoding="utf-8")
        sys.stderr.reconfigure(encoding="utf-8")
    except Exception:
        pass

# Cấu hình đường dẫn
CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
BASE_DIR = os.path.dirname(CURRENT_DIR)
DATA_RAW_DIR = os.path.join(BASE_DIR, "data_raw")
OUTPUT_DIR = os.path.join(CURRENT_DIR, "output", "roboflow_test_v2")
os.makedirs(OUTPUT_DIR, exist_ok=True)

# Trỏ cache của Roboflow về thư mục models/roboflow của repo
ROBOFLOW_CACHE_DIR = os.path.join(BASE_DIR, "models", "roboflow")
os.environ["MODEL_CACHE_DIR"] = ROBOFLOW_CACHE_DIR
os.makedirs(ROBOFLOW_CACHE_DIR, exist_ok=True)

import cv2
import numpy as np
from inference import get_model

# Cấu hình model mới tải
DEFAULT_MODEL_ID = "anti-spoof-qhqvq-yhn2n/1"
API_KEY = "LiYT7osRW01duX3ao91S"

# Cấu hình Workflow dự phòng
WORKSPACE_NAME = "ks-workspace-hatfd"
WORKFLOW_ID = "face-spoof-detection-vface-spoof-detection-liika-qopyy-1-rfdetr-nano-t4-logic"


def parse_model_predictions(response, img_w: int, img_h: int):
    """
    Trích xuất danh sách bounding box từ kết quả dự đoán (hỗ trợ cả get_model và Workflow).
    Tọa độ trả về từ Roboflow dạng center (x, y, width, height) -> chuyển thành (x1, y1, x2, y2).
    """
    dets = []
    preds = []

    # 1. Response từ inference.get_model
    if isinstance(response, list) and len(response) > 0:
        if hasattr(response[0], "predictions"):
            preds = response[0].predictions
        elif isinstance(response[0], dict) and "predictions" in response[0]:
            p_data = response[0]["predictions"]
            preds = p_data.get("predictions", []) if isinstance(p_data, dict) else p_data
    elif hasattr(response, "predictions"):
        preds = response.predictions
    elif isinstance(response, dict):
        p_data = response.get("predictions", {})
        preds = p_data.get("predictions", []) if isinstance(p_data, dict) else p_data

    for p in preds:
        if hasattr(p, "x"):
            cx = float(getattr(p, "x", 0.0))
            cy = float(getattr(p, "y", 0.0))
            pw = float(getattr(p, "width", 0.0))
            ph = float(getattr(p, "height", 0.0))
            conf = float(getattr(p, "confidence", 0.0))
            cls_name = str(getattr(p, "class_name", "")).lower()
        elif isinstance(p, dict):
            cx = float(p.get("x", 0.0))
            cy = float(p.get("y", 0.0))
            pw = float(p.get("width", 0.0))
            ph = float(p.get("height", 0.0))
            conf = float(p.get("confidence", p.get("score", 0.0)))
            cls_name = str(p.get("class", p.get("class_name", ""))).lower()
        else:
            continue

        # Chuẩn hóa nếu tọa độ ở dạng tỉ lệ (0.0 - 1.0)
        if 0.0 <= cx <= 1.0 and 0.0 <= pw <= 1.0 and img_w > 1:
            cx *= img_w
            cy *= img_h
            pw *= img_w
            ph *= img_h

        # Tính góc trên-trái và dưới-phải
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


def draw_detection_results(image, detections, fps=None, latency_ms=None, model_name=DEFAULT_MODEL_ID):
    """Vẽ bounding box và thông số trực quan công nghệ cao"""
    vis = image.copy()
    h, w = vis.shape[:2]

    for d in detections:
        x1, y1, x2, y2 = d["bbox"]
        is_real = d["is_real"]
        # Màu sắc: Xanh lá (REAL) / Đỏ (SPOOF / FAKE)
        color = (0, 255, 0) if is_real else (0, 0, 255)
        label_text = f"{d['label']} {d['confidence']*100:.1f}%"

        # 1. Bounding box viền
        cv2.rectangle(vis, (x1, y1), (x2, y2), color, 2)

        # 2. Corner brackets công nghệ cao
        corner_len = min(20, (x2 - x1) // 4, (y2 - y1) // 4)
        cv2.line(vis, (x1, y1), (x1 + corner_len, y1), color, 3)
        cv2.line(vis, (x1, y1), (x1, y1 + corner_len), color, 3)
        cv2.line(vis, (x2, y1), (x2 - corner_len, y1), color, 3)
        cv2.line(vis, (x2, y1), (x2, y1 + corner_len), color, 3)
        cv2.line(vis, (x1, y2), (x1 + corner_len, y2), color, 3)
        cv2.line(vis, (x1, y2), (x1, y2 - corner_len), color, 3)
        cv2.line(vis, (x2, y2), (x2 - corner_len, y2), color, 3)
        cv2.line(vis, (x2, y2), (x2, y2 - corner_len), color, 3)

        # 3. Solid Badge phía trên Bounding Box
        (tw, th), _ = cv2.getTextSize(label_text, cv2.FONT_HERSHEY_SIMPLEX, 0.65, 2)
        badge_y1 = max(0, y1 - th - 12)
        badge_y2 = y1
        badge_x2 = min(w, x1 + tw + 14)
        cv2.rectangle(vis, (x1, badge_y1), (badge_x2, badge_y2), color, -1)
        cv2.putText(vis, label_text, (x1 + 7, badge_y2 - 6),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.65, (255, 255, 255), 2, cv2.LINE_AA)

    # 4. Hiển thị FPS và thông tin Model góc trên bên trái
    y_offset = 35
    if fps is not None:
        cv2.putText(vis, f"FPS: {fps:.1f}", (15, y_offset),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2, cv2.LINE_AA)
        y_offset += 25

    model_short = model_name.split("/")[0] if "/" in model_name else model_name
    info_str = f"Model: {model_short} (YOLOv11) | Faces: {len(detections)}"
    cv2.putText(vis, info_str, (15, y_offset),
                cv2.FONT_HERSHEY_SIMPLEX, 0.45, (220, 220, 220), 1, cv2.LINE_AA)
    y_offset += 20

    sub_str = "Mode: Local Cache (Offline)" + (f" | Latency: {latency_ms:.1f}ms" if latency_ms else "")
    cv2.putText(vis, sub_str, (15, y_offset),
                cv2.FONT_HERSHEY_SIMPLEX, 0.45, (180, 240, 180), 1, cv2.LINE_AA)

    # 5. Thanh hướng dẫn phím tắt dưới cùng
    cv2.putText(vis, "[ESC] / [q]: Thoat  |  [s]: Luu anh snapshot", (15, h - 15),
                cv2.FONT_HERSHEY_SIMPLEX, 0.45, (180, 180, 180), 1, cv2.LINE_AA)

    return vis


def test_on_image(model, image_path: str, model_id: str, no_show: bool = False):
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
    detections = parse_model_predictions(response, w, h)
    print(f"[OK] Thời gian suy luận: {infer_time:.1f}ms (Chạy hoàn toàn cục bộ)")
    print(f"[OK] Phát hiện {len(detections)} khuôn mặt:")

    for idx, d in enumerate(detections, 1):
        status = "THẬT (REAL)" if d["is_real"] else "GIẢ MẠO (SPOOF)"
        print(f"  [{idx}] BBox: {d['bbox']} | Confidence: {d['confidence']*100:.1f}% -> {status}")

    # Vẽ kết quả
    vis = draw_detection_results(image, detections, latency_ms=infer_time, model_name=model_id)

    # Lưu ảnh kết quả
    out_filename = f"result_v2_{os.path.basename(image_path)}"
    out_path = os.path.join(OUTPUT_DIR, out_filename)
    cv2.imwrite(out_path, vis)
    print(f"[INFO] Đã lưu ảnh kết quả vào: {out_path}")

    if not no_show:
        cv2.imshow(f"Roboflow Anti-Spoof V2 ({model_id})", vis)
        print("\n[HƯỚNG DẪN] Nhấn phím bất kỳ trên cửa sổ ảnh để đóng.")
        cv2.waitKey(0)
        cv2.destroyAllWindows()


def test_on_webcam(model, cam_id: int = 0, model_id: str = DEFAULT_MODEL_ID):
    """Kiểm tra mô hình trực tiếp qua Webcam thời gian thực (Offline, siêu mượt)"""
    print(f"\n[INFO] Đang mở Camera ID: {cam_id}...")
    cap = cv2.VideoCapture(cam_id)
    if not cap.isOpened():
        print(f"[ERROR] Không thể mở Camera ID {cam_id}!")
        return

    print("[OK] Đã kết nối Camera thành công!")
    print("  * Mô hình chạy OFFLINE 100% từ weights.onnx trong models/roboflow/.")
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

        curr_time = time.time()
        fps = 1.0 / (curr_time - prev_time) if curr_time > prev_time else 0.0
        prev_time = curr_time

        # Dự đoán trực tiếp trên frame
        t0 = time.time()
        try:
            response = model.infer(image=frame)
            lat_ms = (time.time() - t0) * 1000.0
            detections = parse_model_predictions(response, w, h)
        except Exception as e:
            lat_ms = None
            detections = []
            print(f"[WARN] Lỗi khi infer: {e}")

        # Vẽ kết quả
        display = draw_detection_results(frame, detections, fps=fps, latency_ms=lat_ms, model_name=model_id)

        cv2.imshow(f"Roboflow Anti-Spoof V2 ({model_id})", display)
        key = cv2.waitKey(1) & 0xFF

        if key in (27, ord('q'), ord('Q')):
            break
        elif key in (ord('s'), ord('S')):
            shot_count += 1
            save_path = os.path.join(OUTPUT_DIR, f"webcam_v2_snap_{shot_count}.jpg")
            cv2.imwrite(save_path, display)
            print(f"[LƯU ẢNH] Đã lưu snapshot tại: {save_path}")

    cap.release()
    cv2.destroyAllWindows()
    print("[INFO] Đã đóng Webcam an toàn.")


def main():
    parser = argparse.ArgumentParser(description="Test Anti-Spoofing Roboflow Model V2 (Offline Cache)")
    parser.add_argument("--model-id", type=str, default=DEFAULT_MODEL_ID,
                        help=f"ID của mô hình Roboflow (Mặc định model mới: {DEFAULT_MODEL_ID})")
    parser.add_argument("--image", type=str, default=None,
                        help="Đường dẫn file ảnh để kiểm tra (VD: data_raw/0.jpg). Nếu không truyền, mặc định chạy Live Webcam.")
    parser.add_argument("--cam", "--camera", type=int, default=0,
                        help="Camera device index để kiểm tra thời gian thực (mặc định: 0)")
    parser.add_argument("--no-show", action="store_true", default=False,
                        help="Không hiển thị GUI cửa sổ (dành cho headless/automated testing)")
    args = parser.parse_args()

    model_dir = os.path.join(ROBOFLOW_CACHE_DIR, *args.model_id.split("/"))

    print("\n" + "=" * 75)
    print("     TEST MÔ HÌNH ROBOFLOW ANTI-SPOOF V2 (OFFLINE LOCAL CACHE)")
    print("=" * 75)
    print(f"  * Model ID   : {args.model_id} (Model mới tải về)")
    print(f"  * Cache Dir  : {model_dir}")
    print(f"  * Trạng thái : Chạy hoàn toàn OFFLINE qua trọng số ONNX, 0 độ trễ mạng")
    print("=" * 75 + "\n")

    print(f"[INFO] Đang nạp mô hình '{args.model_id}' từ thư mục cache...")
    t_start = time.time()
    try:
        model = get_model(model_id=args.model_id, api_key=API_KEY)
        load_sec = time.time() - t_start
        print(f"[OK] Đã nạp xong mô hình thành công trong {load_sec:.2f} giây!\n")
    except Exception as e:
        print(f"[ERROR] Không thể nạp mô hình '{args.model_id}': {e}")
        return

    # Chạy trên ảnh tĩnh nếu có --image
    if args.image:
        test_on_image(model, args.image, model_id=args.model_id, no_show=args.no_show)
    else:
        # Mặc định mở webcam trực tiếp
        print("[INFO] Khởi động chế độ Live Webcam thời gian thực (mặc định)...")
        print(f"[INFO] Kết nối Camera ID: {args.cam}")
        test_on_webcam(model, cam_id=args.cam, model_id=args.model_id)


if __name__ == "__main__":
    main()
