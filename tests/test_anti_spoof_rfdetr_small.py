# -*- coding: utf-8 -*-
"""
=============================================================================
Test Anti-Spoofing: RF-DETR Small (Roboflow Detection Transformer)
=============================================================================
Sử dụng mô hình Anti-Spoofing RF-DETR Small từ Roboflow:
  - Model ID: k-thi-gia-s-workspace/face-spoof-detection-liika-owgrl-1-rfdetr-small-t1
  - Thư mục Cache: models/roboflow/k-thi-gia-s-workspace/face-spoof-detection-liika-owgrl-1-rfdetr-small-t1/
  - File trọng số: weights.onnx (~108.9 MB)
  - Kiến trúc: RF-DETR Small (Detection Transformer)
  - Nhãn phân loại: background_class83422, real, spoof (3 lớp)

Lưu ý:
  - RF-DETR Small sử dụng kiến trúc Transformer nên độ chính xác và khả năng
    khái quát hóa cao hơn so với các mô hình CNN truyền thống.
  - Tự động lọc bỏ nhãn 'background_class83422' (vùng nền).
  - Chạy OFFLINE 100% từ thư mục models/roboflow/ không phụ thuộc Internet.

Cách dùng:
  # 1. Test trên Webcam thời gian thực:
  py -3.11 tests/test_anti_spoof_rfdetr_small.py --cam 0

  # 2. Test trên 1 ảnh tĩnh:
  py -3.11 tests/test_anti_spoof_rfdetr_small.py --image data_raw/0.jpg

  # 3. Test hàng loạt trong một thư mục ảnh:
  py -3.11 tests/test_anti_spoof_rfdetr_small.py --dir data_raw/
=============================================================================
"""

import sys
import os
import time
import glob
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
OUTPUT_DIR = os.path.join(CURRENT_DIR, "output", "rfdetr_small_test")
os.makedirs(OUTPUT_DIR, exist_ok=True)

# Trỏ cache của Roboflow về thư mục models/roboflow của repo
ROBOFLOW_CACHE_DIR = os.path.join(BASE_DIR, "models", "roboflow")
os.environ["MODEL_CACHE_DIR"] = ROBOFLOW_CACHE_DIR
os.makedirs(ROBOFLOW_CACHE_DIR, exist_ok=True)

import json
import cv2
import numpy as np
import onnxruntime as ort

# =============================================================================
# CẤU HÌNH TĂNG TỐC PHẦN CỨNG (GPU: DIRECTML / CUDA TRÊN WINDOWS)
# =============================================================================
AVAILABLE_PROVIDERS = ort.get_available_providers()
TARGET_PROVIDERS = []
DEVICE_STR = "CPU"

if "CUDAExecutionProvider" in AVAILABLE_PROVIDERS:
    TARGET_PROVIDERS.append("CUDAExecutionProvider")
    DEVICE_STR = "GPU (CUDA)"
    os.environ["DEVICE"] = "cuda"
if "DmlExecutionProvider" in AVAILABLE_PROVIDERS:
    TARGET_PROVIDERS.append("DmlExecutionProvider")
    if DEVICE_STR == "CPU":
        DEVICE_STR = "GPU (DirectML)"

# Chỉ cấu hình biến môi trường khi có GPU (Dml hoặc CUDA) và dùng dạng chuỗi phân cách dấu phẩy
if TARGET_PROVIDERS:
    TARGET_PROVIDERS.append("CPUExecutionProvider")
    os.environ["ONNXRUNTIME_EXECUTION_PROVIDERS"] = ",".join(TARGET_PROVIDERS)
elif "ONNXRUNTIME_EXECUTION_PROVIDERS" in os.environ:
    del os.environ["ONNXRUNTIME_EXECUTION_PROVIDERS"]

from inference import get_model

# Cấu hình model RF-DETR Small
DEFAULT_MODEL_ID = "k-thi-gia-s-workspace/face-spoof-detection-liika-owgrl-1-rfdetr-small-t1"
SHORT_MODEL_ID = "face-spoof-detection-liika-owgrl-1-rfdetr-small-t1"
API_KEY = "ydUs8YBnVWjyjFFVvcpx"

# Cấu hình Workflow dự phòng
WORKSPACE_NAME = "k-thi-gia-s-workspace"
WORKFLOW_ID = "face-spoof-detection-vface-spoof-detection-liika-owgrl-1-rfdetr-small-t1-logic"

# Nhãn phân loại
CLASS_NAMES = ["background_class83422", "real", "spoof"]
BACKGROUND_CLASS = "background_class83422"

# Ngưỡng tin cậy mặc định
DEFAULT_CONF_THRESHOLD = 0.5


def parse_rfdetr_predictions(response, img_w: int, img_h: int, conf_threshold: float = DEFAULT_CONF_THRESHOLD):
    """
    Trích xuất danh sách bounding box từ kết quả dự đoán RF-DETR Small.
    Lọc bỏ nhãn 'background_class83422' (vùng nền).
    Lọc theo ngưỡng tin cậy conf_threshold.
    Chuyển đổi tọa độ center (x, y, width, height) -> (x1, y1, x2, y2).
    """
    dets = []
    preds = []

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
            cls_name = str(getattr(p, "class_name", "")).lower().strip()
        elif isinstance(p, dict):
            cx = float(p.get("x", 0.0))
            cy = float(p.get("y", 0.0))
            pw = float(p.get("width", 0.0))
            ph = float(p.get("height", 0.0))
            conf = float(p.get("confidence", p.get("score", 0.0)))
            cls_name = str(p.get("class", p.get("class_name", ""))).lower().strip()
        else:
            continue

        # Bỏ qua nhãn background
        if "background" in cls_name:
            continue

        # Lọc bỏ nếu độ tin cậy thấp hơn ngưỡng conf_threshold
        if conf < conf_threshold:
            continue

        # Chuẩn hóa nếu ở dạng tỉ lệ (0.0 - 1.0)
        if 0.0 <= cx <= 1.0 and 0.0 <= pw <= 1.0 and img_w > 1:
            cx *= img_w
            cy *= img_h
            pw *= img_w
            ph *= img_h

        # Tính góc bounding box
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


def draw_detection_results(image, detections, fps=None, latency_ms=None, model_name=DEFAULT_MODEL_ID,
                           device_name=DEVICE_STR, conf_threshold=DEFAULT_CONF_THRESHOLD):
    """Vẽ bounding box và giao diện HUD trực quan cho RF-DETR Small"""
    vis = image.copy()
    h, w = vis.shape[:2]

    for d in detections:
        x1, y1, x2, y2 = d["bbox"]
        is_real = d["is_real"]
        # Màu sắc: Xanh lá (REAL) / Đỏ-cam (SPOOF)
        color = (0, 255, 0) if is_real else (0, 50, 255)
        label_text = f"{d['label']} {d['confidence']*100:.1f}%"

        # 1. Bounding box viền
        cv2.rectangle(vis, (x1, y1), (x2, y2), color, 2)

        # 2. Corner brackets công nghệ cao
        corner_len = min(25, max(10, (x2 - x1) // 4), max(10, (y2 - y1) // 4))
        thick = 3
        cv2.line(vis, (x1, y1), (x1 + corner_len, y1), color, thick)
        cv2.line(vis, (x1, y1), (x1, y1 + corner_len), color, thick)
        cv2.line(vis, (x2, y1), (x2 - corner_len, y1), color, thick)
        cv2.line(vis, (x2, y1), (x2, y1 + corner_len), color, thick)
        cv2.line(vis, (x1, y2), (x1 + corner_len, y2), color, thick)
        cv2.line(vis, (x1, y2), (x1, y2 - corner_len), color, thick)
        cv2.line(vis, (x2, y2), (x2 - corner_len, y2), color, thick)
        cv2.line(vis, (x2, y2), (x2, y2 - corner_len), color, thick)

        # 3. Solid Badge phía trên Bounding Box
        (tw, th_text), _ = cv2.getTextSize(label_text, cv2.FONT_HERSHEY_SIMPLEX, 0.65, 2)
        badge_y1 = max(0, y1 - th_text - 14)
        badge_y2 = y1
        badge_x2 = min(w, x1 + tw + 16)
        cv2.rectangle(vis, (x1, badge_y1), (badge_x2, badge_y2), color, -1)
        cv2.putText(vis, label_text, (x1 + 8, badge_y2 - 6),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.65, (255, 255, 255), 2, cv2.LINE_AA)

    # 4. HUD Panel thông tin góc trên bên trái
    overlay = vis.copy()
    cv2.rectangle(overlay, (8, 8), (480, 146), (20, 20, 25), -1)
    cv2.addWeighted(overlay, 0.7, vis, 0.3, 0, vis)
    cv2.rectangle(vis, (8, 8), (480, 146), (70, 70, 80), 1)

    y_offset = 32
    if fps is not None:
        fps_color = (0, 255, 0) if fps >= 15 else (0, 200, 255) if fps >= 8 else (0, 100, 255)
        cv2.putText(vis, f"FPS: {fps:.1f}", (18, y_offset),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.75, fps_color, 2, cv2.LINE_AA)
        if latency_ms:
            cv2.putText(vis, f"Latency: {latency_ms:.1f} ms", (150, y_offset),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.55, (200, 240, 255), 1, cv2.LINE_AA)
        y_offset += 24

    # Device & Threshold
    dev_color = (0, 255, 0) if "GPU" in device_name else (0, 200, 255)
    cv2.putText(vis, f"Device: {device_name}", (18, y_offset),
                cv2.FONT_HERSHEY_SIMPLEX, 0.48, dev_color, 1, cv2.LINE_AA)
    cv2.putText(vis, f"Conf Thresh: {conf_threshold:.2f}", (230, y_offset),
                cv2.FONT_HERSHEY_SIMPLEX, 0.48, (220, 220, 220), 1, cv2.LINE_AA)
    y_offset += 22

    model_short = model_name.split("/")[-1] if "/" in model_name else model_name
    cv2.putText(vis, f"Model: RF-DETR Small ({model_short})", (18, y_offset),
                cv2.FONT_HERSHEY_SIMPLEX, 0.45, (220, 220, 220), 1, cv2.LINE_AA)
    y_offset += 22

    cv2.putText(vis, f"Arch: Detection Transformer | Faces: {len(detections)}", (18, y_offset),
                cv2.FONT_HERSHEY_SIMPLEX, 0.42, (180, 200, 255), 1, cv2.LINE_AA)
    y_offset += 20

    cv2.putText(vis, "Mode: Local Cache (100% Offline)", (18, y_offset),
                cv2.FONT_HERSHEY_SIMPLEX, 0.42, (150, 255, 150), 1, cv2.LINE_AA)

    # 5. Thanh hướng dẫn phím tắt dưới cùng
    cv2.putText(vis, "[ESC]/[q]: Thoat  |  [s]: Snapshot  |  [SPACE]: Tam dung", (15, h - 15),
                cv2.FONT_HERSHEY_SIMPLEX, 0.45, (200, 200, 200), 1, cv2.LINE_AA)

    return vis


def test_on_image(model, image_path: str, model_id: str, conf_threshold: float = DEFAULT_CONF_THRESHOLD, no_show: bool = False):
    """Kiểm tra mô hình RF-DETR Small trên 1 ảnh tĩnh"""
    if not os.path.exists(image_path):
        print(f"[ERROR] Không tìm thấy file ảnh tại: {image_path}")
        return None

    print(f"\n[INFO] Đang đọc ảnh: {image_path}")
    image = cv2.imread(image_path)
    if image is None:
        print(f"[ERROR] Không thể đọc định dạng ảnh từ: {image_path}")
        return None

    h, w = image.shape[:2]
    print(f"[INFO] Kích thước ảnh: {w}x{h}")

    t0 = time.time()
    response = model.infer(image=image)
    infer_time = (time.time() - t0) * 1000.0

    detections = parse_rfdetr_predictions(response, w, h, conf_threshold=conf_threshold)
    print(f"[OK] Thời gian suy luận: {infer_time:.1f}ms (Device: {DEVICE_STR})")
    print(f"[OK] Phát hiện {len(detections)} khuôn mặt (Conf >= {conf_threshold:.2f}):")

    for idx, d in enumerate(detections, 1):
        status = "THẬT (REAL)" if d["is_real"] else "GIẢ MẠO (SPOOF)"
        print(f"  [{idx}] BBox: {d['bbox']} | {status} | Độ tin cậy: {d['confidence']*100:.2f}%")

    vis = draw_detection_results(image, detections, latency_ms=infer_time, model_name=model_id,
                                 device_name=DEVICE_STR, conf_threshold=conf_threshold)

    # Lưu kết quả
    base_name = os.path.splitext(os.path.basename(image_path))[0]
    out_path = os.path.join(OUTPUT_DIR, f"{base_name}_rfdetr_small_result.jpg")
    cv2.imwrite(out_path, vis)
    print(f"[OK] Đã lưu ảnh kết quả vào: {out_path}")

    if not no_show:
        win_name = "RF-DETR Small Anti-Spoof Test"
        cv2.namedWindow(win_name, cv2.WINDOW_NORMAL)
        cv2.imshow(win_name, vis)
        print("  * Nhấn phím bất kỳ để đóng cửa sổ...")
        cv2.waitKey(0)
        cv2.destroyAllWindows()

    return detections


def test_on_directory(model, dir_path: str, model_id: str, conf_threshold: float = DEFAULT_CONF_THRESHOLD):
    """Kiểm tra hàng loạt toàn bộ ảnh trong một thư mục"""
    extensions = ("*.jpg", "*.jpeg", "*.png", "*.bmp", "*.webp")
    image_paths = []
    for ext in extensions:
        image_paths.extend(glob.glob(os.path.join(dir_path, ext)))
        image_paths.extend(glob.glob(os.path.join(dir_path, ext.upper())))

    image_paths = sorted(list(set(image_paths)))
    if not image_paths:
        print(f"[WARN] Không tìm thấy ảnh nào trong: {dir_path}")
        return

    print(f"\n[INFO] Bắt đầu kiểm tra hàng loạt {len(image_paths)} ảnh trong '{dir_path}' (Conf >= {conf_threshold:.2f})...")
    total_real = 0
    total_spoof = 0
    total_faces = 0
    total_time = 0.0

    for idx, img_p in enumerate(image_paths, 1):
        image = cv2.imread(img_p)
        if image is None:
            continue
        h, w = image.shape[:2]
        t0 = time.time()
        response = model.infer(image=image)
        t_ms = (time.time() - t0) * 1000.0
        total_time += t_ms

        dets = parse_rfdetr_predictions(response, w, h, conf_threshold=conf_threshold)
        total_faces += len(dets)
        real_count = sum(1 for d in dets if d["is_real"])
        spoof_count = sum(1 for d in dets if not d["is_real"])
        total_real += real_count
        total_spoof += spoof_count

        status_str = f"REAL: {real_count}, SPOOF: {spoof_count}" if dets else "NO FACE"
        print(f"[{idx}/{len(image_paths)}] {os.path.basename(img_p)} ({w}x{h}) -> {status_str} ({t_ms:.1f}ms)")

        vis = draw_detection_results(image, dets, latency_ms=t_ms, model_name=model_id,
                                     device_name=DEVICE_STR, conf_threshold=conf_threshold)
        base_name = os.path.splitext(os.path.basename(img_p))[0]
        out_p = os.path.join(OUTPUT_DIR, f"{base_name}_rfdetr_small.jpg")
        cv2.imwrite(out_p, vis)

    avg_time = total_time / len(image_paths) if image_paths else 0.0
    print("\n" + "=" * 65)
    print("           TỔNG KẾT KIỂM THỬ HÀNG LOẠT RF-DETR SMALL")
    print("=" * 65)
    print(f"  * Tổng số ảnh đã xử lý    : {len(image_paths)}")
    print(f"  * Tổng số khuôn mặt       : {total_faces}")
    print(f"  * Khuôn mặt THẬT (REAL)   : {total_real}")
    print(f"  * Khuôn mặt GIẢ MẠO(SPOOF): {total_spoof}")
    print(f"  * Độ trễ trung bình/ảnh   : {avg_time:.1f} ms")
    print(f"  * Thiết bị xử lý          : {DEVICE_STR}")
    print(f"  * Thư mục ảnh kết quả     : {OUTPUT_DIR}")
    print("=" * 65 + "\n")


def test_on_webcam(model, cam_id: int = 0, model_id: str = DEFAULT_MODEL_ID, conf_threshold: float = DEFAULT_CONF_THRESHOLD):
    """Kiểm tra trực tiếp trên Webcam với độ mượt tối đa"""
    cap = cv2.VideoCapture(cam_id)
    if not cap.isOpened():
        print(f"[ERROR] Không thể mở Camera ID {cam_id}!")
        return

    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

    win_name = "RF-DETR Small Anti-Spoofing (Live Webcam)"
    cv2.namedWindow(win_name, cv2.WINDOW_NORMAL)
    cv2.resizeWindow(win_name, 960, 720)

    print("\n" + "=" * 65)
    print("  ĐÃ KHỞI ĐỘNG CAMERA - CHẾ ĐỘ THỜI GIAN THỰC")
    print("=" * 65)
    print(f"  * Thiết bị đang chạy : {DEVICE_STR}")
    print(f"  * Conf Threshold     : {conf_threshold:.2f}")
    print("  * [ESC] hoặc [q]     : Thoát chương trình")
    print("  * [s]                 : Lưu ảnh chụp màn hình hiện tại")
    print("  * [SPACE]             : Tạm dừng / Tiếp tục")
    print("=" * 65 + "\n")

    fps = 0.0
    prev_time = time.time()
    paused = False
    display = None

    while True:
        if not paused:
            ret, frame = cap.read()
            if not ret:
                print("[WARN] Không nhận được khung hình từ Camera.")
                break

            frame = cv2.flip(frame, 1)
            h, w = frame.shape[:2]

            curr_time = time.time()
            fps = 1.0 / (curr_time - prev_time) if curr_time > prev_time else 0.0
            prev_time = curr_time

            t0 = time.time()
            response = model.infer(image=frame)
            latency_ms = (time.time() - t0) * 1000.0

            detections = parse_rfdetr_predictions(response, w, h, conf_threshold=conf_threshold)
            display = draw_detection_results(frame, detections, fps=fps, latency_ms=latency_ms,
                                            model_name=model_id, device_name=DEVICE_STR,
                                            conf_threshold=conf_threshold)

        cv2.imshow(win_name, display)
        key = cv2.waitKey(1) & 0xFF

        if key in (27, ord('q'), ord('Q')):
            break
        elif key in (ord('s'), ord('S')):
            snap_path = os.path.join(OUTPUT_DIR, f"snapshot_rfdetr_small_{int(time.time())}.jpg")
            cv2.imwrite(snap_path, display)
            print(f"[OK] Đã lưu snapshot vào: {snap_path}")
        elif key == ord(' '):
            paused = not paused
            print(f"[INFO] {'TẠM DỪNG' if paused else 'TIẾP TỤC'}")

    cap.release()
    cv2.destroyAllWindows()


def main():
    parser = argparse.ArgumentParser(
        description="Test Anti-Spoofing RF-DETR Small (Roboflow Detection Transformer ~109MB)",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Ví dụ sử dụng:
  # 1. Test webcam:
  py -3.11 tests/test_anti_spoof_rfdetr_small.py --cam 0

  # 2. Test webcam với ngưỡng tin cậy 0.6:
  py -3.11 tests/test_anti_spoof_rfdetr_small.py --cam 0 --conf 0.6

  # 3. Test 1 ảnh:
  py -3.11 tests/test_anti_spoof_rfdetr_small.py --image data_raw/0.jpg

  # 4. Test toàn bộ thư mục:
  py -3.11 tests/test_anti_spoof_rfdetr_small.py --dir data_raw/
        """
    )
    parser.add_argument("--model-id", type=str, default=DEFAULT_MODEL_ID,
                        help=f"ID mô hình (mặc định: {DEFAULT_MODEL_ID})")
    parser.add_argument("--image", type=str, default=None,
                        help="Đường dẫn ảnh tĩnh.")
    parser.add_argument("--dir", type=str, default=None,
                        help="Đường dẫn thư mục ảnh để test hàng loạt.")
    parser.add_argument("--cam", "--camera", type=int, default=0,
                        help="Camera device index (mặc định: 0)")
    parser.add_argument("--conf", "--threshold", type=float, default=DEFAULT_CONF_THRESHOLD,
                        help=f"Ngưỡng độ tin cậy tối thiểu (mặc định: {DEFAULT_CONF_THRESHOLD})")
    parser.add_argument("--no-show", action="store_true", default=False,
                        help="Không hiển thị cửa sổ GUI")
    args = parser.parse_args()

    # Kiểm tra kích thước file weights cục bộ
    # Thử cả 2 đường dẫn (có workspace và alias)
    candidate_weight_paths = [
        os.path.join(ROBOFLOW_CACHE_DIR, *args.model_id.split("/"), "weights.onnx"),
        os.path.join(ROBOFLOW_CACHE_DIR, SHORT_MODEL_ID, "weights.onnx"),
    ]
    weights_path = next((p for p in candidate_weight_paths if os.path.exists(p)), None)
    weights_size_mb = os.path.getsize(weights_path) / (1024 * 1024) if weights_path else 0.0

    print("\n" + "=" * 78)
    print("   TEST MÔ HÌNH RF-DETR SMALL ANTI-SPOOF (DETECTION TRANSFORMER)")
    print("=" * 78)
    print(f"  * Model ID      : {args.model_id}")
    print(f"  * Kiến trúc     : RF-DETR Small (Detection Transformer)")
    print(f"  * Weights       : {weights_size_mb:.1f} MB ({os.path.basename(weights_path) if weights_path else 'chưa có'})")
    print(f"  * Phần cứng     : {DEVICE_STR}")
    print(f"  * Providers     : {', '.join(AVAILABLE_PROVIDERS)}")
    print(f"  * Conf Thresh   : {args.conf:.2f}")
    print(f"  * Nhãn lớp      : real, spoof (lọc bỏ background_class83422)")
    print(f"  * Chế độ        : OFFLINE 100% (local weights.onnx)")
    print(f"  * Cache Dir     : {ROBOFLOW_CACHE_DIR}")

    if DEVICE_STR == "CPU":
        print("\n  [LƯU Ý]: Hiện tại ONNX Runtime đang chạy bằng CPU (chậm).")
        print("           Để kích hoạt GPU (DirectML) tăng tốc mượt mà trên Windows:")
        print("           pip uninstall -y onnxruntime")
        print("           pip install onnxruntime-directml")

    print("=" * 78 + "\n")

    # Nạp model
    print(f"[INFO] Đang nạp mô hình RF-DETR Small từ cache cục bộ ({DEVICE_STR})...")
    t_start = time.time()
    try:
        model = get_model(model_id=args.model_id, api_key=API_KEY)
        load_sec = time.time() - t_start
        print(f"[OK] Đã nạp thành công mô hình trong {load_sec:.2f} giây!\n")
    except Exception as e:
        print(f"[ERROR] Không thể nạp mô hình '{args.model_id}': {e}")
        # Thử fallback qua short model id
        try:
            print(f"[INFO] Thử nạp lại với Short ID '{SHORT_MODEL_ID}'...")
            model = get_model(model_id=SHORT_MODEL_ID, api_key=API_KEY)
            print(f"[OK] Đã nạp thành công với Short ID!")
        except Exception as e2:
            print(f"[FATAL] Nạp mô hình thất bại: {e2}")
            return

    # Điều hướng chế độ test
    if args.image:
        test_on_image(model, args.image, model_id=args.model_id, conf_threshold=args.conf, no_show=args.no_show)
    elif args.dir:
        test_on_directory(model, args.dir, model_id=args.model_id, conf_threshold=args.conf)
    else:
        test_on_webcam(model, cam_id=args.cam, model_id=args.model_id, conf_threshold=args.conf)


if __name__ == "__main__":
    main()
