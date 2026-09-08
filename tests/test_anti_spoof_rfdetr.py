# -*- coding: utf-8 -*-
"""
=============================================================================
Test Anti-Spoofing: RF-DETR Nano (Roboflow Detection Transformer)
=============================================================================
Sử dụng mô hình Anti-Spoofing RF-DETR Nano từ Roboflow Workflow:
  - Thư mục Cache: models/roboflow/face-spoof-detection-liika-qopyy-1-rfdetr-nano-t5/
  - File trọng số: weights.onnx (~108 MB)
  - Kiến trúc: RF-DETR Nano (Detection Transformer) - Chính xác cao hơn YOLOv11n
  - Nhãn phân loại: background_class83422, real, spoof (3 lớp)
  - Model ID: ks-workspace-hatfd/face-spoof-detection-liika-qopyy-1-rfdetr-nano-t5

Lưu ý quan trọng:
  - RF-DETR Nano có kích thước ~108MB (lớn hơn ~10x so với YOLOv11n ~10MB)
    do sử dụng kiến trúc Transformer (nhiều attention layers, encoder-decoder).
  - Độ chính xác cao hơn nhưng tốc độ inference chậm hơn YOLOv11n.
  - Có thêm nhãn 'background_class83422' (vùng nền, không phải khuôn mặt).

Chế độ hoạt động:
  - MẶC ĐỊNH: Chạy OFFLINE 100% bằng trọng số cục bộ trong models/roboflow/.
    + Không phụ thuộc mạng Internet, không có độ trễ kết nối.
  - Tùy chọn `--workflow`: Chạy qua Roboflow Workflow Inference HTTP Client.

Cách chạy:
  # 1. Test trên Webcam thời gian thực:
  py -3.11 tests/test_anti_spoof_rfdetr.py --cam 0

  # 2. Test trên ảnh tĩnh:
  py -3.11 tests/test_anti_spoof_rfdetr.py --image data_raw/0.jpg

  # 3. So sánh với YOLOv11n (chạy song song):
  py -3.11 tests/test_anti_spoof_rfdetr.py --cam 0
  py -3.11 tests/test_anti_spoof_roboflow_v2.py --cam 0
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
OUTPUT_DIR = os.path.join(CURRENT_DIR, "output", "rfdetr_test")
os.makedirs(OUTPUT_DIR, exist_ok=True)

# Trỏ cache của Roboflow về thư mục models/roboflow của repo
ROBOFLOW_CACHE_DIR = os.path.join(BASE_DIR, "models", "roboflow")
os.environ["MODEL_CACHE_DIR"] = ROBOFLOW_CACHE_DIR
os.makedirs(ROBOFLOW_CACHE_DIR, exist_ok=True)

import cv2
import numpy as np
from inference import get_model

# Cấu hình model RF-DETR Nano
# Lưu ý: model_id chứa workspace prefix vì đây là workflow model
DEFAULT_MODEL_ID = "ks-workspace-hatfd/face-spoof-detection-liika-qopyy-1-rfdetr-nano-t5"
API_KEY = "LiYT7osRW01duX3ao91S"

# Cấu hình Workflow dự phòng
WORKSPACE_NAME = "ks-workspace-hatfd"
WORKFLOW_ID = "face-spoof-detection-vface-spoof-detection-liika-qopyy-1-rfdetr-nano-t5-logic"

# Nhãn lớp (3 lớp thay vì 2 như YOLOv11n)
CLASS_NAMES = ["background_class83422", "real", "spoof"]
# Nhãn background sẽ bị bỏ qua khi hiển thị
BACKGROUND_CLASS = "background_class83422"


def parse_rfdetr_predictions(response, img_w: int, img_h: int):
    """
    Trích xuất danh sách bounding box từ kết quả dự đoán RF-DETR.
    Lọc bỏ nhãn 'background_class83422' (vùng nền, không phải khuôn mặt).
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

        # BỎ QUA nhãn background (vùng nền, không phải khuôn mặt)
        if "background" in cls_name:
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
    """Vẽ bounding box và thông số trực quan công nghệ cao cho RF-DETR"""
    vis = image.copy()
    h, w = vis.shape[:2]

    for d in detections:
        x1, y1, x2, y2 = d["bbox"]
        is_real = d["is_real"]
        # Màu sắc: Xanh lá (REAL) / Đỏ-cam (SPOOF)
        color = (0, 255, 0) if is_real else (0, 50, 255)
        label_text = f"{d['label']} {d['confidence']*100:.1f}%"

        # 1. Bounding box viền dày hơn cho RF-DETR
        cv2.rectangle(vis, (x1, y1), (x2, y2), color, 2)

        # 2. Corner brackets công nghệ cao
        corner_len = min(25, (x2 - x1) // 4, (y2 - y1) // 4)
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

    # 4. Thông tin góc trên bên trái - FPS, Model, Architecture
    y_offset = 35
    if fps is not None:
        fps_color = (0, 255, 0) if fps >= 15 else (0, 200, 255) if fps >= 8 else (0, 0, 255)
        cv2.putText(vis, f"FPS: {fps:.1f}", (15, y_offset),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, fps_color, 2, cv2.LINE_AA)
        y_offset += 28

    # Hiển thị tên model ngắn gọn
    model_short = model_name.split("/")[-1] if "/" in model_name else model_name
    info_str = f"Model: RF-DETR Nano ({model_short}) | Faces: {len(detections)}"
    cv2.putText(vis, info_str, (15, y_offset),
                cv2.FONT_HERSHEY_SIMPLEX, 0.45, (220, 220, 220), 1, cv2.LINE_AA)
    y_offset += 22

    arch_str = "Arch: Detection Transformer | Classes: real, spoof"
    cv2.putText(vis, arch_str, (15, y_offset),
                cv2.FONT_HERSHEY_SIMPLEX, 0.4, (200, 200, 255), 1, cv2.LINE_AA)
    y_offset += 20

    sub_str = "Mode: Local Cache (Offline)" + (f" | Latency: {latency_ms:.1f}ms" if latency_ms else "")
    cv2.putText(vis, sub_str, (15, y_offset),
                cv2.FONT_HERSHEY_SIMPLEX, 0.45, (180, 240, 180), 1, cv2.LINE_AA)

    # 5. Thanh hướng dẫn phím tắt dưới cùng
    cv2.putText(vis, "[ESC] / [q]: Thoat  |  [s]: Luu anh snapshot", (15, h - 15),
                cv2.FONT_HERSHEY_SIMPLEX, 0.45, (180, 180, 180), 1, cv2.LINE_AA)

    return vis


def test_on_image(model, image_path: str, model_id: str, no_show: bool = False):
    """Kiểm tra mô hình RF-DETR trên 1 ảnh tĩnh"""
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
    print("[INFO] Đang chạy inference RF-DETR Nano (có thể chậm hơn YOLO ~2-5x)...")
    t0 = time.time()
    response = model.infer(image=image)
    infer_time = (time.time() - t0) * 1000.0

    # Trích xuất kết quả (lọc bỏ background)
    detections = parse_rfdetr_predictions(response, w, h)
    print(f"[OK] Thời gian suy luận RF-DETR: {infer_time:.1f}ms (Chạy hoàn toàn cục bộ)")
    print(f"[OK] Phát hiện {len(detections)} khuôn mặt (đã lọc bỏ background_class):")

    for idx, d in enumerate(detections, 1):
        status = "THẬT (REAL)" if d["is_real"] else "GIẢ MẠO (SPOOF)"
        print(f"  [{idx}] BBox: {d['bbox']} | Confidence: {d['confidence']*100:.1f}% -> {status}")

    # Vẽ kết quả
    vis = draw_detection_results(image, detections, latency_ms=infer_time, model_name=model_id)

    # Lưu ảnh kết quả
    out_filename = f"result_rfdetr_{os.path.basename(image_path)}"
    out_path = os.path.join(OUTPUT_DIR, out_filename)
    cv2.imwrite(out_path, vis)
    print(f"[INFO] Đã lưu ảnh kết quả vào: {out_path}")

    if not no_show:
        cv2.imshow(f"RF-DETR Anti-Spoof ({model_id})", vis)
        print("\n[HƯỚNG DẪN] Nhấn phím bất kỳ trên cửa sổ ảnh để đóng.")
        cv2.waitKey(0)
        cv2.destroyAllWindows()


def test_on_webcam(model, cam_id: int = 0, model_id: str = DEFAULT_MODEL_ID):
    """Kiểm tra mô hình RF-DETR trực tiếp qua Webcam thời gian thực"""
    print(f"\n[INFO] Đang mở Camera ID: {cam_id}...")
    cap = cv2.VideoCapture(cam_id)
    if not cap.isOpened():
        print(f"[ERROR] Không thể mở Camera ID {cam_id}!")
        return

    print("[OK] Đã kết nối Camera thành công!")
    print("  * Mô hình RF-DETR Nano chạy OFFLINE 100% từ weights.onnx (108MB).")
    print("  * LƯU Ý: RF-DETR chậm hơn YOLO, FPS có thể thấp hơn (~5-15 FPS trên CPU).")
    print("  * Nhấn phím [q] hoặc [ESC] để thoát.")
    print("  * Nhấn phím [s] để lưu ảnh chụp hiện tại.\n")

    prev_time = time.time()
    shot_count = 0
    frame_count = 0
    fps_smooth = 0.0

    while True:
        ret, frame = cap.read()
        if not ret:
            print("[WARN] Mất tín hiệu Webcam.")
            break

        frame = cv2.flip(frame, 1)
        h, w = frame.shape[:2]

        curr_time = time.time()
        dt = curr_time - prev_time if curr_time > prev_time else 0.001
        fps_instant = 1.0 / dt
        # Smoothed FPS (exponential moving average)
        fps_smooth = 0.8 * fps_smooth + 0.2 * fps_instant if frame_count > 0 else fps_instant
        prev_time = curr_time
        frame_count += 1

        # Dự đoán trực tiếp trên frame
        t0 = time.time()
        try:
            response = model.infer(image=frame)
            lat_ms = (time.time() - t0) * 1000.0
            detections = parse_rfdetr_predictions(response, w, h)
        except Exception as e:
            lat_ms = None
            detections = []
            print(f"[WARN] Lỗi khi infer: {e}")

        # Vẽ kết quả
        display = draw_detection_results(frame, detections, fps=fps_smooth, latency_ms=lat_ms, model_name=model_id)

        cv2.imshow(f"RF-DETR Anti-Spoof ({model_id.split('/')[-1]})", display)
        key = cv2.waitKey(1) & 0xFF

        if key in (27, ord('q'), ord('Q')):
            break
        elif key in (ord('s'), ord('S')):
            shot_count += 1
            save_path = os.path.join(OUTPUT_DIR, f"webcam_rfdetr_snap_{shot_count}.jpg")
            cv2.imwrite(save_path, display)
            print(f"[LƯU ẢNH] Đã lưu snapshot tại: {save_path}")

    cap.release()
    cv2.destroyAllWindows()
    print("[INFO] Đã đóng Webcam an toàn.")


def test_on_webcam_workflow(cam_id: int = 0):
    """Kiểm tra qua Roboflow Workflow (online, cần kết nối mạng)"""
    try:
        from inference_sdk import InferenceHTTPClient, InferenceConfiguration
    except ImportError:
        print("[ERROR] Cần cài đặt: pip install inference-sdk")
        return

    print(f"\n[INFO] Khởi tạo Workflow Client kết nối tới Roboflow Cloud...")
    client = InferenceHTTPClient(
        api_url="https://serverless.roboflow.com",
        api_key=API_KEY
    ).configure(InferenceConfiguration(
        api_key_transport="header"
    ))

    print(f"[OK] Workflow: {WORKFLOW_ID}")
    print(f"[OK] Workspace: {WORKSPACE_NAME}")

    cap = cv2.VideoCapture(cam_id)
    if not cap.isOpened():
        print(f"[ERROR] Không thể mở Camera ID {cam_id}!")
        return

    print("[OK] Đã kết nối Camera. Chế độ WORKFLOW (online).")
    print("  * Nhấn [q] / [ESC] để thoát.\n")

    prev_time = time.time()

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        frame = cv2.flip(frame, 1)
        h, w = frame.shape[:2]

        curr_time = time.time()
        fps = 1.0 / (curr_time - prev_time) if curr_time > prev_time else 0.0
        prev_time = curr_time

        t0 = time.time()
        try:
            result = client.run_workflow(
                workspace_name=WORKSPACE_NAME,
                workflow_id=WORKFLOW_ID,
                images={"image": frame},
                use_cache=True
            )
            lat_ms = (time.time() - t0) * 1000.0
            detections = parse_rfdetr_predictions(result, w, h)
        except Exception as e:
            lat_ms = None
            detections = []
            print(f"[WARN] Workflow error: {e}")

        display = draw_detection_results(frame, detections, fps=fps, latency_ms=lat_ms,
                                         model_name="RF-DETR Workflow")

        cv2.imshow("RF-DETR Anti-Spoof (Workflow)", display)
        key = cv2.waitKey(1) & 0xFF
        if key in (27, ord('q'), ord('Q')):
            break

    cap.release()
    cv2.destroyAllWindows()


def main():
    parser = argparse.ArgumentParser(
        description="Test Anti-Spoofing RF-DETR Nano (Detection Transformer - 108MB)",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Ví dụ sử dụng:
  # Test webcam (mặc định):
  py -3.11 tests/test_anti_spoof_rfdetr.py --cam 0

  # Test ảnh tĩnh:
  py -3.11 tests/test_anti_spoof_rfdetr.py --image data_raw/0.jpg

  # Chạy qua Workflow (online):
  py -3.11 tests/test_anti_spoof_rfdetr.py --cam 0 --workflow
        """
    )
    parser.add_argument("--model-id", type=str, default=DEFAULT_MODEL_ID,
                        help=f"ID mô hình RF-DETR (mặc định: {DEFAULT_MODEL_ID})")
    parser.add_argument("--image", type=str, default=None,
                        help="Đường dẫn ảnh tĩnh. Nếu không truyền, mặc định chạy Webcam.")
    parser.add_argument("--cam", "--camera", type=int, default=0,
                        help="Camera device index (mặc định: 0)")
    parser.add_argument("--no-show", action="store_true", default=False,
                        help="Không hiển thị GUI (dành cho headless)")
    parser.add_argument("--workflow", action="store_true", default=False,
                        help="Chạy qua Roboflow Workflow (online) thay vì model cục bộ")
    args = parser.parse_args()

    # Header thông tin
    model_dir = os.path.join(ROBOFLOW_CACHE_DIR, "face-spoof-detection-liika-qopyy-1-rfdetr-nano-t5")
    weights_path = os.path.join(model_dir, "weights.onnx")
    weights_size_mb = os.path.getsize(weights_path) / (1024 * 1024) if os.path.exists(weights_path) else 0

    print("\n" + "=" * 78)
    print("   TEST MÔ HÌNH RF-DETR NANO ANTI-SPOOF (DETECTION TRANSFORMER)")
    print("=" * 78)
    print(f"  * Model ID    : {args.model_id}")
    print(f"  * Kiến trúc   : RF-DETR Nano (Detection Transformer)")
    print(f"  * Weights     : {weights_size_mb:.1f} MB (weights.onnx)")
    print(f"  * Nhãn        : real, spoof (+ background bị lọc)")
    print(f"  * Input Size  : 640x640")
    print(f"  * Cache Dir   : {model_dir}")
    if args.workflow:
        print(f"  * Chế độ      : WORKFLOW (online qua Roboflow Cloud)")
    else:
        print(f"  * Chế độ      : OFFLINE 100% (local weights.onnx)")
    print("=" * 78 + "\n")

    # Chế độ Workflow
    if args.workflow:
        if args.image:
            print("[WARN] Workflow mode chưa hỗ trợ ảnh tĩnh, chuyển sang webcam.")
        test_on_webcam_workflow(cam_id=args.cam)
        return

    # Nạp model local
    print(f"[INFO] Đang nạp mô hình RF-DETR Nano '{args.model_id}' từ cache...")
    print(f"[INFO] LƯU Ý: File weights 108MB, quá trình nạp có thể mất 5-15 giây...")
    t_start = time.time()
    try:
        model = get_model(model_id=args.model_id, api_key=API_KEY)
        load_sec = time.time() - t_start
        print(f"[OK] Đã nạp xong mô hình thành công trong {load_sec:.2f} giây!\n")
    except Exception as e:
        print(f"[ERROR] Không thể nạp mô hình '{args.model_id}': {e}")
        print(f"\n[GỢI Ý] Kiểm tra thư mục cache: {model_dir}")
        print(f"[GỢI Ý] Các file cần có: weights.onnx, class_names.txt, environment.json, model_type.json")
        return

    # Chạy test
    if args.image:
        test_on_image(model, args.image, model_id=args.model_id, no_show=args.no_show)
    else:
        print("[INFO] Khởi động chế độ Live Webcam thời gian thực...")
        print(f"[INFO] Camera ID: {args.cam}")
        test_on_webcam(model, cam_id=args.cam, model_id=args.model_id)


if __name__ == "__main__":
    main()
