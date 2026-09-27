# -*- coding: utf-8 -*-
"""
=============================================================================
Test Phát Hiện Kính và Khẩu Trang (Glass & Mask Detection)
=============================================================================
Mô hình YOLO26n Object Detection từ Roboflow:
  - Workspace: 14-lop-11a1-thai-gia-ky
  - Workflow ID: glass-and-mask-vglass-and-mask-q5de1-2-yolo26n-t1-logic
  - Model ID: glass-and-mask-q5de1/2
  - Thư mục Cache: models/roboflow/glass-and-mask-q5de1/2/
  - File trọng số: weights.onnx (~9.8 MB)
  - Kiến trúc: YOLO26 Nano (Object Detection, mAP 96.72%)
  - Nhãn phân loại (4 lớp):
      + glass     : Người dùng đang đeo kính mắt
      + mask      : Người dùng đang đeo khẩu trang
      + no_glass  : Người dùng KHÔNG đeo kính
      + no_mask   : Người dùng KHÔNG đeo khẩu trang

Tính năng kiểm tra:
  1. Chế độ OFFLINE (Mặc định):
     - Chạy trực tiếp từ cache cục bộ `models/roboflow/` thông qua `inference.get_model`.
     - 100% Offline, không phụ thuộc kết nối Internet hay Local Docker Server.
     - Tự động tận dụng GPU (DirectML trên Windows) hoặc CPU đa luồng cực nhanh.
  2. Chế độ WORKFLOW (Tùy chọn --workflow):
     - Sử dụng `inference_sdk.InferenceHTTPClient` kết nối tới Roboflow Workflow.
     - Hỗ trợ server cục bộ (VD: http://localhost:9001) hoặc Roboflow Serverless Cloud.

Cách sử dụng:
  # 1. Test trên ảnh tĩnh (mặc định data_raw/0.jpg - Chế độ Offline):
  py -3.11 tests/test_glass_and_mask_model.py --image data_raw/0.jpg

  # 2. Test trên ảnh không đeo kính:
  py -3.11 tests/test_glass_and_mask_model.py --image data_raw/7.jpg

  # 3. Test hàng loạt toàn bộ thư mục data_raw:
  py -3.11 tests/test_glass_and_mask_model.py --dir data_raw/

  # 4. Test Webcam thời gian thực:
  py -3.11 tests/test_glass_and_mask_model.py --cam 0

  # 5. Test bằng Roboflow Workflow Client (Cloud / Serverless):
  py -3.11 tests/test_glass_and_mask_model.py --image data_raw/0.jpg --workflow

  # 6. Test bằng Roboflow Local Server (Docker localhost:9001):
  py -3.11 tests/test_glass_and_mask_model.py --image data_raw/0.jpg --workflow --api-url http://localhost:9001
=============================================================================
"""

import sys
import os
import time
import glob
import argparse
import warnings

# Tắt các cảnh báo phụ trợ từ inference
os.environ["CORE_MODEL_GAZE_ENABLED"] = "False"
os.environ["CORE_MODEL_SAM_ENABLED"] = "False"
os.environ["CORE_MODEL_SAM3_ENABLED"] = "False"
warnings.filterwarnings("ignore")

if hasattr(sys.stdout, "reconfigure"):
    try:
        sys.stdout.reconfigure(encoding="utf-8", line_buffering=True)
        sys.stderr.reconfigure(encoding="utf-8", line_buffering=True)
    except Exception:
        pass

# Cấu hình đường dẫn
CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
BASE_DIR = os.path.dirname(CURRENT_DIR)
OUTPUT_DIR = os.path.join(CURRENT_DIR, "output", "glass_mask_test")
os.makedirs(OUTPUT_DIR, exist_ok=True)

# Cấu hình thư mục cache cục bộ cho Roboflow models
ROBOFLOW_CACHE_DIR = os.path.join(BASE_DIR, "models", "roboflow")
os.environ["MODEL_CACHE_DIR"] = ROBOFLOW_CACHE_DIR
os.makedirs(ROBOFLOW_CACHE_DIR, exist_ok=True)

import cv2
import numpy as np
import onnxruntime as ort

# =============================================================================
# CẤU HÌNH TĂNG TỐC PHẦN CỨNG (GPU DIRECTML / CUDA)
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

if TARGET_PROVIDERS:
    TARGET_PROVIDERS.append("CPUExecutionProvider")
    os.environ["ONNXRUNTIME_EXECUTION_PROVIDERS"] = ",".join(TARGET_PROVIDERS)
elif "ONNXRUNTIME_EXECUTION_PROVIDERS" in os.environ:
    del os.environ["ONNXRUNTIME_EXECUTION_PROVIDERS"]

# Roboflow Model / Workflow Thông tin
DEFAULT_MODEL_ID = "glass-and-mask-q5de1/2"
API_KEY = "lGvF9eLaX4ZhhERgN5u2"
WORKSPACE_NAME = "14-lop-11a1-thai-gia-ky"
WORKFLOW_ID = "glass-and-mask-vglass-and-mask-q5de1-2-yolo26n-t1-logic"

# Bảng màu hiển thị trực quan cho các nhãn
CLASS_COLORS = {
    "glass": (0, 140, 255),       # Cam - Cảnh báo đeo kính
    "mask": (0, 30, 230),         # Đỏ - Cảnh báo khẩu trang
    "no_glass": (40, 200, 80),    # Xanh lá - An toàn, không kính
    "no_mask": (220, 180, 0),     # Xanh lam/vàng nhạt - An toàn, không khẩu trang
}

CLASS_VN_NAMES = {
    "glass": "DEO KINH",
    "mask": "DEO KHAU TRANG",
    "no_glass": "KHONG KINH",
    "no_mask": "KHONG KHAU TRANG",
}


def parse_predictions(response, img_w: int, img_h: int, conf_threshold: float = 0.4):
    """
    Trích xuất và chuẩn hóa dự đoán từ kết quả inference (hỗ trợ cả get_model và Workflow client).
    Chuyển đổi tọa độ (center x, center y, width, height) -> (x1, y1, x2, y2).
    """
    dets = []
    preds = []

    # Định dạng kết quả từ Workflow Client: [{'predictions': {'predictions': [...]}}]
    if isinstance(response, list) and len(response) > 0:
        first = response[0]
        if isinstance(first, dict) and "predictions" in first:
            p_data = first["predictions"]
            preds = p_data.get("predictions", []) if isinstance(p_data, dict) else p_data
        elif hasattr(first, "predictions"):
            preds = first.predictions
    # Định dạng từ get_model: response.predictions hoặc response[0].predictions
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
            cls_name = str(getattr(p, "class_name", getattr(p, "class_", ""))).lower().strip()
        elif isinstance(p, dict):
            cx = float(p.get("x", 0.0))
            cy = float(p.get("y", 0.0))
            pw = float(p.get("width", 0.0))
            ph = float(p.get("height", 0.0))
            conf = float(p.get("confidence", p.get("score", 0.0)))
            cls_name = str(p.get("class", p.get("class_name", ""))).lower().strip()
        else:
            continue

        if conf < conf_threshold:
            continue

        # Chuẩn hóa nếu tọa độ ở dạng tỉ lệ (0.0 - 1.0)
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

        dets.append({
            "bbox": [x1, y1, x2, y2],
            "confidence": conf,
            "class_name": cls_name,
            "label_vn": CLASS_VN_NAMES.get(cls_name, cls_name.upper())
        })

    return dets


def evaluate_compliance(detections):
    """
    Đánh giá trạng thái tuân thủ chính sách khuôn mặt:
    - has_glass: Phát hiện đeo kính
    - has_mask: Phát hiện đeo khẩu trang
    - is_compliant: Đạt chuẩn (không kính, không khẩu trang)
    """
    has_glass = any(d["class_name"] == "glass" for d in detections)
    has_mask = any(d["class_name"] == "mask" for d in detections)
    no_glass_detected = any(d["class_name"] == "no_glass" for d in detections)
    no_mask_detected = any(d["class_name"] == "no_mask" for d in detections)

    is_compliant = (not has_glass) and (not has_mask)
    return {
        "has_glass": has_glass,
        "has_mask": has_mask,
        "no_glass_detected": no_glass_detected,
        "no_mask_detected": no_mask_detected,
        "is_compliant": is_compliant
    }


def draw_detection_hud(image, detections, fps=None, latency_ms=None,
                       mode_str="Offline Cache (ONNX)", conf_threshold=0.4):
    """Vẽ bounding box và giao diện HUD trực quan cho kết quả phát hiện"""
    vis = image.copy()
    h, w = vis.shape[:2]

    comp = evaluate_compliance(detections)

    for d in detections:
        x1, y1, x2, y2 = d["bbox"]
        cls = d["class_name"]
        color = CLASS_COLORS.get(cls, (200, 200, 200))
        text = f"{d['label_vn']} {d['confidence']*100:.1f}%"

        # 1. Bounding box
        cv2.rectangle(vis, (x1, y1), (x2, y2), color, 2)

        # 2. Corner brackets công nghệ
        clen = min(22, max(8, (x2 - x1) // 5), max(8, (y2 - y1) // 5))
        thick = 3
        cv2.line(vis, (x1, y1), (x1 + clen, y1), color, thick)
        cv2.line(vis, (x1, y1), (x1, y1 + clen), color, thick)
        cv2.line(vis, (x2, y1), (x2 - clen, y1), color, thick)
        cv2.line(vis, (x2, y1), (x2, y1 + clen), color, thick)
        cv2.line(vis, (x1, y2), (x1 + clen, y2), color, thick)
        cv2.line(vis, (x1, y2), (x1, y2 - clen), color, thick)
        cv2.line(vis, (x2, y2), (x2 - clen, y2), color, thick)
        cv2.line(vis, (x2, y2), (x2, y2 - clen), color, thick)

        # 3. Label Badge
        (tw, th_text), _ = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 0.55, 2)
        by1 = max(0, y1 - th_text - 12)
        by2 = y1
        bx2 = min(w, x1 + tw + 14)
        cv2.rectangle(vis, (x1, by1), (bx2, by2), color, -1)
        cv2.putText(vis, text, (x1 + 6, by2 - 5),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 2, cv2.LINE_AA)

    # 4. HUD Status Panel (Góc trên bên trái)
    panel_w = min(460, w - 16)
    panel_h = 160
    overlay = vis.copy()
    cv2.rectangle(overlay, (8, 8), (8 + panel_w, 8 + panel_h), (18, 18, 22), -1)
    cv2.addWeighted(overlay, 0.75, vis, 0.25, 0, vis)
    cv2.rectangle(vis, (8, 8), (8 + panel_w, 8 + panel_h), (60, 60, 75), 1)

    y = 30
    # Tiêu đề
    cv2.putText(vis, "FACE OCCLUSION: GLASS & MASK", (18, y),
                cv2.FONT_HERSHEY_SIMPLEX, 0.58, (255, 255, 255), 2, cv2.LINE_AA)
    y += 24

    # Performance
    perf_str = ""
    if fps is not None:
        perf_str += f"FPS: {fps:.1f} | "
    if latency_ms is not None:
        perf_str += f"Latency: {latency_ms:.1f} ms | "
    perf_str += f"Device: {DEVICE_STR}"
    cv2.putText(vis, perf_str, (18, y),
                cv2.FONT_HERSHEY_SIMPLEX, 0.44, (190, 230, 255), 1, cv2.LINE_AA)
    y += 24

    # Chi tiết Kính
    glass_color = (0, 140, 255) if comp["has_glass"] else (50, 220, 100)
    glass_text = "CO DEO KINH (GLASS DETECTED)" if comp["has_glass"] else "KHONG DEO KINH (NO GLASS)"
    cv2.putText(vis, f"Kinh mat   : {glass_text}", (18, y),
                cv2.FONT_HERSHEY_SIMPLEX, 0.48, glass_color, 1, cv2.LINE_AA)
    y += 22

    # Chi tiết Khẩu trang
    mask_color = (0, 30, 230) if comp["has_mask"] else (50, 220, 100)
    mask_text = "CO DEO KHAU TRANG (MASK DETECTED)" if comp["has_mask"] else "KHONG KHAU TRANG (NO MASK)"
    cv2.putText(vis, f"Khau trang : {mask_text}", (18, y),
                cv2.FONT_HERSHEY_SIMPLEX, 0.48, mask_color, 1, cv2.LINE_AA)
    y += 24

    # Trạng thái tuân thủ Chính sách A
    if comp["is_compliant"]:
        comp_badge = "CHUAN: KHUON MAT RO RANG (CLEAR)"
        badge_color = (0, 200, 50)
    else:
        reasons = []
        if comp["has_glass"]:
            reasons.append("DEO KINH")
        if comp["has_mask"]:
            reasons.append("DEO KHAU TRANG")
        comp_badge = f"CANH BAO: {', '.join(reasons)}"
        badge_color = (0, 140, 255) if (comp["has_glass"] and not comp["has_mask"]) else (0, 50, 240)

    cv2.putText(vis, f"Chinh sach : {comp_badge}", (18, y),
                cv2.FONT_HERSHEY_SIMPLEX, 0.48, badge_color, 2, cv2.LINE_AA)

    # Mode banner góc dưới bên phải HUD
    cv2.putText(vis, f"Mode: {mode_str} (Conf >= {conf_threshold:.2f})", (18, 8 + panel_h - 8),
                cv2.FONT_HERSHEY_SIMPLEX, 0.40, (160, 160, 170), 1, cv2.LINE_AA)

    # 5. Thanh hướng dẫn phím tắt dưới cùng
    cv2.putText(vis, "[ESC]/[q]: Thoat  |  [s]: Snapshot  |  [SPACE]: Tam dung", (15, h - 15),
                cv2.FONT_HERSHEY_SIMPLEX, 0.45, (220, 220, 220), 1, cv2.LINE_AA)

    return vis


def infer_with_model(model, image, is_workflow=False, client=None, img_path=None):
    """
    Thực hiện suy luận:
    - Nếu is_workflow=True: Sử dụng Roboflow InferenceHTTPClient Workflow
    - Nếu is_workflow=False: Sử dụng Local ONNX get_model (Offline)
    """
    t0 = time.time()
    if is_workflow:
        # Client yêu cầu file path hoặc numpy array qua workflow
        # Lưu tạm nếu chỉ có numpy array
        temp_img_path = img_path
        created_temp = False
        if temp_img_path is None or not os.path.exists(temp_img_path):
            temp_img_path = os.path.join(OUTPUT_DIR, "_temp_workflow_frame.jpg")
            cv2.imwrite(temp_img_path, image)
            created_temp = True

        try:
            response = client.run_workflow(
                workspace_name=WORKSPACE_NAME,
                workflow_id=WORKFLOW_ID,
                images={"image": temp_img_path},
                use_cache=True
            )
        finally:
            if created_temp and os.path.exists(temp_img_path):
                try:
                    os.remove(temp_img_path)
                except Exception:
                    pass
    else:
        # Offline ONNX inference trực tiếp trên numpy frame
        response = model.infer(image=image)

    latency_ms = (time.time() - t0) * 1000.0
    return response, latency_ms


def test_on_image(engine, image_path: str, is_workflow=False, conf_threshold=0.4, no_show=False):
    """Kiểm tra mô hình trên 1 ảnh tĩnh"""
    if not os.path.exists(image_path):
        print(f"[ERROR] Không tìm thấy file ảnh: {image_path}")
        return None

    print(f"\n[INFO] Đang xử lý ảnh: {image_path}")
    image = cv2.imread(image_path)
    if image is None:
        print(f"[ERROR] Không thể đọc định dạng ảnh từ: {image_path}")
        return None

    h, w = image.shape[:2]
    mode_str = "Roboflow Workflow" if is_workflow else "Offline Cache (ONNX)"

    model = None if is_workflow else engine
    client = engine if is_workflow else None

    response, latency_ms = infer_with_model(model, image, is_workflow=is_workflow, client=client, img_path=image_path)
    dets = parse_predictions(response, w, h, conf_threshold=conf_threshold)
    comp = evaluate_compliance(dets)

    print(f"[OK] Thời gian suy luận: {latency_ms:.1f}ms (Chế độ: {mode_str}, Phần cứng: {DEVICE_STR})")
    print(f"[OK] Số đối tượng phát hiện: {len(dets)} (Conf >= {conf_threshold:.2f})")
    for idx, d in enumerate(dets, 1):
        print(f"  [{idx}] Nhãn: {d['class_name']} ({d['label_vn']}) | BBox: {d['bbox']} | Độ tin cậy: {d['confidence']*100:.2f}%")

    print(f"[ĐÁNH GIÁ]")
    print(f"  * Kính mắt     : {'CÓ ĐEO KÍNH' if comp['has_glass'] else 'Không phát hiện đeo kính'}")
    print(f"  * Khẩu trang   : {'CÓ ĐEO KHẨU TRANG' if comp['has_mask'] else 'Không phát hiện khẩu trang'}")
    print(f"  * Tuân thủ     : {'ĐẠT (PASS - Rõ khuôn mặt)' if comp['is_compliant'] else 'CẢNH BÁO (FAIL - Cần tháo kính/khẩu trang)'}")

    vis = draw_detection_hud(image, dets, latency_ms=latency_ms, mode_str=mode_str, conf_threshold=conf_threshold)

    base_name = os.path.splitext(os.path.basename(image_path))[0]
    out_file = os.path.join(OUTPUT_DIR, f"{base_name}_glass_mask_result.jpg")
    cv2.imwrite(out_file, vis)
    print(f"[OK] Ảnh kết quả đã được lưu tại: {out_file}")

    if not no_show:
        win_name = "Glass & Mask Detection Test"
        cv2.namedWindow(win_name, cv2.WINDOW_NORMAL)
        cv2.resizeWindow(win_name, min(1080, w), min(800, h))
        cv2.imshow(win_name, vis)
        print("  * Nhấn phím bất kỳ trên cửa sổ ảnh để tiếp tục...")
        cv2.waitKey(0)
        cv2.destroyAllWindows()

    return dets


def test_on_directory(engine, dir_path: str, is_workflow=False, conf_threshold=0.4):
    """Kiểm tra hàng loạt toàn bộ ảnh trong một thư mục"""
    extensions = ("*.jpg", "*.jpeg", "*.png", "*.bmp", "*.webp")
    image_paths = []
    for ext in extensions:
        image_paths.extend(glob.glob(os.path.join(dir_path, ext)))
        image_paths.extend(glob.glob(os.path.join(dir_path, ext.upper())))

    image_paths = sorted(list(set(image_paths)))
    if not image_paths:
        print(f"[WARN] Không tìm thấy ảnh nào trong thư mục: {dir_path}")
        return

    mode_str = "Roboflow Workflow" if is_workflow else "Offline Cache (ONNX)"
    print(f"\n[INFO] Bắt đầu kiểm tra hàng loạt {len(image_paths)} ảnh trong '{dir_path}'...")
    print(f"       Chế độ: {mode_str} | Ngưỡng tin cậy: {conf_threshold:.2f}\n")

    total_glass = 0
    total_mask = 0
    total_compliant = 0
    total_time = 0.0

    model = None if is_workflow else engine
    client = engine if is_workflow else None

    for idx, img_p in enumerate(image_paths, 1):
        image = cv2.imread(img_p)
        if image is None:
            continue
        h, w = image.shape[:2]

        response, latency_ms = infer_with_model(model, image, is_workflow=is_workflow, client=client, img_path=img_p)
        total_time += latency_ms

        dets = parse_predictions(response, w, h, conf_threshold=conf_threshold)
        comp = evaluate_compliance(dets)

        if comp["has_glass"]:
            total_glass += 1
        if comp["has_mask"]:
            total_mask += 1
        if comp["is_compliant"]:
            total_compliant += 1

        status_str = []
        if comp["has_glass"]:
            status_str.append("GLASS")
        if comp["has_mask"]:
            status_str.append("MASK")
        if not status_str:
            status_str.append("CLEAR")

        print(f"[{idx:02d}/{len(image_paths):02d}] {os.path.basename(img_p):<18} ({w}x{h}) -> {', '.join(status_str):<12} ({latency_ms:.1f}ms)")

        vis = draw_detection_hud(image, dets, latency_ms=latency_ms, mode_str=mode_str, conf_threshold=conf_threshold)
        base_name = os.path.splitext(os.path.basename(img_p))[0]
        cv2.imwrite(os.path.join(OUTPUT_DIR, f"{base_name}_glass_mask.jpg"), vis)

    avg_time = total_time / len(image_paths) if image_paths else 0.0
    print("\n" + "=" * 68)
    print("      TỔNG KẾT KIỂM THỬ HÀNG LOẠT PHÁT HIỆN KÍNH & KHẨU TRANG")
    print("=" * 68)
    print(f"  * Tổng số ảnh kiểm thử    : {len(image_paths)}")
    print(f"  * Phát hiện có đeo kính   : {total_glass} ({total_glass*100/len(image_paths):.1f}%)")
    print(f"  * Phát hiện đeo khẩu trang: {total_mask} ({total_mask*100/len(image_paths):.1f}%)")
    print(f"  * Đạt chuẩn khuôn mặt rõ : {total_compliant} ({total_compliant*100/len(image_paths):.1f}%)")
    print(f"  * Độ trễ trung bình/ảnh   : {avg_time:.1f} ms")
    print(f"  * Phần cứng xử lý         : {DEVICE_STR}")
    print(f"  * Thư mục ảnh kết quả     : {OUTPUT_DIR}")
    print("=" * 68 + "\n")


def test_on_webcam(engine, cam_id: int = 0, is_workflow=False, conf_threshold=0.4):
    """Kiểm tra thời gian thực trên Webcam"""
    cap = cv2.VideoCapture(cam_id)
    if not cap.isOpened():
        print(f"[ERROR] Không thể mở Camera ID {cam_id}!")
        return

    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

    win_name = "Live Glass & Mask Detection (Webcam)"
    cv2.namedWindow(win_name, cv2.WINDOW_NORMAL)
    cv2.resizeWindow(win_name, 960, 720)

    mode_str = "Roboflow Workflow" if is_workflow else "Offline Cache (ONNX)"
    model = None if is_workflow else engine
    client = engine if is_workflow else None

    print("\n" + "=" * 68)
    print("  ĐÃ KHỞI ĐỘNG WEBCAM - PHÁT HIỆN KÍNH & KHẨU TRANG THỜI GIAN THỰC")
    print("=" * 68)
    print(f"  * Chế độ suy luận : {mode_str}")
    print(f"  * Thiết bị phần cứng: {DEVICE_STR}")
    print(f"  * Ngưỡng tin cậy  : {conf_threshold:.2f}")
    print("  * [ESC] hoặc [q]  : Thoát chương trình")
    print("  * [s]              : Lưu ảnh chụp màn hình hiện tại")
    print("  * [SPACE]          : Tạm dừng / Tiếp tục")
    print("=" * 68 + "\n")

    fps = 0.0
    prev_time = time.time()
    paused = False
    display = None

    while True:
        if not paused:
            ret, frame = cap.read()
            if not ret:
                print("[WARN] Không nhận được frame từ camera!")
                break

            h, w = frame.shape[:2]
            response, latency_ms = infer_with_model(model, frame, is_workflow=is_workflow, client=client)
            dets = parse_predictions(response, w, h, conf_threshold=conf_threshold)

            # Tính FPS
            curr_time = time.time()
            fps = 0.9 * fps + 0.1 * (1.0 / max(1e-5, (curr_time - prev_time)))
            prev_time = curr_time

            display = draw_detection_hud(frame, dets, fps=fps, latency_ms=latency_ms,
                                         mode_str=mode_str, conf_threshold=conf_threshold)

        if display is not None:
            cv2.imshow(win_name, display)

        key = cv2.waitKey(1) & 0xFF
        if key in (27, ord('q'), ord('Q')):
            print("[INFO] Đã thoát theo yêu cầu của người dùng.")
            break
        elif key in (ord('s'), ord('S')):
            snap_time = int(time.time())
            snap_path = os.path.join(OUTPUT_DIR, f"snapshot_webcam_{snap_time}.jpg")
            if display is not None:
                cv2.imwrite(snap_path, display)
                print(f"[OK] Đã lưu ảnh chụp nhanh vào: {snap_path}")
        elif key == 32:  # Phím SPACE
            paused = not paused
            print(f"[INFO] {'TẠM DỪNG' if paused else 'TIẾP TỤC'} video luồng.")

    cap.release()
    cv2.destroyAllWindows()


def main():
    parser = argparse.ArgumentParser(
        description="Kiểm thử mô hình phát hiện Kính & Khẩu trang (Roboflow YOLO26n / Offline ONNX)",
        formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--image", type=str, default="data_raw/0.jpg",
                        help="Đường dẫn file ảnh tĩnh cần test (mặc định: data_raw/0.jpg)")
    parser.add_argument("--cam", type=int, default=None,
                        help="ID Webcam để test thời gian thực (ví dụ: --cam 0)")
    parser.add_argument("--dir", type=str, default=None,
                        help="Thư mục ảnh để test hàng loạt (ví dụ: --dir data_raw/)")
    parser.add_argument("--conf", type=float, default=0.4,
                        help="Ngưỡng độ tin cậy confidence threshold (mặc định: 0.40)")
    parser.add_argument("--workflow", action="store_true",
                        help="Sử dụng InferenceHTTPClient chạy Roboflow Workflow thay vì Offline ONNX")
    parser.add_argument("--api-url", type=str, default="https://serverless.roboflow.com",
                        help="URL server cho InferenceHTTPClient (mặc định: https://serverless.roboflow.com hoặc http://localhost:9001)")
    parser.add_argument("--api-key", type=str, default=API_KEY,
                        help="API key Roboflow")
    parser.add_argument("--model-id", type=str, default=DEFAULT_MODEL_ID,
                        help=f"Model ID Roboflow (mặc định: {DEFAULT_MODEL_ID})")
    parser.add_argument("--no-show", action="store_true",
                        help="Không hiển thị cửa sổ GUI OpenCV (hữu ích cho chế độ kiểm thử tự động)")

    args = parser.parse_args()

    # Kiểm tra file weights trong cache cục bộ
    cached_model_path = os.path.join(ROBOFLOW_CACHE_DIR, args.model_id.replace("/", os.sep))
    weights_path = os.path.join(cached_model_path, "weights.onnx")
    weights_exist = os.path.exists(weights_path)
    weights_size_mb = (os.path.getsize(weights_path) / (1024 * 1024)) if weights_exist else 0.0

    print("=" * 78)
    print("   TEST MÔ HÌNH PHÁT HIỆN KÍNH VÀ KHẨU TRANG (GLASS & MASK DETECTION)")
    print("=" * 78)
    print(f"  * Model ID          : {args.model_id}")
    print(f"  * Kiến trúc         : YOLO26 Nano (Object Detection, mAP 96.72%)")
    print(f"  * Cache Weights     : {weights_size_mb:.1f} MB ({'ĐÃ TẢI CỤC BỘ' if weights_exist else 'CHƯA CÓ'})")
    print(f"  * Thiết bị phần cứng: {DEVICE_STR}")
    print(f"  * Providers         : {', '.join(AVAILABLE_PROVIDERS)}")
    print(f"  * Conf Threshold    : {args.conf:.2f}")
    print(f"  * Các lớp nhãn      : glass, mask, no_glass, no_mask (4 classes)")
    print(f"  * Chế độ suy luận   : {'Roboflow Workflow' if args.workflow else 'OFFLINE 100% (Local ONNX Weights)'}")
    print(f"  * Thư mục Cache     : {ROBOFLOW_CACHE_DIR}")
    print("=" * 78 + "\n")

    # Khởi tạo Engine tương ứng
    if args.workflow:
        print(f"[INFO] Đang khởi tạo InferenceHTTPClient (URL: {args.api_url})...")
        try:
            from inference_sdk import InferenceHTTPClient, InferenceConfiguration
            client = InferenceHTTPClient(
                api_url=args.api_url,
                api_key=args.api_key
            ).configure(InferenceConfiguration(
                api_key_transport="header"
            ))
            engine = client
            print("[OK] Đã kết nối thành công tới Workflow Engine!\n")
        except Exception as e:
            print(f"[ERROR] Không thể khởi tạo InferenceHTTPClient: {e}")
            if "localhost" in args.api_url:
                print("  [GỢI Ý]: Local server tại localhost:9001 chưa chạy.")
                print("           Bạn có thể bỏ cờ --workflow để chạy trực tiếp OFFLINE từ cache,")
                print("           hoặc khởi chạy Docker inference server: docker run --rm -p 9001:9001 roboflow/roboflow-inference-server-cpu")
            return
    else:
        print(f"[INFO] Đang nạp mô hình YOLO26n từ cache cục bộ ({DEVICE_STR})...")
        t0 = time.time()
        try:
            from inference import get_model
            model = get_model(model_id=args.model_id, api_key=args.api_key)
            load_sec = time.time() - t0
            print(f"[OK] Đã nạp thành công mô hình YOLO26n trong {load_sec:.2f} giây!\n")
            engine = model
        except Exception as e:
            print(f"[ERROR] Không thể nạp mô hình từ cache: {e}")
            print(f"  [GỢI Ý]: Kiểm tra lại đường dẫn: {cached_model_path}")
            return

    # Điều hướng chế độ thực thi
    if args.cam is not None:
        test_on_webcam(engine, cam_id=args.cam, is_workflow=args.workflow, conf_threshold=args.conf)
    elif args.dir is not None:
        test_on_directory(engine, args.dir, is_workflow=args.workflow, conf_threshold=args.conf)
    else:
        test_on_image(engine, args.image, is_workflow=args.workflow, conf_threshold=args.conf, no_show=args.no_show)


if __name__ == "__main__":
    main()
