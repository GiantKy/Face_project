# -*- coding: utf-8 -*-
"""
=============================================================================
Ensemble Pipeline: YOLO_4 + RF-DETR Small + Quality Filter (OpenCV)
=============================================================================
Pipeline kết hợp:
  1. Image Quality Filter (OpenCV): Lọc mờ (Laplacian), lọc sáng/tối (HSV)
  2. Model 1: Anti_Spoof_YOLO_4.pt (YOLO Face Anti-Spoof)
  3. Model 2: RF-DETR Small (Roboflow Transformer Anti-Spoof)
  4. Ensemble Fusion: IoU matching & Weighted Soft-Voting / Strict Spoof Veto

Usage:
  # Test webcam trực tiếp
  py -3.11 tests/test_ensemble_yolo_rfdetr.py --cam 0

  # Test trên 1 ảnh
  py -3.11 tests/test_ensemble_yolo_rfdetr.py --image data_raw/0.jpg
=============================================================================
"""

import sys
import os
import time
import argparse
import warnings
import cv2
import numpy as np

# Tắt cảnh báo thư viện
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

CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
BASE_DIR = os.path.dirname(CURRENT_DIR)

# Trỏ cache của Roboflow
ROBOFLOW_CACHE_DIR = os.path.join(BASE_DIR, "models", "roboflow")
os.environ["MODEL_CACHE_DIR"] = ROBOFLOW_CACHE_DIR

# Import YOLO
from ultralytics import YOLO

# Import inference (Roboflow)
from inference import get_model


# =============================================================================
# 1. MODULE LỌC CHẤT LƯỢNG ẢNH (OPENCV QUALITY FILTER)
# =============================================================================
class ImageQualityFilter:
    def __init__(self, min_blur_var=70.0, min_brightness=40.0, max_brightness=225.0, min_face_size=80):
        """
        :param min_blur_var: Ngưỡng nét tối thiểu theo phương sai Laplacian (nhỏ hơn => mờ)
        :param min_brightness: Độ sáng tối thiểu (V channel trong HSV)
        :param max_brightness: Độ sáng tối đa (tránh cháy sáng)
        :param min_face_size: Chiều rộng/cao tối thiểu của khuôn mặt (px)
        """
        self.min_blur_var = min_blur_var
        self.min_brightness = min_brightness
        self.max_brightness = max_brightness
        self.min_face_size = min_face_size

    def evaluate_quality(self, frame, bbox=None):
        """
        Đánh giá chất lượng của toàn frame hoặc vùng mặt (bbox = [x1, y1, x2, y2])
        Returns: (is_valid: bool, reason: str, metrics: dict)
        """
        h, w = frame.shape[:2]
        if bbox is not None:
            x1, y1, x2, y2 = [int(v) for v in bbox]
            x1, y1 = max(0, x1), max(0, y1)
            x2, y2 = min(w, x2), min(h, y2)
            face_w, face_h = x2 - x1, y2 - y1

            if face_w < self.min_face_size or face_h < self.min_face_size:
                return False, f"Face too small ({face_w}x{face_h} < {self.min_face_size})", {
                    "blur": 0, "brightness": 0, "size": (face_w, face_h)
                }
            crop = frame[y1:y2, x1:x2]
        else:
            crop = frame

        if crop.size == 0:
            return False, "Empty crop", {"blur": 0, "brightness": 0}

        # 1. Đo độ nét (Laplacian Variance)
        gray = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY)
        blur_var = float(cv2.Laplacian(gray, cv2.CV_64F).var())

        # 2. Đo độ sáng (V channel in HSV)
        hsv = cv2.cvtColor(crop, cv2.COLOR_BGR2HSV)
        v_channel = hsv[:, :, 2]
        mean_brightness = float(np.mean(v_channel))

        metrics = {
            "blur_var": round(blur_var, 1),
            "brightness": round(mean_brightness, 1)
        }

        if blur_var < self.min_blur_var:
            return False, f"Blurry frame ({blur_var:.1f} < {self.min_blur_var})", metrics

        if mean_brightness < self.min_brightness:
            return False, f"Too dark ({mean_brightness:.1f} < {self.min_brightness})", metrics

        if mean_brightness > self.max_brightness:
            return False, f"Overexposed ({mean_brightness:.1f} > {self.max_brightness})", metrics

        return True, "Passed", metrics


# =============================================================================
# 2. LOAD HAI MÔ HÌNH: YOLO_4 VÀ RF-DETR SMALL
# =============================================================================
def load_models(yolo_name="Anti_Spoof_YOLO_4.pt"):
    # 1. Load YOLO_4
    yolo_path = os.path.join(BASE_DIR, "models", yolo_name)
    if not os.path.exists(yolo_path):
        # fallback
        yolo_path = os.path.join(BASE_DIR, "models", "Anti_Spoof_YOLO.pt")
    print(f"[INIT] Loading YOLO Model: {yolo_path}")
    yolo_model = YOLO(yolo_path)

    # 2. Load RF-DETR Small
    rfdetr_model_id = "k-thi-gia-s-workspace/face-spoof-detection-liika-owgrl-1-rfdetr-small-t1"
    api_key = "ydUs8YBnVWjyjFFVvcpx"
    print(f"[INIT] Loading RF-DETR Small: {rfdetr_model_id}")
    rfdetr_model = get_model(model_id=rfdetr_model_id, api_key=api_key)

    return yolo_model, rfdetr_model


# =============================================================================
# 3. HELPER CHUẨN HOÁ DỰ ĐOÁN
# =============================================================================
def get_yolo_predictions(yolo_model, frame, conf_threshold=0.45):
    """
    Returns list of dict: [{'bbox': [x1, y1, x2, y2], 'label': 'real'/'spoof', 'conf': float, 'p_real': float}]
    """
    results = yolo_model(frame, conf=conf_threshold, verbose=False)
    detections = []
    for r in results:
        for box in r.boxes:
            x1, y1, x2, y2 = map(int, box.xyxy[0])
            conf = float(box.conf[0])
            cls_id = int(box.cls[0])
            raw_label = yolo_model.names[cls_id].lower()
            
            # Map nhãn
            label = "real" if "real" in raw_label else "spoof"
            p_real = conf if label == "real" else (1.0 - conf)

            detections.append({
                "bbox": [x1, y1, x2, y2],
                "label": label,
                "conf": conf,
                "p_real": p_real
            })
    return detections


def get_rfdetr_predictions(rfdetr_model, frame, conf_threshold=0.45):
    """
    Returns list of dict: [{'bbox': [x1, y1, x2, y2], 'label': 'real'/'spoof', 'conf': float, 'p_real': float}]
    """
    h, w = frame.shape[:2]
    preds = rfdetr_model.infer(frame)
    
    # Trích xuất predictions
    pred_list = []
    if isinstance(preds, list) and len(preds) > 0:
        pred_list = getattr(preds[0], "predictions", [])
    elif hasattr(preds, "predictions"):
        pred_list = preds.predictions

    detections = []
    for p in pred_list:
        cls_name = str(getattr(p, "class_name", "")).lower().strip()
        conf = float(getattr(p, "confidence", 0.0))
        if "background" in cls_name or conf < conf_threshold:
            continue

        cx = float(getattr(p, "x", 0.0))
        cy = float(getattr(p, "y", 0.0))
        pw = float(getattr(p, "width", 0.0))
        ph = float(getattr(p, "height", 0.0))

        # Chuẩn hóa nếu tỷ lệ 0..1
        if 0.0 <= cx <= 1.0 and 0.0 <= pw <= 1.0:
            cx *= w; cy *= h; pw *= w; ph *= h

        x1 = max(0, int(cx - pw / 2.0))
        y1 = max(0, int(cy - ph / 2.0))
        x2 = min(w, int(cx + pw / 2.0))
        y2 = min(h, int(cy + ph / 2.0))

        label = "real" if "real" in cls_name else "spoof"
        p_real = conf if label == "real" else (1.0 - conf)

        detections.append({
            "bbox": [x1, y1, x2, y2],
            "label": label,
            "conf": conf,
            "p_real": p_real
        })
    return detections


def compute_iou(box1, box2):
    x1 = max(box1[0], box2[0])
    y1 = max(box1[1], box2[1])
    x2 = min(box1[2], box2[2])
    y2 = min(box1[3], box2[3])

    inter = max(0, x2 - x1) * max(0, y2 - y1)
    area1 = (box1[2] - box1[0]) * (box1[3] - box1[1])
    area2 = (box2[2] - box2[0]) * (box2[3] - box2[1])
    union = area1 + area2 - inter

    return inter / union if union > 0 else 0.0


# =============================================================================
# 4. ENSEMBLE FUSION LOGIC
# =============================================================================
def ensemble_predictions(yolo_dets, rfdetr_dets, iou_thresh=0.45, w_yolo=0.5, w_rfdetr=0.5, strict_spoof_veto=True):
    """
    Ensemble matching giữa YOLO_4 và RF-DETR:
      - w_yolo, w_rfdetr: Trọng số của từng mô hình (Soft-voting)
      - strict_spoof_veto: Nếu True, khi 1 trong 2 model có độ tin cậy spoof > 0.70 thì lập tức coi là spoof
    """
    final_results = []
    matched_rf = set()

    for y_det in yolo_dets:
        y_box = y_det["bbox"]
        best_iou = 0.0
        best_idx = -1

        for idx, r_det in enumerate(rfdetr_dets):
            if idx in matched_rf:
                continue
            iou = compute_iou(y_box, r_det["bbox"])
            if iou > best_iou:
                best_iou = iou
                best_idx = idx

        if best_iou >= iou_thresh and best_idx != -1:
            matched_rf.add(best_idx)
            r_det = rfdetr_dets[best_idx]

            # Hợp nhất Bounding box (Lấy trung bình tọa độ)
            f_box = [
                int((y_box[i] + r_det["bbox"][i]) / 2) for i in range(4)
            ]

            # Soft Voting: Tính xác suất Real tổng hợp
            p_real_ensemble = (y_det["p_real"] * w_yolo + r_det["p_real"] * w_rfdetr) / (w_yolo + w_rfdetr)

            # Security Veto: Bảo mật cao chống giả mạo
            if strict_spoof_veto:
                if (y_det["label"] == "spoof" and y_det["conf"] >= 0.70) or \
                   (r_det["label"] == "spoof" and r_det["conf"] >= 0.70):
                    final_label = "spoof"
                    final_conf = max(
                        y_det["conf"] if y_det["label"] == "spoof" else 0,
                        r_det["conf"] if r_det["label"] == "spoof" else 0
                    )
                else:
                    final_label = "real" if p_real_ensemble >= 0.5 else "spoof"
                    final_conf = p_real_ensemble if final_label == "real" else (1.0 - p_real_ensemble)
            else:
                final_label = "real" if p_real_ensemble >= 0.5 else "spoof"
                final_conf = p_real_ensemble if final_label == "real" else (1.0 - p_real_ensemble)

            final_results.append({
                "bbox": f_box,
                "label": final_label,
                "conf": round(final_conf, 3),
                "source": "Ensemble (YOLO + RF-DETR)",
                "yolo_res": f"{y_det['label']}:{y_det['conf']:.2f}",
                "rfdetr_res": f"{r_det['label']}:{r_det['conf']:.2f}",
                "agreement": y_det["label"] == r_det["label"]
            })
        else:
            # Chỉ có YOLO bắt được
            final_results.append({
                "bbox": y_box,
                "label": y_det["label"],
                "conf": round(y_det["conf"], 3),
                "source": "YOLO_Only",
                "yolo_res": f"{y_det['label']}:{y_det['conf']:.2f}",
                "rfdetr_res": "N/A",
                "agreement": False
            })

    # Những box chỉ RF-DETR bắt được
    for idx, r_det in enumerate(rfdetr_dets):
        if idx not in matched_rf:
            final_results.append({
                "bbox": r_det["bbox"],
                "label": r_det["label"],
                "conf": round(r_det["conf"], 3),
                "source": "RFDETR_Only",
                "yolo_res": "N/A",
                "rfdetr_res": f"{r_det['label']}:{r_det['conf']:.2f}",
                "agreement": False
            })

    return final_results


# =============================================================================
# 5. PIPELINE THỰC THI CHÍNH
# =============================================================================
def run_pipeline_on_frame(frame, quality_filter, yolo_model, rfdetr_model):
    """
    Thực hiện trọn gói pipeline: Filter -> Detect & Predict (YOLO + RF-DETR) -> Ensemble
    """
    # Bước 1: YOLO dự đoán nhanh trước để xác định các khuôn mặt
    yolo_dets = get_yolo_predictions(yolo_model, frame)
    rfdetr_dets = get_rfdetr_predictions(rfdetr_model, frame)

    # Bước 2: Ghép ensemble
    ensemble_dets = ensemble_predictions(yolo_dets, rfdetr_dets)

    # Bước 3: Đánh giá Quality Filter trên từng khuôn mặt phát hiện
    results = []
    for item in ensemble_dets:
        is_valid, reason, metrics = quality_filter.evaluate_quality(frame, item["bbox"])
        item["quality_passed"] = is_valid
        item["quality_reason"] = reason
        item["metrics"] = metrics
        results.append(item)

    return results


# =============================================================================
# 6. MAIN TEST (WEBCAM / IMAGE)
# =============================================================================
def main():
    parser = argparse.ArgumentParser(description="Test Ensemble YOLO_4 + RF-DETR Small")
    parser.add_argument("--cam", type=int, default=None, help="Camera index (e.g. 0)")
    parser.add_argument("--image", type=str, default=None, help="Path to single test image")
    args = parser.parse_args()

    quality_filter = ImageQualityFilter(min_blur_var=70.0, min_brightness=40.0, max_brightness=225.0)
    yolo_model, rfdetr_model = load_models()

    if args.image:
        if not os.path.exists(args.image):
            print(f"[ERROR] Image not found: {args.image}")
            return
        frame = cv2.imread(args.image)
        t0 = time.time()
        results = run_pipeline_on_frame(frame, quality_filter, yolo_model, rfdetr_model)
        latency = (time.time() - t0) * 1000
        print(f"\n--- KẾT QUẢ INFERENCE ({latency:.1f} ms) ---")
        for i, res in enumerate(results, 1):
            print(f"Face #{i}:")
            print(f"  - Label        : {res['label'].upper()} (Confidence: {res['conf']:.2f})")
            print(f"  - Nguồn        : {res['source']} (YOLO: {res['yolo_res']} | RF-DETR: {res['rfdetr_res']})")
            print(f"  - Độ đồng thuận: {res['agreement']}")
            print(f"  - Lọc OpenCV   : {'PASSED' if res['quality_passed'] else 'REJECTED (' + res['quality_reason'] + ')'}")
            print(f"  - Metrics      : {res['metrics']}")
        return

    # Webcam mode (Mặc định nếu không truyền argument)
    cam_id = args.cam if args.cam is not None else 0
    cap = cv2.VideoCapture(cam_id)
    if not cap.isOpened():
        print(f"[ERROR] Cannot open camera {cam_id}")
        return

    print(f"\n[INFO] Starting Webcam Ensemble Pipeline. Press 'ESC' or 'q' to exit.")
    prev_time = time.time()

    while True:
        ret, frame = cap.read()
        if not ret:
            break
        frame = cv2.flip(frame, 1)

        t_start = time.time()
        results = run_pipeline_on_frame(frame, quality_filter, yolo_model, rfdetr_model)
        proc_time = (time.time() - t_start) * 1000

        # Vẽ kết quả
        for res in results:
            x1, y1, x2, y2 = res["bbox"]
            quality_ok = res["quality_passed"]
            label = res["label"]

            if not quality_ok:
                color = (0, 165, 255) # Cam: Chất lượng ảnh kém
                tag = f"LOW QUALITY: {res['quality_reason']}"
            elif label == "real":
                color = (0, 255, 0)   # Xanh lá: Real
                tag = f"REAL {res['conf']:.2f} ({res['source']})"
            else:
                color = (0, 0, 255)   # Đỏ: Fake/Spoof
                tag = f"SPOOF {res['conf']:.2f} ({res['source']})"

            # Vẽ Box
            cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
            cv2.putText(frame, tag, (x1, max(20, y1 - 10)),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.55, color, 2)

            # Sub-info
            sub_info = f"YOLO: {res['yolo_res']} | RF: {res['rfdetr_res']} | Blur: {res['metrics'].get('blur_var', 0)}"
            cv2.putText(frame, sub_info, (x1, y2 + 18),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.45, (220, 220, 220), 1)

        # FPS
        fps = 1.0 / (time.time() - prev_time + 1e-6)
        prev_time = time.time()
        cv2.putText(frame, f"FPS: {fps:.1f} | Latency: {proc_time:.1f}ms", (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)

        cv2.imshow("Ensemble Pipeline: YOLO_4 + RF-DETR Small + OpenCV Filter", frame)
        key = cv2.waitKey(1) & 0xFF
        if key == 27 or key == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
