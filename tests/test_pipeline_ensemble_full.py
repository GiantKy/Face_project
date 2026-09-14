# -*- coding: utf-8 -*-
"""
=============================================================================
Full E-KYC Pipeline: Ensemble (YOLO_4 + RF-DETR Small) & Interactive Capture
=============================================================================
Quy trình thực hiện:
  1. Mở Webcam: Hiển thị giao diện xem trước & canh chỉnh khuôn mặt trong khung Oval.
  2. Chụp ảnh (Phím SPACE / 'c' / 's' hoặc Tự động khi mặt chuẩn):
     - Lưu ảnh gốc vào data_raw/<id>.jpg.
  3. Chạy Full AI Pipeline trên ảnh vừa chụp:
     - Face Detection -> Landmark 468 -> Pose 3D -> Face Align & Crop 224x224
     - ENSEMBLE ANTI-SPOOF: Kết hợp song song:
       + Model 1: Anti_Spoof_YOLO_4.pt
       + Model 2: RF-DETR Small (Transformer trích xuất vân ảnh & đặc trưng sâu)
       + Ensemble Fusion: IoU Bounding Box Matching & Weighted Soft-Voting / Spoof Veto
  4. Active Liveness trên luồng Live Webcam:
     - Blink Detection: Đo EAR (Eye Aspect Ratio)
     - Head Movement: Thử thách quay đầu ngẫu nhiên (Trái / Phải)
  5. Xuất báo cáo eKYC và Dashboard Side-by-Side vào output/<id>/ gồm:
     - 1_pipeline_result.jpg
     - 2_face_crop_224.jpg
     - 3_aligned_full.jpg
     - 4_report.json
     - Cập nhật batch_summary_ensemble.csv

Phím tắt:
  - SPACE hoặc 'c': Chụp ảnh và chạy Full quy trình (AI + Live Liveness)
  - 's': CHỤP NHANH & LƯU NGAY (Chạy AI Model -> Lưu kết quả ngay lập tức)
  - 'a': Bật/Tắt chế độ tự động chụp khi mặt đúng chuẩn trong oval
  - 'r': Khởi tạo lại phiên eKYC mới (ảnh tiếp theo)
  - 'q' hoặc ESC: Thoát chương trình
=============================================================================
"""

import sys
import os
import argparse
import time
import math
import json
import csv
import glob
import unicodedata
import warnings
from enum import Enum
from pathlib import Path
from typing import Optional, Union, Tuple, Dict, Any, List
import cv2
import numpy as np

# Tắt cảnh báo không cần thiết
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

# Cấu hình đường dẫn import
CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
BASE_DIR = os.path.dirname(CURRENT_DIR)

if BASE_DIR not in sys.path:
    sys.path.insert(0, BASE_DIR)

# Trỏ cache của Roboflow
ROBOFLOW_CACHE_DIR = os.path.join(BASE_DIR, "models", "roboflow")
os.environ["MODEL_CACHE_DIR"] = ROBOFLOW_CACHE_DIR
os.makedirs(ROBOFLOW_CACHE_DIR, exist_ok=True)

from ultralytics import YOLO
from inference import get_model

from src.face_detection import FaceDetector
from src.landmark_detection import LandmarkDetector
from src.landmark_detection.draw_landmarks import draw_landmarks
from src.landmark_detection.utils import get_landmark_point
from src.pose_validation import PoseValidator
from src.pose_validation.draw_pose import draw_pose_info
from src.face_alignment_crop import FaceAligner
from src.head_movement import HeadMovementDetector, HeadAction, ChallengeState
from src.illumination import check_illumination_quality, enhance_low_light
from server_module.utils import (
    create_pipeline_result_dashboard,
    create_side_by_side_result
)

DATA_RAW_DIR = os.path.join(BASE_DIR, "data_raw")
OUTPUT_DIR = os.path.join(CURRENT_DIR, "output")


def remove_vietnamese_accents(text: str) -> str:
    """Chuyển đổi văn bản tiếng Việt có dấu thành không dấu để cv2.putText hiển thị chuẩn"""
    if not text:
        return ""
    text = str(text)
    text = text.replace("đ", "d").replace("Đ", "D")
    nfkd = unicodedata.normalize("NFKD", text)
    return "".join([c for c in nfkd if not unicodedata.combining(c)])


def calculate_iou(boxA, boxB):
    """Tính Intersection over Union (IoU) giữa 2 bounding box [x1, y1, x2, y2]"""
    xA = max(boxA[0], boxB[0])
    yA = max(boxA[1], boxB[1])
    xB = min(boxA[2], boxB[2])
    yB = min(boxA[3], boxB[3])

    interArea = max(0, xB - xA) * max(0, yB - yA)
    boxAArea = (boxA[2] - boxA[0]) * (boxA[3] - boxA[1])
    boxBArea = (boxB[2] - boxB[0]) * (boxB[3] - boxB[1])

    iou = interArea / float(boxAArea + boxBArea - interArea + 1e-6)
    return iou


def calc_dist(p1, p2):
    return math.hypot(p1[0] - p2[0], p1[1] - p2[1])


def compute_eye_aspect_ratio(landmarks):
    """Tính EAR (Eye Aspect Ratio) từ MediaPipe 478 landmarks"""
    if not landmarks or len(landmarks) < 468:
        return 0.0, 0.0, 0.0

    l_top = (calc_dist(landmarks[160], landmarks[144]) + calc_dist(landmarks[158], landmarks[153])) / 2.0
    l_width = calc_dist(landmarks[33], landmarks[133])
    ear_left = (l_top / l_width) if l_width > 0 else 0.0

    r_top = (calc_dist(landmarks[385], landmarks[380]) + calc_dist(landmarks[387], landmarks[373])) / 2.0
    r_width = calc_dist(landmarks[362], landmarks[263])
    ear_right = (r_top / r_width) if r_width > 0 else 0.0

    ear_avg = (ear_left + ear_right) / 2.0
    return ear_left, ear_right, ear_avg


def get_next_image_index(data_dir):
    os.makedirs(data_dir, exist_ok=True)
    existing_files = glob.glob(os.path.join(data_dir, "*.*"))
    max_idx = -1
    for file_path in existing_files:
        base_name = os.path.splitext(os.path.basename(file_path))[0]
        if base_name.isdigit():
            idx = int(base_name)
            if idx > max_idx:
                max_idx = idx
    return max_idx + 1


def json_serialize_helper(obj):
    if isinstance(obj, (np.bool_, bool)):
        return bool(obj)
    if isinstance(obj, (np.integer, int)):
        return int(obj)
    if isinstance(obj, (np.floating, float)):
        return float(obj)
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    return str(obj)


# =============================================================================
# 1. ENSEMBLE ANTI-SPOOF DETECTOR (YOLO_4 + RF-DETR SMALL)
# =============================================================================
class EnsembleAntiSpoofDetector:
    def __init__(self, yolo_file="Anti_Spoof_YOLO_4.pt"):
        # 1. Load YOLO_4
        self.yolo_path = os.path.join(BASE_DIR, "models", yolo_file)
        if not os.path.exists(self.yolo_path):
            pts = glob.glob(os.path.join(BASE_DIR, "models", "*Anti_Spoof*.pt"))
            self.yolo_path = pts[0] if pts else os.path.join(BASE_DIR, "models", "Anti_Spoof_YOLO.pt")

        print(f"[INFO] Loading Model 1 (YOLO): {self.yolo_path}")
        self.yolo_model = YOLO(self.yolo_path)
        self.yolo_classes = self.yolo_model.names
        print(f"[OK] YOLO loaded with classes: {self.yolo_classes}")

        # 2. Load RF-DETR Small
        self.rfdetr_model_id = "k-thi-gia-s-workspace/face-spoof-detection-liika-owgrl-1-rfdetr-small-t1"
        self.api_key = "ydUs8YBnVWjyjFFVvcpx"
        print(f"[INFO] Loading Model 2 (RF-DETR Small): {self.rfdetr_model_id}")
        self.rfdetr_model = get_model(model_id=self.rfdetr_model_id, api_key=self.api_key)
        print("[OK] RF-DETR Small loaded successfully!\n")

    def _predict_yolo(self, frame, conf_threshold=0.35):
        results = self.yolo_model(frame, verbose=False, conf=conf_threshold)
        detections = []
        for r in results:
            if hasattr(r, 'boxes') and len(r.boxes) > 0:
                for box in r.boxes:
                    x1, y1, x2, y2 = map(int, box.xyxy[0])
                    conf = float(box.conf[0])
                    cls_id = int(box.cls[0])
                    raw_label = self.yolo_classes.get(cls_id, f"class_{cls_id}").lower()
                    is_real = ("real" in raw_label)
                    p_real = conf if is_real else (1.0 - conf)
                    detections.append({
                        "bbox": [x1, y1, x2, y2],
                        "is_real": is_real,
                        "label": "REAL" if is_real else "SPOOF",
                        "confidence": conf,
                        "p_real": p_real,
                        "raw_class": raw_label
                    })
        return detections

    def _predict_rfdetr(self, frame, conf_threshold=0.35):
        h, w = frame.shape[:2]
        preds = self.rfdetr_model.infer(frame)
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

            if 0.0 <= cx <= 1.0 and 0.0 <= pw <= 1.0:
                cx *= w; cy *= h; pw *= w; ph *= h

            x1 = max(0, int(cx - pw / 2.0))
            y1 = max(0, int(cy - ph / 2.0))
            x2 = min(w, int(cx + pw / 2.0))
            y2 = min(h, int(cy + ph / 2.0))

            is_real = ("real" in cls_name)
            p_real = conf if is_real else (1.0 - conf)
            detections.append({
                "bbox": [x1, y1, x2, y2],
                "is_real": is_real,
                "label": "REAL" if is_real else "SPOOF",
                "confidence": conf,
                "p_real": p_real,
                "raw_class": cls_name
            })
        return detections

    def predict_ensemble(self, frame, conf_threshold=0.35, iou_thresh=0.40, w_yolo=0.5, w_rfdetr=0.5, strict_spoof_veto=True):
        """
        Ghép cặp và ensemble 2 mô hình:
          - IoU matching giữa YOLO và RF-DETR
          - Soft voting trọng số cho xác suất REAL
          - Strict Spoof Veto: Nếu 1 trong 2 model cảnh báo SPOOF với conf >= 0.70 thì VETO thành SPOOF
        """
        yolo_dets = self._predict_yolo(frame, conf_threshold=conf_threshold)
        rfdetr_dets = self._predict_rfdetr(frame, conf_threshold=conf_threshold)

        final_dets = []
        matched_rf = set()

        for y_det in yolo_dets:
            y_box = y_det["bbox"]
            best_iou = 0.0
            best_idx = -1

            for idx, r_det in enumerate(rfdetr_dets):
                if idx in matched_rf:
                    continue
                iou = calculate_iou(y_box, r_det["bbox"])
                if iou > best_iou:
                    best_iou = iou
                    best_idx = idx

            if best_iou >= iou_thresh and best_idx != -1:
                matched_rf.add(best_idx)
                r_det = rfdetr_dets[best_idx]

                # Hợp nhất Bounding box
                f_box = [int((y_box[i] + r_det["bbox"][i]) / 2) for i in range(4)]

                # Soft Voting xác suất REAL
                p_real_ensemble = (y_det["p_real"] * w_yolo + r_det["p_real"] * w_rfdetr) / (w_yolo + w_rfdetr)

                # Strict Spoof Veto
                if strict_spoof_veto and (
                    (not y_det["is_real"] and y_det["confidence"] >= 0.68) or
                    (not r_det["is_real"] and r_det["confidence"] >= 0.68)
                ):
                    is_real = False
                    conf = max(
                        y_det["confidence"] if not y_det["is_real"] else 0.0,
                        r_det["confidence"] if not r_det["is_real"] else 0.0
                    )
                else:
                    is_real = (p_real_ensemble >= 0.50)
                    conf = p_real_ensemble if is_real else (1.0 - p_real_ensemble)

                final_dets.append({
                    "bbox": f_box,
                    "is_real": is_real,
                    "label": "REAL" if is_real else "SPOOF",
                    "confidence": round(float(conf), 4),
                    "source": "Ensemble (Both Models Matched)",
                    "yolo_res": f"{y_det['label']} ({y_det['confidence']*100:.1f}%)",
                    "rfdetr_res": f"{r_det['label']} ({r_det['confidence']*100:.1f}%)",
                    "agreement": y_det["is_real"] == r_det["is_real"],
                    "both_detected": True
                })
            else:
                # Chỉ YOLO bắt được (Thiếu RF-DETR đồng thuận -> Đánh dấu để lược bỏ)
                final_dets.append({
                    "bbox": y_box,
                    "is_real": False,
                    "label": "UNCERTAIN",
                    "confidence": round(float(y_det["confidence"]), 4),
                    "source": "YOLO_Only (No Consensus)",
                    "yolo_res": f"{y_det['label']} ({y_det['confidence']*100:.1f}%)",
                    "rfdetr_res": "N/A",
                    "agreement": False,
                    "both_detected": False
                })

        # Những box chỉ RF-DETR bắt được (Thiếu YOLO đồng thuận -> Đánh dấu để lược bỏ)
        for idx, r_det in enumerate(rfdetr_dets):
            if idx not in matched_rf:
                final_dets.append({
                    "bbox": r_det["bbox"],
                    "is_real": False,
                    "label": "UNCERTAIN",
                    "confidence": round(float(r_det["confidence"]), 4),
                    "source": "RFDETR_Only (No Consensus)",
                    "yolo_res": "N/A",
                    "rfdetr_res": f"{r_det['label']} ({r_det['confidence']*100:.1f}%)",
                    "agreement": False,
                    "both_detected": False
                })

        return final_dets, yolo_dets, rfdetr_dets


# =============================================================================
# 2. HUD & OVAL GUIDE DRAWING UTILITIES
# =============================================================================
def draw_ui_card(image, x, y, w, h, bg_color=(15, 15, 20), alpha=0.85):
    overlay = image.copy()
    cv2.rectangle(overlay, (x, y), (x + w, y + h), bg_color, -1)
    cv2.addWeighted(overlay, alpha, image, 1 - alpha, 0, image)
    cv2.rectangle(image, (x, y), (x + w, y + h), (100, 100, 100), 1)


def draw_oval_face_guide(image, center, axes, is_aligned=False, is_detected=False, color=(0, 255, 127)):
    h, w = image.shape[:2]
    cx, cy = center
    ax, ay = axes

    # Mask oval
    mask = np.zeros((h, w), dtype=np.uint8)
    cv2.ellipse(mask, (cx, cy), (ax, ay), 0, 0, 360, 255, -1)
    outside_mask = (mask == 0)

    # Làm mờ bokeh ngoại vi
    blurred = cv2.GaussianBlur(image, (35, 35), 0)
    image[outside_mask] = (blurred[outside_mask] * 0.60).astype(np.uint8)

    # Viền glow
    glow_color = (int(color[0] * 0.35), int(color[1] * 0.35), int(color[2] * 0.35))
    cv2.ellipse(image, (cx, cy), (ax + 3, ay + 3), 0, 0, 360, glow_color, 1, cv2.LINE_AA)
    cv2.ellipse(image, (cx, cy), (max(10, ax - 3), max(10, ay - 3)), 0, 0, 360, glow_color, 1, cv2.LINE_AA)

    # Viền oval chính
    thickness = 3 if is_aligned else 2
    cv2.ellipse(image, (cx, cy), (ax, ay), 0, 0, 360, color, thickness, cv2.LINE_AA)

    # Biometric ticks ở 4 góc
    tick_len = 16
    cv2.line(image, (cx, cy - ay - tick_len), (cx, cy - ay + 6), color, 2, cv2.LINE_AA)
    cv2.line(image, (cx, cy + ay - 6), (cx, cy + ay + tick_len), color, 2, cv2.LINE_AA)
    cv2.line(image, (cx - ax - tick_len, cy), (cx - ax + 6, cy), color, 2, cv2.LINE_AA)
    cv2.line(image, (cx + ax - 6, cy), (cx + ax + tick_len, cy), color, 2, cv2.LINE_AA)

    return image


def is_point_in_oval(pt: Tuple[float, float], center: Tuple[int, int], axes: Tuple[int, int], tolerance: float = 1.0) -> bool:
    cx, cy = center
    ax, ay = axes
    if ax <= 0 or ay <= 0:
        return False
    norm_x = (float(pt[0]) - cx) / float(ax * tolerance)
    norm_y = (float(pt[1]) - cy) / float(ay * tolerance)
    return (norm_x ** 2 + norm_y ** 2) <= 1.0


def is_face_in_oval(bbox: Union[List[int], Tuple[int, ...]], center: Tuple[int, int], axes: Tuple[int, int], tolerance: float = 1.08) -> bool:
    x1, y1, x2, y2 = bbox
    face_cx = (x1 + x2) / 2.0
    face_cy = (y1 + y2) / 2.0
    return is_point_in_oval((face_cx, face_cy), center, axes, tolerance=tolerance)


def get_oval_masked_frame(frame: np.ndarray, center: Tuple[int, int], axes: Tuple[int, int], blur_ksize: int = 45, dim_factor: float = 0.35) -> np.ndarray:
    h, w = frame.shape[:2]
    cx, cy = center
    ax, ay = axes
    mask = np.zeros((h, w), dtype=np.uint8)
    cv2.ellipse(mask, (cx, cy), (ax, ay), 0, 0, 360, 255, -1)
    outside_mask = (mask == 0)

    masked = frame.copy()
    ksize = blur_ksize if blur_ksize % 2 == 1 else blur_ksize + 1
    blurred = cv2.GaussianBlur(masked, (ksize, ksize), 0)
    masked[outside_mask] = (blurred[outside_mask] * dim_factor).astype(np.uint8)
    return masked


# =============================================================================
# 3. PIPELINE STAGE ENUM
# =============================================================================
class PipelineStage(Enum):
    PREVIEW_ALIGN = 1       # Xem trước & canh chỉnh góc mặt trong oval
    RUN_AI_STATIC = 2       # Chụp & Chạy AI Pipeline (Face -> Landmark -> Pose -> Crop -> Ensemble Anti-Spoof)
    LIVE_BLINK = 3          # Active Liveness: Chớp mắt
    LIVE_HEAD_MOVEMENT = 4  # Active Liveness: Thử thách quay đầu
    FINAL_DECISION = 5      # Xuất Dashboard Side-by-Side & Lưu báo cáo


# =============================================================================
# 4. MAIN PIPELINE FULL WITH ENSEMBLE
# =============================================================================
def main_pipeline_ensemble(cam_id=0, skip_liveness=False, yolo_file="Anti_Spoof_YOLO_4.pt"):
    print("\n" + "=" * 78)
    print("      FULL E-KYC PIPELINE: ENSEMBLE (YOLO_4 + RF-DETR SMALL) & CAPTURE")
    print("=" * 78)
    print(f"  * Thư mục ảnh gốc     : {DATA_RAW_DIR}")
    print(f"  * Thư mục lưu kết quả : {OUTPUT_DIR}")
    print(f"  * Model 1 (YOLO)      : {yolo_file}")
    print(f"  * Model 2 (RF-DETR)   : RF-DETR Small (Transformer)")
    print("  * Phím tắt điều khiển:")
    print("      [SPACE] hoặc [c]  : Chụp ảnh & Chạy Full quy trình (AI + Live Liveness)")
    print("      [s]               : CHỤP NHANH & LƯU NGAY (Chạy Ensemble AI -> Lưu kết quả)")
    print("      [a]               : Bật/Tắt chế độ tự động chụp khi mặt chuẩn")
    print("      [r]               : Khởi tạo phiên eKYC mới tiếp theo")
    print("      [q] hoặc [ESC]    : Thoát")
    print("=" * 78 + "\n")

    os.makedirs(DATA_RAW_DIR, exist_ok=True)
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    # Khởi tạo toàn bộ các Component Models
    print("[INFO] Đang khởi tạo các module thị giác máy tính...")
    detector = FaceDetector()
    landmark_detector = LandmarkDetector()
    pose_validator = PoseValidator()
    aligner = FaceAligner()
    ensemble_anti_spoof = EnsembleAntiSpoofDetector(yolo_file=yolo_file)
    head_movement_detector = HeadMovementDetector(yaw_threshold=16.0, pitch_threshold=12.0, timeout=7.0)
    print("[OK] Đã sẵn sàng toàn bộ hệ thống Models!\n")

    cap = cv2.VideoCapture(cam_id)
    if not cap.isOpened():
        print(f"[ERROR] Không thể mở Camera ID {cam_id}!")
        return

    # Trạng thái
    stage = PipelineStage.PREVIEW_ALIGN
    auto_capture_mode = False
    quick_snapshot_mode = False
    consecutive_center_frames = 0
    is_aligned_good = False
    capture_blocked_frames = 0
    is_light_ok = True
    mean_lum = 100.0

    current_img_idx = get_next_image_index(DATA_RAW_DIR)
    captured_frame = None
    captured_img_path = None
    captured_result_dir = None

    primary_face = None
    landmarks_static = None
    pose_dict_static = None
    pose_valid_static = False
    face_crop_static = None
    aligned_img_static = None
    best_spoof_static = None
    all_spoof_dets = []
    yolo_debug_dets = []
    rfdetr_debug_dets = []

    blink_counter = 0
    blink_state = False
    blink_passed = False

    head_movement_passed = False
    current_head_action = HeadAction.NONE
    head_action_prompt = ""

    final_pass = False
    reasons = []
    final_display_img = None
    final_record = None

    prev_fps_time = time.time()

    def start_new_session():
        nonlocal stage, current_img_idx, captured_frame, captured_img_path, captured_result_dir
        nonlocal primary_face, landmarks_static, pose_dict_static, pose_valid_static
        nonlocal face_crop_static, aligned_img_static, best_spoof_static
        nonlocal all_spoof_dets, yolo_debug_dets, rfdetr_debug_dets
        nonlocal blink_counter, blink_state, blink_passed, head_movement_passed, current_head_action, head_action_prompt
        nonlocal final_pass, reasons, final_display_img, final_record, consecutive_center_frames, quick_snapshot_mode
        nonlocal is_aligned_good, capture_blocked_frames, is_light_ok, mean_lum

        current_img_idx = get_next_image_index(DATA_RAW_DIR)
        stage = PipelineStage.PREVIEW_ALIGN
        captured_frame = None
        captured_img_path = None
        captured_result_dir = None
        quick_snapshot_mode = False

        primary_face = None
        landmarks_static = None
        pose_dict_static = None
        pose_valid_static = False
        face_crop_static = None
        aligned_img_static = None
        best_spoof_static = None
        all_spoof_dets.clear()
        yolo_debug_dets.clear()
        rfdetr_debug_dets.clear()

        blink_counter = 0
        blink_state = False
        blink_passed = False

        head_movement_passed = False
        current_head_action = HeadAction.NONE
        head_action_prompt = ""
        head_movement_detector.reset()

        final_pass = False
        reasons.clear()
        final_display_img = None
        final_record = None
        consecutive_center_frames = 0
        is_aligned_good = False
        capture_blocked_frames = 0
        is_light_ok = True
        mean_lum = 100.0

        print(f"\n[PHIÊN MỚI] Sẵn sàng chụp ảnh ID tiếp theo: {current_img_idx}.jpg")

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        frame = cv2.flip(frame, 1)
        h, w = frame.shape[:2]
        display = frame.copy()

        # Khung Oval trung tâm
        oval_cx = w // 2
        oval_cy = int(h * 0.505)
        oval_ay = int(h * 0.38)
        oval_ax = int(oval_ay * 0.65)
        oval_center = (oval_cx, oval_cy)
        oval_axes = (oval_ax, oval_ay)

        # =====================================================================
        # GIAI ĐOẠN 1: PREVIEW & CANH CHỈNH GÓC MẶT
        # =====================================================================
        if stage == PipelineStage.PREVIEW_ALIGN:
            frame_for_detect = get_oval_masked_frame(frame, oval_center, oval_axes)
            landmarks_live = landmark_detector.detect(frame_for_detect)
            if landmarks_live and len(landmarks_live) >= 468:
                xs = [p[0] for p in landmarks_live]
                ys = [p[1] for p in landmarks_live]
                if not is_point_in_oval(((min(xs) + max(xs)) / 2.0, (min(ys) + max(ys)) / 2.0), oval_center, oval_axes, tolerance=1.15):
                    landmarks_live = None

            pose_valid_live = False
            pose_dict_live = None
            face_in_oval = False
            is_too_far = False
            is_too_close = False
            is_off_center = False
            off_center_hint = ""

            if landmarks_live and len(landmarks_live) >= 468:
                pose_valid_live, _, pose_dict_live = pose_validator.validate(landmarks_live, get_landmark_point)
                xs = [p[0] for p in landmarks_live]
                ys = [p[1] for p in landmarks_live]
                f_cx = (min(xs) + max(xs)) / 2.0
                f_cy = (min(ys) + max(ys)) / 2.0
                f_h = max(ys) - min(ys)

                face_in_oval = is_point_in_oval((f_cx, f_cy), oval_center, oval_axes, tolerance=1.05)
                ideal_h = oval_ay * 1.55
                is_too_far = (f_h < ideal_h * 0.62)
                is_too_close = (f_h > ideal_h * 1.35)

                dx = f_cx - oval_cx
                dy = f_cy - oval_cy
                if abs(dx) > oval_ax * 0.35 or abs(dy) > oval_ay * 0.35:
                    is_off_center = True
                    hints = []
                    if dx > oval_ax * 0.35:
                        hints.append("Qua Trai")
                    elif dx < -oval_ax * 0.35:
                        hints.append("Qua Phai")
                    if dy > oval_ay * 0.35:
                        hints.append("Len Tren")
                    elif dy < -oval_ay * 0.35:
                        hints.append("Xuong Duoi")
                    off_center_hint = f"Dich mat {' + '.join(hints)} vao tam oval"

            # Kiểm tra ánh sáng
            light_info = check_illumination_quality(frame)
            mean_lum = light_info.get("mean_luminance", 100.0)
            is_light_ok = (mean_lum >= 55.0)

            is_aligned_good = (
                (landmarks_live is not None) and
                face_in_oval and
                pose_valid_live and
                not is_too_far and
                not is_too_close and
                not is_off_center and
                is_light_ok
            )

            if is_aligned_good:
                consecutive_center_frames += 1
                guide_color = (0, 255, 127)
                if auto_capture_mode:
                    remain_f = max(0, 25 - consecutive_center_frames)
                    align_msg = f"GIU YEN! Tu dong chup sau: {remain_f} frames..." if remain_f > 0 else "DANG CHUP ANH..."
                else:
                    align_msg = "KHUON MAT CHUAN! NHAN [SPACE] HOAC [c] DE CHUP"
                align_col = (0, 255, 127)
            elif is_too_far:
                guide_color = (0, 165, 255)
                align_msg = "VUI LONG TIEN LAI GAN CAMERA HON (Khuon mat qua nho)..."
                align_col = (0, 165, 255)
                consecutive_center_frames = max(0, consecutive_center_frames - 1)
            elif is_too_close:
                guide_color = (0, 165, 255)
                align_msg = "VUI LONG LUI RA XA CAMERA HON (Khuon mat qua to)..."
                align_col = (0, 165, 255)
                consecutive_center_frames = max(0, consecutive_center_frames - 1)
            elif is_off_center:
                guide_color = (0, 200, 255)
                align_msg = off_center_hint if off_center_hint else "CAN CHINH MAT VAO CHINH GIUA KHUNG OVAL..."
                align_col = (0, 200, 255)
                consecutive_center_frames = max(0, consecutive_center_frames - 1)
            else:
                guide_color = (0, 200, 255)
                align_msg = "VUI LONG NHIN THANG VAO CAMERA (Giu dau thang)..."
                align_col = (0, 200, 255)
                consecutive_center_frames = max(0, consecutive_center_frames - 1)

            if capture_blocked_frames > 0:
                capture_blocked_frames -= 1
                if not is_light_ok:
                    align_msg = f"ANH SANG YEU (L:{mean_lum:.0f}/55) - KHOA CHUP! VUI LONG BAT DEN."
                else:
                    align_msg = "CHUA DUA MAT VAO OVAL - KHONG THE NHAN CHUP!"
                align_col = (0, 0, 255)
                guide_color = (0, 0, 255)

            display = draw_oval_face_guide(
                display, center=oval_center, axes=oval_axes,
                is_aligned=is_aligned_good, is_detected=(landmarks_live is not None), color=guide_color
            )

            # Header Banner
            draw_ui_card(display, 15, 8, w - 30, 48, bg_color=(15, 15, 25), alpha=0.85)
            cv2.putText(display, f"ENSEMBLE PIPELINE (YOLO_4 + RF-DETR) | ID: {current_img_idx}.jpg", (28, 28),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.52, (0, 230, 255), 2, cv2.LINE_AA)
            mode_str = f"Auto-Capture: {'BAT' if auto_capture_mode else 'TAT'} | Luminance: {mean_lum:.0f}/255"
            cv2.putText(display, mode_str, (28, 46), cv2.FONT_HERSHEY_SIMPLEX, 0.38, (180, 180, 180), 1, cv2.LINE_AA)

            # Footer Banner
            bot_y = h - 56
            draw_ui_card(display, 15, bot_y, w - 30, 48, bg_color=(15, 15, 25), alpha=0.88)
            cv2.putText(display, align_msg, (28, bot_y + 22), cv2.FONT_HERSHEY_SIMPLEX, 0.48, align_col, 2, cv2.LINE_AA)

            if is_aligned_good:
                shortcut_hint = "[SPACE]/[c]: CHUP & FULL E-KYC | [s]: CHUP NHANH & LUU | [a]: Auto | [q]: Thoat"
                shortcut_col = (0, 255, 127)
            elif not is_light_ok:
                shortcut_hint = f"[KHOA CHUP: THIEU SANG L:{mean_lum:.0f}/55] Vui long bat den..."
                shortcut_col = (0, 140, 255)
            else:
                shortcut_hint = "[SPACE]/[c]: KHOA CHUP (Dua mat vao oval de mo) | [s]: Chup ngay | [q]: Thoat"
                shortcut_col = (150, 150, 150)

            cv2.putText(display, shortcut_hint, (28, bot_y + 40), cv2.FONT_HERSHEY_SIMPLEX, 0.38, shortcut_col, 1, cv2.LINE_AA)

            if auto_capture_mode and consecutive_center_frames >= 25 and is_aligned_good:
                key_trigger = ord(' ')
            else:
                key_trigger = None

        # =====================================================================
        # GIAI ĐOẠN 2: CHẠY ENSEMBLE AI MODEL TỰ ĐỘNG TRÊN ẢNH VỪA CHỤP
        # =====================================================================
        elif stage == PipelineStage.RUN_AI_STATIC:
            draw_ui_card(display, 20, 20, w - 40, 95, bg_color=(15, 15, 25), alpha=0.92)
            cv2.putText(display, f"DANG CHAY ENSEMBLE AI TREN ANH ID {current_img_idx}.jpg...", (35, 52),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.62, (0, 230, 255), 2)
            cv2.putText(display, "Pipeline: Face Detect -> Landmark -> Pose 3D -> Crop -> [YOLO_4 + RF-DETR Ensemble]", (35, 80),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.44, (0, 255, 127), 1)
            cv2.imshow("Full E-KYC Pipeline 4", display)
            cv2.waitKey(1)

            # 1. Lưu ảnh gốc vào data_raw/<id>.jpg
            captured_img_path = os.path.join(DATA_RAW_DIR, f"{current_img_idx}.jpg")
            cv2.imwrite(captured_img_path, captured_frame)
            print(f"\n[1. CHỤP ẢNH GỐC] Đã lưu ảnh vào: {captured_img_path}")

            # 2. Tạo thư mục output/<id>/
            captured_result_dir = os.path.join(OUTPUT_DIR, str(current_img_idx))
            os.makedirs(captured_result_dir, exist_ok=True)
            all_faces_dir = os.path.join(captured_result_dir, "all_faces_cropped")
            os.makedirs(all_faces_dir, exist_ok=True)

            # 3. Face Detection
            raw_faces = detector.detect(captured_frame)
            faces = [f for f in raw_faces if is_face_in_oval(f["bbox"], oval_center, oval_axes)]
            num_faces = len(faces)
            print(f"[2. Face Detection] Tổng phát hiện: {len(raw_faces)} mặt | Trong oval: {num_faces} mặt.")

            all_face_crops_info = []
            h_f, w_f = captured_frame.shape[:2]

            for idx_f, f_item in enumerate(faces, 1):
                fx1, fy1, fx2, fy2 = f_item["bbox"]
                fx1_c = max(0, min(w_f - 1, fx1))
                fy1_c = max(0, min(h_f - 1, fy1))
                fx2_c = max(0, min(w_f, fx2))
                fy2_c = max(0, min(h_f, fy2))

                crop_f = captured_frame[fy1_c:fy2_c, fx1_c:fx2_c]
                if crop_f.size > 0:
                    crop_filename = f"face_{idx_f}.jpg"
                    crop_save_path = os.path.join(all_faces_dir, crop_filename)
                    cv2.imwrite(crop_save_path, crop_f)
                    all_face_crops_info.append({
                        "face_index": idx_f,
                        "bbox": [fx1, fy1, fx2, fy2],
                        "confidence": round(float(f_item["confidence"]), 4),
                        "crop_file": crop_filename
                    })

            # Primary Face
            primary_face = None
            if faces:
                def get_face_priority(f):
                    bx1, by1, bx2, by2 = f["bbox"]
                    area = (bx2 - bx1) * (by2 - by1)
                    cx, cy = (bx1 + bx2) / 2.0, (by1 + by2) / 2.0
                    dist_center = math.hypot(cx - w_f / 2.0, cy - h_f / 2.0)
                    return area - (dist_center * 10)

                primary_face = max(faces, key=get_face_priority)
                print(f"  -> Đã chọn Primary Face: BBox={primary_face['bbox']} (Conf: {primary_face['confidence']:.2f})")

            # 4. Landmarks
            captured_masked = get_oval_masked_frame(captured_frame, oval_center, oval_axes)
            landmarks_static = landmark_detector.detect(captured_masked)
            if landmarks_static and len(landmarks_static) >= 468:
                xs = [p[0] for p in landmarks_static]
                ys = [p[1] for p in landmarks_static]
                if not is_point_in_oval(((min(xs) + max(xs)) / 2.0, (min(ys) + max(ys)) / 2.0), oval_center, oval_axes, tolerance=1.15):
                    landmarks_static = None
            print(f"[3. Landmarks] Trích xuất được {len(landmarks_static) if landmarks_static else 0} điểm.")

            # 5. Pose 3D
            pose_valid_static = False
            pose_dict_static = None
            if landmarks_static:
                pose_valid_static, _, pose_dict_static = pose_validator.validate(landmarks_static, get_landmark_point)
                if pose_dict_static:
                    print(f"[4. Head Pose 3D] Y={pose_dict_static['yaw']:+.1f}° | P={pose_dict_static['pitch']:+.1f}° | R={pose_dict_static['roll']:+.1f}° -> {'PASS' if pose_valid_static else 'FAIL'}")

            # 6. Face Align & Crop 224x224
            aligned_img_static = None
            face_crop_static = None
            if primary_face is not None:
                face_crop_static = aligner.crop_face(
                    captured_frame, bbox=primary_face["bbox"], padding=25, output_size=(224, 224), mode="bbox"
                )
            elif landmarks_static:
                face_crop_static = aligner.crop_face(
                    captured_frame, landmarks=landmarks_static, padding=25, output_size=(224, 224)
                )

            if landmarks_static:
                aligned_img_static = aligner.align_face(captured_frame, landmarks_static)
            if aligned_img_static is None:
                aligned_img_static = captured_frame.copy()

            if face_crop_static is not None:
                cv2.imwrite(os.path.join(captured_result_dir, "2_face_crop_224.jpg"), face_crop_static)
            if aligned_img_static is not None:
                cv2.imwrite(os.path.join(captured_result_dir, "3_aligned_full.jpg"), aligned_img_static)

            # 7. ENSEMBLE ANTI-SPOOF (YOLO_4 + RF-DETR SMALL)
            captured_light = check_illumination_quality(captured_frame, bbox=primary_face["bbox"] if primary_face else None)
            if captured_light["mean_luminance"] < 75.0:
                input_spoof = enhance_low_light(captured_frame)
                print(f"[Ensemble Preprocess] Luminance={captured_light['mean_luminance']:.1f}. Áp dụng CLAHE.")
            else:
                input_spoof = captured_frame

            t_ens = time.time()
            all_spoof_dets, yolo_debug_dets, rfdetr_debug_dets = ensemble_anti_spoof.predict_ensemble(
                input_spoof, conf_threshold=0.30, iou_thresh=0.40, w_yolo=0.5, w_rfdetr=0.5, strict_spoof_veto=True
            )
            ens_latency = (time.time() - t_ens) * 1000

            # Lọc trong oval
            spoof_res = [sd for sd in all_spoof_dets if is_face_in_oval(sd["bbox"], oval_center, oval_axes)]
            print(f"[5. Ensemble Anti-Spoof ({ens_latency:.1f}ms)] Tìm thấy {len(all_spoof_dets)} vùng -> {len(spoof_res)} trong oval.")

            best_spoof_static = None
            primary_spoof_iou = 0.0
            if primary_face and spoof_res:
                matching_spoofs = [sd for sd in spoof_res if calculate_iou(primary_face["bbox"], sd["bbox"]) > 0.15]
                if matching_spoofs:
                    best_spoof_static = max(matching_spoofs, key=lambda x: x["confidence"])
                    primary_spoof_iou = calculate_iou(primary_face["bbox"], best_spoof_static["bbox"])
                else:
                    best_spoof_static = max(spoof_res, key=lambda x: x["confidence"])
                    primary_spoof_iou = calculate_iou(primary_face["bbox"], best_spoof_static["bbox"])

            if best_spoof_static is None and spoof_res:
                best_spoof_static = spoof_res[0]

            has_any_spoof_in_frame = any(not sd["is_real"] for sd in spoof_res) if spoof_res else False

            if best_spoof_static:
                print(f"  -> Kết quả ENSEMBLE: {best_spoof_static['label']} ({best_spoof_static['confidence']*100:.1f}%) | Nguồn: {best_spoof_static.get('source')} | YOLO: {best_spoof_static.get('yolo_res')} | RF-DETR: {best_spoof_static.get('rfdetr_res')}")

            if quick_snapshot_mode or skip_liveness:
                print("\n[INFO] Chế độ Quick Save / Skip Liveness -> Chuyển thẳng đến Final Decision...")
                blink_passed = True
                head_movement_passed = True
                stage = PipelineStage.FINAL_DECISION
            else:
                print("\n[INFO] Chuyển sang giai đoạn Live Active Liveness (Chớp mắt & Quay đầu)...")
                print("  (Mẹo: Nhấn phím 's' bất cứ lúc nào để lưu kết quả ngay lập tức)")
                stage = PipelineStage.LIVE_BLINK
                blink_counter = 0
                blink_state = False
                blink_passed = False

        # =====================================================================
        # GIAI ĐOẠN 3: ACTIVE LIVENESS - BLINK DETECTION (LIVE WEBCAM)
        # =====================================================================
        elif stage == PipelineStage.LIVE_BLINK:
            frame_for_detect = get_oval_masked_frame(frame, oval_center, oval_axes)
            landmarks_live = landmark_detector.detect(frame_for_detect)
            if landmarks_live and len(landmarks_live) >= 468:
                xs = [p[0] for p in landmarks_live]
                ys = [p[1] for p in landmarks_live]
                if not is_point_in_oval(((min(xs) + max(xs)) / 2.0, (min(ys) + max(ys)) / 2.0), oval_center, oval_axes, tolerance=1.15):
                    landmarks_live = None

            ear_l, ear_r, ear_avg = compute_eye_aspect_ratio(landmarks_live) if landmarks_live else (0.0, 0.0, 0.0)

            if ear_avg > 0.05 and ear_avg < 0.18:
                if not blink_state:
                    blink_state = True
            elif ear_avg >= 0.22:
                if blink_state:
                    blink_counter += 1
                    blink_state = False
                    print(f"  [BLINK] Phát hiện chớp mắt thành công! Tổng số: {blink_counter}/1")

            if blink_counter >= 1:
                blink_passed = True
                print(f"[LIVENESS 1: BLINK] DA XAC NHAN CHOP MAT ({blink_counter} lan) -> PASS!")
                stage = PipelineStage.LIVE_HEAD_MOVEMENT
                current_head_action = head_movement_detector.start_challenge()
                head_action_prompt = head_movement_detector.get_prompt()
                print(f"[LIVENESS 2: HEAD MOVEMENT] Thu thach: {current_head_action.value} -> {head_action_prompt}")

            blink_col = (0, 255, 127) if (blink_state or blink_counter >= 1) else (0, 230, 255)
            display = draw_oval_face_guide(display, center=oval_center, axes=oval_axes,
                                           is_aligned=True, is_detected=(landmarks_live is not None), color=blink_col)
            
            if landmarks_live:
                display = draw_landmarks(display, landmarks_live)

            draw_ui_card(display, 15, 8, w - 30, 48, bg_color=(15, 15, 25), alpha=0.88)
            cv2.putText(display, f"E-KYC BUOC 1/2: THU THACH CHOP MAT (ID: {current_img_idx})", (28, 28),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.52, (0, 230, 255), 2, cv2.LINE_AA)
            cv2.putText(display, f"VUI LONG CHOP MAT TU NHIEN | EAR: {ear_avg:.2f} | Blinks: {blink_counter}/1", (28, 46),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.38, (0, 255, 255), 1, cv2.LINE_AA)

            bot_y = h - 56
            draw_ui_card(display, 15, bot_y, w - 30, 48, bg_color=(15, 15, 25), alpha=0.88)
            b_prog = 1.0 if blink_counter >= 1 else (0.5 if blink_state else 0.0)
            bar_w = w - 80
            cv2.rectangle(display, (28, bot_y + 16), (28 + bar_w, bot_y + 32), (40, 40, 50), -1)
            if b_prog > 0:
                cv2.rectangle(display, (28, bot_y + 16), (28 + int(bar_w * b_prog), bot_y + 32), (0, 255, 127), -1)
            cv2.rectangle(display, (28, bot_y + 16), (28 + bar_w, bot_y + 32), (100, 100, 100), 1)
            prog_label = "DA XAC NHAN CHOP MAT! (100%)" if blink_counter >= 1 else ("DANG CHOP MAT... (50%)" if blink_state else "DANG DOI CHOP MAT... (0%)")
            cv2.putText(display, prog_label, (35, bot_y + 28), cv2.FONT_HERSHEY_SIMPLEX, 0.36, (255, 255, 255), 1, cv2.LINE_AA)

        # =====================================================================
        # GIAI ĐOẠN 4: ACTIVE LIVENESS - HEAD MOVEMENT CHALLENGE
        # =====================================================================
        elif stage == PipelineStage.LIVE_HEAD_MOVEMENT:
            frame_for_detect = get_oval_masked_frame(frame, oval_center, oval_axes)
            landmarks_live = landmark_detector.detect(frame_for_detect)
            if landmarks_live and len(landmarks_live) >= 468:
                xs = [p[0] for p in landmarks_live]
                ys = [p[1] for p in landmarks_live]
                if not is_point_in_oval(((min(xs) + max(xs)) / 2.0, (min(ys) + max(ys)) / 2.0), oval_center, oval_axes, tolerance=1.15):
                    landmarks_live = None

            pose_dict_live = None
            if landmarks_live:
                _, _, pose_dict_live = pose_validator.validate(landmarks_live, get_landmark_point)

            hm_status = head_movement_detector.update(pose_dict_live)
            prompt_str = hm_status.get("prompt", "")
            time_left = hm_status.get("time_left", 0.0)
            progress_val = hm_status.get("progress", 0.0)

            if hm_status["passed"]:
                head_movement_passed = True
                print(f"[LIVENESS 2: HEAD MOVEMENT] DA HOAN THANH CU DONG DAU [{current_head_action.value}] -> PASS!")
                stage = PipelineStage.FINAL_DECISION
            elif hm_status["state"] == "FAILED":
                head_movement_passed = False
                print(f"[LIVENESS 2: HEAD MOVEMENT] HET THOI GIAN THUC HIEN -> FAIL!")
                stage = PipelineStage.FINAL_DECISION

            hm_col = (0, 255, 127) if hm_status["passed"] else (0, 230, 255)
            display = draw_oval_face_guide(display, center=oval_center, axes=oval_axes,
                                           is_aligned=True, is_detected=(landmarks_live is not None), color=hm_col)

            if landmarks_live:
                display = draw_landmarks(display, landmarks_live)

            draw_ui_card(display, 15, 8, w - 30, 48, bg_color=(15, 15, 25), alpha=0.88)
            cv2.putText(display, f"E-KYC BUOC 2/2: THU THACH CU DONG DAU (ID: {current_img_idx})", (28, 28),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.52, (0, 230, 255), 2, cv2.LINE_AA)
            cv2.putText(display, f"{prompt_str.upper()} | Thoi gian: {time_left:.1f}s", (28, 46),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.38, hm_col, 1, cv2.LINE_AA)

            bot_y = h - 56
            draw_ui_card(display, 15, bot_y, w - 30, 48, bg_color=(15, 15, 25), alpha=0.88)
            bar_w = w - 80
            cv2.rectangle(display, (28, bot_y + 16), (28 + bar_w, bot_y + 32), (40, 40, 50), -1)
            fill_w = int(bar_w * progress_val)
            if fill_w > 0:
                cv2.rectangle(display, (28, bot_y + 16), (28 + fill_w, bot_y + 32), (0, 255, 127), -1)
            cv2.rectangle(display, (28, bot_y + 16), (28 + bar_w, bot_y + 32), (100, 100, 100), 1)
            pct_hm = int(progress_val * 100)
            cv2.putText(display, f"TIEN TRINH QUAY DAU: {pct_hm}%", (35, bot_y + 28), cv2.FONT_HERSHEY_SIMPLEX, 0.36, (255, 255, 255), 1, cv2.LINE_AA)

        # =====================================================================
        # GIAI ĐOẠN 5: FINAL DECISION & XUẤT SIDE-BY-SIDE DASHBOARD
        # =====================================================================
        elif stage == PipelineStage.FINAL_DECISION:
            if final_display_img is None:
                # Đánh giá điều kiện PASS
                reasons.clear()
                if primary_face is None:
                    reasons.append("Khong phat hien khuon mat trong khung oval")
                elif num_faces > 1:
                    reasons.append(f"Phat hien nhieu khuon mat ({num_faces} mat) trong khung oval")

                if not pose_valid_static:
                    reasons.append("Goc mat nghieng/khong thang ve phia camera")

                # Đánh giá Anti-spoof từ Ensemble
                if best_spoof_static is None:
                    reasons.append("Khong phat hien duoc dac trung chong gia mao (Anti-Spoof None)")
                elif not best_spoof_static.get("both_detected", False):
                    # CHỈ CÓ 1 MODEL NHẬN DIỆN -> LƯỢC BỎ ẢNH, YÊU CẦU CHỤP LẠI
                    reasons.append(f"Chi co 1 model nhan dien ({best_spoof_static.get('source')}) -> Luoc bo anh (Thieu su dong thuan ca 2 model)")
                elif not best_spoof_static["is_real"]:
                    reasons.append(f"Phat hien gia mao (SPOOF) voi do tin cay {best_spoof_static['confidence']*100:.1f}%")

                if not blink_passed:
                    reasons.append("Chua vuot qua thu thach chop mat (Blink Liveness)")

                if not head_movement_passed:
                    reasons.append("Chua vuot qua thu thach quay dau (Head Movement)")

                final_pass = (len(reasons) == 0)

                # Vẽ Bounding Box và Tag trên ảnh gốc
                res_img = captured_frame.copy()
                if primary_face:
                    bx1, by1, bx2, by2 = primary_face["bbox"]
                    cv2.rectangle(res_img, (bx1, by1), (bx2, by2), (0, 255, 0), 2)

                for sd in spoof_res:
                    sx1, sy1, sx2, sy2 = sd["bbox"]
                    if not sd.get("both_detected", False):
                        scol = (0, 165, 255) # Màu Cam: Cảnh báo thiếu đồng thuận
                        tag = f"DISCARD: 1 Model Only ({sd['confidence']*100:.1f}%)"
                    elif sd["is_real"]:
                        scol = (0, 255, 0)   # Màu Xanh: Real
                        tag = f"REAL {sd['confidence']*100:.1f}% (Ensemble Matched)"
                    else:
                        scol = (0, 0, 255)   # Màu Đỏ: Spoof
                        tag = f"SPOOF {sd['confidence']*100:.1f}% (Ensemble Matched)"

                    cv2.rectangle(res_img, (sx1, sy1), (sx2, sy2), scol, 2)
                    cv2.putText(res_img, tag, (sx1, max(20, sy1 - 8)),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.45, scol, 1, cv2.LINE_AA)
                    # Sub-tag: YOLO & RF-DETR
                    sub_tag = f"YOLO: {sd.get('yolo_res', 'N/A')} | RF: {sd.get('rfdetr_res', 'N/A')}"
                    cv2.putText(res_img, sub_tag, (sx1, sy2 + 16),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.38, (220, 220, 220), 1, cv2.LINE_AA)

                # Tạo Dashboard Side-by-Side
                dashboard_img = create_pipeline_result_dashboard(
                    img_idx=current_img_idx,
                    face_info=primary_face,
                    num_faces=num_faces,
                    pose_info=pose_dict_static,
                    pose_valid=pose_valid_static,
                    anti_spoof_info=best_spoof_static,
                    spoof_iou=primary_spoof_iou,
                    blink_passed=blink_passed,
                    blink_count=blink_counter,
                    head_movement_passed=head_movement_passed,
                    head_action_name=current_head_action.value,
                    final_pass=final_pass,
                    reasons=reasons,
                    face_crop=face_crop_static,
                    target_height=h
                )

                clean_img = res_img.copy()
                verdict_badge = "eKYC: APPROVED" if final_pass else "eKYC: REJECTED"
                badge_col = (0, 255, 0) if final_pass else (0, 0, 255)
                cv2.rectangle(clean_img, (w - 240, 15), (w - 15, 55), (15, 15, 20), -1)
                cv2.rectangle(clean_img, (w - 240, 15), (w - 15, 55), badge_col, 2)
                cv2.putText(clean_img, verdict_badge, (w - 225, 42), cv2.FONT_HERSHEY_SIMPLEX, 0.62, badge_col, 2)

                side_by_side_img = create_side_by_side_result(clean_img, dashboard_img)

                # Lưu các file ảnh
                cv2.imwrite(os.path.join(captured_result_dir, "1_pipeline_result_clean.jpg"), clean_img)
                cv2.imwrite(os.path.join(captured_result_dir, "1_dashboard_panel.jpg"), dashboard_img)
                cv2.imwrite(os.path.join(captured_result_dir, "1_pipeline_side_by_side.jpg"), side_by_side_img)
                cv2.imwrite(os.path.join(captured_result_dir, "1_pipeline_result.jpg"), side_by_side_img)

                final_display_img = side_by_side_img

                # Lưu 4_report.json
                final_record = {
                    "image_id": current_img_idx,
                    "image_name": f"{current_img_idx}.jpg",
                    "raw_image_path": captured_img_path,
                    "output_folder": captured_result_dir,
                    "models_used": {
                        "model_1": yolo_file,
                        "model_2": "RF-DETR Small (Transformer)"
                    },
                    "face_detection": {
                        "face_detected": primary_face is not None,
                        "num_faces_detected": num_faces,
                        "single_person_passed": (num_faces == 1),
                        "primary_face_bbox": primary_face["bbox"] if primary_face else None,
                        "primary_face_confidence": round(primary_face["confidence"], 4) if primary_face else 0.0,
                        "all_faces_cropped_folder": all_faces_dir,
                        "all_faces": all_face_crops_info
                    },
                    "pose_validation": {
                        "is_valid": bool(pose_valid_static),
                        "yaw": round(pose_dict_static["yaw"], 2) if pose_dict_static else 0.0,
                        "pitch": round(pose_dict_static["pitch"], 2) if pose_dict_static else 0.0,
                        "roll": round(pose_dict_static["roll"], 2) if pose_dict_static else 0.0,
                    },
                    "ensemble_anti_spoof": {
                        "label": best_spoof_static["label"] if best_spoof_static else "NONE",
                        "is_real": bool(best_spoof_static["is_real"]) if best_spoof_static else False,
                        "confidence": round(best_spoof_static["confidence"], 4) if best_spoof_static else 0.0,
                        "matched_iou": round(primary_spoof_iou, 4),
                        "source": best_spoof_static.get("source") if best_spoof_static else "NONE",
                        "yolo_detail": best_spoof_static.get("yolo_res") if best_spoof_static else "N/A",
                        "rfdetr_detail": best_spoof_static.get("rfdetr_res") if best_spoof_static else "N/A",
                        "agreement": best_spoof_static.get("agreement") if best_spoof_static else False,
                        "all_ensemble_detections": spoof_res
                    },
                    "active_liveness": {
                        "blink_passed": bool(blink_passed),
                        "blink_count": int(blink_counter),
                        "head_movement_passed": bool(head_movement_passed),
                        "head_action": current_head_action.value,
                    },
                    "final_verdict": "APPROVED" if final_pass else "REJECTED",
                    "reasons": reasons
                }

                out_json_path = os.path.join(captured_result_dir, "4_report.json")
                with open(out_json_path, "w", encoding="utf-8") as f:
                    json.dump(final_record, f, ensure_ascii=False, indent=2, default=json_serialize_helper)

                # Lưu CSV
                batch_csv_path = os.path.join(OUTPUT_DIR, "batch_summary_ensemble.csv")
                file_exists = os.path.exists(batch_csv_path)
                with open(batch_csv_path, "a", newline="", encoding="utf-8-sig") as f:
                    writer = csv.writer(f)
                    if not file_exists:
                        writer.writerow([
                            "Image ID", "Verdict", "Num Faces", "Ensemble Label", "Ensemble Conf",
                            "YOLO Detail", "RF-DETR Detail", "Pose Valid", "Blink", "Head Movement", "Reasons", "Output Folder"
                        ])
                    writer.writerow([
                        f"{current_img_idx}.jpg",
                        final_record["final_verdict"],
                        num_faces,
                        final_record["ensemble_anti_spoof"]["label"],
                        final_record["ensemble_anti_spoof"]["confidence"],
                        final_record["ensemble_anti_spoof"]["yolo_detail"],
                        final_record["ensemble_anti_spoof"]["rfdetr_detail"],
                        "PASS" if final_record["pose_validation"]["is_valid"] else "FAIL",
                        "PASS" if blink_passed else "FAIL",
                        f"PASS ({current_head_action.value})" if head_movement_passed else f"FAIL ({current_head_action.value})",
                        "; ".join(reasons) if reasons else "None",
                        captured_result_dir
                    ])

                print("\n" + "=" * 68)
                print(f"  [HOÀN TẤT eKYC ID: {current_img_idx}] Kết quả: {final_record['final_verdict']}")
                print(f"  * Ensemble Verdict : {final_record['ensemble_anti_spoof']['label']} ({final_record['ensemble_anti_spoof']['confidence']*100:.1f}%)")
                print(f"  * Chi tiết YOLO_4  : {final_record['ensemble_anti_spoof']['yolo_detail']}")
                print(f"  * Chi tiết RF-DETR : {final_record['ensemble_anti_spoof']['rfdetr_detail']}")
                print(f"  * Ảnh kết quả lưu  : {captured_result_dir}")
                print("=" * 68 + "\n")

            display = final_display_img.copy()
            disp_h, disp_w = display.shape[:2]
            draw_ui_card(display, 20, disp_h - 70, disp_w - 40, 50, bg_color=(15, 15, 20), alpha=0.85)
            cv2.putText(display, "[r]: Tiep tuc chup anh tiep theo | [q]: Thoat", (35, disp_h - 38),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 230, 255), 2)

        # Thanh FPS
        curr_time = time.time()
        fps = 1.0 / (curr_time - prev_fps_time) if curr_time > prev_fps_time else 0.0
        prev_fps_time = curr_time
        cv2.putText(display, f"FPS: {fps:.1f}", (w - 120, 45), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 200, 200), 1)

        cv2.imshow("Full E-KYC Pipeline 4", display)

        key = cv2.waitKey(1) & 0xFF
        if key == 27 or key == ord('q'):
            break

        elif key == ord('r'):
            start_new_session()

        elif key == ord('a') and stage == PipelineStage.PREVIEW_ALIGN:
            auto_capture_mode = not auto_capture_mode
            print(f"[INFO] Auto-Capture: {'BẬT' if auto_capture_mode else 'TẮT'}")

        # Phím 's': Chụp ảnh nhanh (Snapshot Mode) và lưu kết quả ngay lập tức
        elif (key == ord('s') or key == ord('S')):
            if stage == PipelineStage.PREVIEW_ALIGN:
                if not is_aligned_good:
                    capture_blocked_frames = 40
                    if not is_light_ok:
                        print(f"\n[CHẶN CHỤP] Ánh sáng quá yếu (Luminance={mean_lum:.1f} < 55.0)! Vui lòng bật đèn.")
                    else:
                        print("\n[CHẶN CHỤP] Vui lòng đưa khuôn mặt vào giữa khung oval và nhìn thẳng.")
                else:
                    captured_frame = frame.copy()
                    quick_snapshot_mode = True
                    stage = PipelineStage.RUN_AI_STATIC
                    print(f"\n[QUICK SAVE] Đã kích hoạt Chụp nhanh & Lưu ngay cho ID: {current_img_idx}!")
            elif stage in (PipelineStage.LIVE_BLINK, PipelineStage.LIVE_HEAD_MOVEMENT):
                print("\n[QUICK SAVE] Bỏ qua Liveness tiếp theo và Lưu kết quả ngay lập tức!")
                blink_passed = True
                head_movement_passed = True
                stage = PipelineStage.FINAL_DECISION

        # Phím SPACE hoặc 'c': Chụp ảnh và chạy Full quy trình eKYC
        elif (key == 32 or key == ord('c') or key_trigger == ord(' ')) and stage == PipelineStage.PREVIEW_ALIGN:
            if not is_aligned_good:
                capture_blocked_frames = 40
                if not is_light_ok:
                    print(f"\n[CHẶN CHỤP] Ánh sáng quá yếu (Luminance={mean_lum:.1f} < 55.0)! Vui lòng bật đèn.")
                else:
                    print("\n[CHẶN CHỤP] Vui lòng đưa khuôn mặt vào giữa khung oval và nhìn thẳng.")
            else:
                captured_frame = frame.copy()
                quick_snapshot_mode = False
                stage = PipelineStage.RUN_AI_STATIC
                print(f"\n[TRIGGER] Đã kích hoạt chụp ảnh Full Quy trình cho ID: {current_img_idx}!")

    cap.release()
    cv2.destroyAllWindows()
    print("[INFO] Đã đóng chương trình Pipeline Ensemble an toàn.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Full E-KYC Pipeline: Ensemble (YOLO_4 + RF-DETR Small)")
    parser.add_argument("--cam", type=int, default=0, help="Camera device index (mặc định 0)")
    parser.add_argument("--static", "--skip-liveness", action="store_true", help="Chế độ chụp và lưu AI nhanh, bỏ qua thử thách Liveness")
    parser.add_argument("--yolo", type=str, default="Anti_Spoof_YOLO_4.pt", help="File trọng số YOLO (mặc định Anti_Spoof_YOLO_4.pt)")
    args = parser.parse_args()

    try:
        main_pipeline_ensemble(
            cam_id=args.cam,
            skip_liveness=getattr(args, 'static', False),
            yolo_file=args.yolo
        )
    except KeyboardInterrupt:
        print("\n[INFO] Đã dừng pipeline theo yêu cầu của người dùng.")
