# -*- coding: utf-8 -*-
"""
=============================================================================
Full E-KYC Pipeline RF-DETR Small: RF-DETR Small Anti-Spoofing + Active Liveness
=============================================================================
Quy trình thực hiện toàn diện (End-to-End eKYC Verification Pipeline RF-DETR Small):
  1. Mở Webcam: Hiển thị giao diện xem trước & khung oval bán nguyệt căn chỉnh khuôn mặt.
     - Vùng bên ngoài khung oval được làm mờ (Gaussian Blur) và giảm sáng (Bokeh effect).
     - Khóa chụp ảnh (Capture Lock): Nếu khuôn mặt chưa đưa vào đúng khung oval hoặc
       chưa nhìn thẳng / thiếu sáng, hệ thống sẽ chặn không cho chụp và hiện cảnh báo.
  2. Chụp ảnh (Phím SPACE / 'c' hoặc Tự động khi mặt chuẩn trong oval):
     - Lưu ảnh gốc vào data_raw/<id>.jpg (đánh số tăng dần tiếp theo).
  3. Chạy AI Models trên ảnh vừa chụp:
     - Face Detection (YOLOv8) -> Landmarks (MediaPipe) -> Pose 3D -> Face Align & Crop 224x224.
     - RF-DETR Small Anti-Spoofing (Detection Transformer):
       + Model ID: k-thi-gia-s-workspace/face-spoof-detection-liika-owgrl-1-rfdetr-small-t1
       + Weights: ~109 MB (RF-DETR Small - Detection Transformer Architecture)
       + Nhãn: real, spoof (+ background_class83422 được lọc bỏ tự động)
       + Chạy OFFLINE 100% từ cache models/roboflow/, không phụ thuộc mạng.
  4. Bắt đầu Active Liveness trên luồng Live Webcam:
     - Blink Detection: Yêu cầu người dùng chớp mắt (đo EAR).
     - Head Movement Challenge: Thử thách quay đầu ngẫu nhiên (Trái/Phải).
  5. Tổng hợp toàn bộ dữ liệu & Đưa ra quyết định cuối cùng (Final eKYC Decision).
  6. Hiển thị giao diện kết quả:
     - Giao diện song song (Side-by-Side): Ảnh khuôn mặt bên trái + Dashboard thông số bên phải.
  7. Lưu toàn bộ kết quả vào output/pipeline_rfdetr_small/<id>/ gồm:
     - 1_pipeline_result.jpg (Ảnh song song Side-by-Side)
     - 1_pipeline_result_clean.jpg (Ảnh khuôn mặt sạch)
     - 1_dashboard_panel.jpg (Bảng Dashboard độc lập)
     - 2_face_crop_224.jpg
     - 3_aligned_full.jpg
     - 4_report.json
     - Cập nhật batch_summary_rfdetr_small.csv.

Phím điều khiển:
  - SPACE / 'c' : Chụp ảnh và bắt đầu quy trình eKYC (yêu cầu mặt trong oval)
  - 's'         : CHỤP NHANH & LƯU NGAY (Bỏ qua Active Liveness, yêu cầu mặt trong oval)
  - 'a'         : Bật / Tắt chế độ tự động chụp khi khuôn mặt đạt chuẩn trong oval
  - 'r'         : Khởi tạo lại phiên eKYC mới (tiếp tục ảnh mới)
  - 'q' / ESC   : Thoát chương trình

Cách chạy:
  # 1. Full Pipeline RF-DETR Small trên Webcam:
  py -3.11 tests/test_pipeline_rfdetr_small.py --cam 0

  # 2. Chụp nhanh (bỏ qua Liveness):
  py -3.11 tests/test_pipeline_rfdetr_small.py --cam 0 --static
=============================================================================
"""

import sys
import os
import time
import math
import json
import csv
import glob
import argparse
import unicodedata
import warnings
from enum import Enum
from pathlib import Path
from typing import Optional, Union, Tuple, Dict, Any, List

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

# Thiết lập đường dẫn import tới Face-Project/
CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
BASE_DIR = os.path.dirname(CURRENT_DIR)

if BASE_DIR not in sys.path:
    sys.path.insert(0, BASE_DIR)

# Trỏ cache của Roboflow về thư mục models/roboflow của repo (chạy offline 100%)
ROBOFLOW_CACHE_DIR = os.path.join(BASE_DIR, "models", "roboflow")
os.environ["MODEL_CACHE_DIR"] = ROBOFLOW_CACHE_DIR
os.makedirs(ROBOFLOW_CACHE_DIR, exist_ok=True)

import cv2
import numpy as np
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
from server_module.utils import create_side_by_side_result

DATA_RAW_DIR = os.path.join(BASE_DIR, "data_raw")
OUTPUT_DIR = os.path.join(CURRENT_DIR, "output", "pipeline_rfdetr_small")
os.makedirs(DATA_RAW_DIR, exist_ok=True)
os.makedirs(OUTPUT_DIR, exist_ok=True)

# Cấu hình Model RF-DETR Small
DEFAULT_MODEL_ID = "k-thi-gia-s-workspace/face-spoof-detection-liika-owgrl-1-rfdetr-small-t1"
SHORT_MODEL_ID = "face-spoof-detection-liika-owgrl-1-rfdetr-small-t1"
ROBOFLOW_API_KEY = "ydUs8YBnVWjyjFFVvcpx"

# Nhãn lớp RF-DETR Small
CLASS_NAMES = ["background_class83422", "real", "spoof"]
BACKGROUND_CLASS = "background_class83422"


# =============================================================================
# 1. HELPER FUNCTIONS: FONT, EAR & FILE INDEX MANAGEMENT
# =============================================================================
def remove_vietnamese_accents(text: str) -> str:
    """Chuyển đổi văn bản tiếng Việt có dấu thành không dấu để OpenCV hiển thị không lỗi phông"""
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


def filter_highest_confidence_boxes(detections, iou_thresh=0.25):
    """Lọc các khung nhận diện bị trùng lặp, chỉ giữ khung có confidence cao nhất."""
    if not detections:
        return []

    sorted_dets = sorted(detections, key=lambda d: d.get("confidence", 0.0), reverse=True)
    kept = []

    for d in sorted_dets:
        is_overlapping = False
        for k in kept:
            if calculate_iou(d["bbox"], k["bbox"]) > iou_thresh:
                is_overlapping = True
                break
        if not is_overlapping:
            kept.append(d)

    return kept


def calc_dist(p1, p2):
    """Tính khoảng cách Euclidean giữa 2 điểm (x, y)"""
    return math.hypot(p1[0] - p2[0], p1[1] - p2[1])


def compute_eye_aspect_ratio(landmarks):
    """Tính EAR (Eye Aspect Ratio) từ 468/478 landmarks MediaPipe"""
    if not landmarks or len(landmarks) < 468:
        return 0.0, 0.0, 0.0

    # Mắt trái: 33, 160, 158, 133, 153, 144
    p33 = landmarks[33][:2]
    p160 = landmarks[160][:2]
    p158 = landmarks[158][:2]
    p133 = landmarks[133][:2]
    p153 = landmarks[153][:2]
    p144 = landmarks[144][:2]

    ear_left_v1 = calc_dist(p160, p144)
    ear_left_v2 = calc_dist(p158, p153)
    ear_left_h = calc_dist(p33, p133)
    ear_left = (ear_left_v1 + ear_left_v2) / (2.0 * ear_left_h + 1e-6)

    # Mắt phải: 362, 385, 387, 263, 373, 380
    p362 = landmarks[362][:2]
    p385 = landmarks[385][:2]
    p387 = landmarks[387][:2]
    p263 = landmarks[263][:2]
    p373 = landmarks[373][:2]
    p380 = landmarks[380][:2]

    ear_right_v1 = calc_dist(p385, p380)
    ear_right_v2 = calc_dist(p387, p373)
    ear_right_h = calc_dist(p362, p263)
    ear_right = (ear_right_v1 + ear_right_v2) / (2.0 * ear_right_h + 1e-6)

    ear_avg = (ear_left + ear_right) / 2.0
    return ear_avg, ear_left, ear_right


def get_next_image_index(data_dir: str) -> int:
    """Quét thư mục data_raw/ tìm số thứ tự ảnh lớn nhất dạng <id>.jpg để tự động tăng dần"""
    if not os.path.exists(data_dir):
        return 0

    existing_files = glob.glob(os.path.join(data_dir, "*.jpg"))
    max_idx = -1
    for file_path in existing_files:
        base_name = os.path.splitext(os.path.basename(file_path))[0]
        if base_name.isdigit():
            idx = int(base_name)
            if idx > max_idx:
                max_idx = idx

    return max_idx + 1


def json_serialize_helper(obj):
    """Chuyển đổi các kiểu dữ liệu numpy sang Python native types cho JSON"""
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
# 2. GIAO DIỆN HUD & KHUNG OVAL FACE GUIDE
# =============================================================================
def draw_ui_card(image, x, y, w, h, bg_color=(15, 15, 20), alpha=0.85):
    """Vẽ khung card bán trong suốt làm nền HUD"""
    overlay = image.copy()
    cv2.rectangle(overlay, (x, y), (x + w, y + h), bg_color, -1)
    cv2.addWeighted(overlay, alpha, image, 1 - alpha, 0, image)
    cv2.rectangle(image, (x, y), (x + w, y + h), (100, 100, 100), 1)


def draw_oval_face_guide(image, center, axes, is_aligned=False, is_detected=False, color=(0, 255, 127)):
    """Vẽ khung oval bán nguyệt/elip ngay giữa màn hình với bokeh blur vùng ngoài."""
    h, w = image.shape[:2]
    cx, cy = center
    ax, ay = axes

    mask = np.zeros((h, w), dtype=np.uint8)
    cv2.ellipse(mask, (cx, cy), (ax, ay), 0, 0, 360, 255, -1)
    outside_mask = (mask == 0)

    blurred = cv2.GaussianBlur(image, (35, 35), 0)
    image[outside_mask] = (blurred[outside_mask] * 0.60).astype(np.uint8)

    glow_color = (int(color[0] * 0.35), int(color[1] * 0.35), int(color[2] * 0.35))
    cv2.ellipse(image, (cx, cy), (ax + 3, ay + 3), 0, 0, 360, glow_color, 1, cv2.LINE_AA)
    cv2.ellipse(image, (cx, cy), (max(10, ax - 3), max(10, ay - 3)), 0, 0, 360, glow_color, 1, cv2.LINE_AA)

    thickness = 3 if is_aligned else 2
    cv2.ellipse(image, (cx, cy), (ax, ay), 0, 0, 360, color, thickness, cv2.LINE_AA)

    tick_len = 16
    cv2.line(image, (cx, cy - ay - tick_len), (cx, cy - ay + 6), color, 2, cv2.LINE_AA)
    cv2.line(image, (cx, cy + ay - 6), (cx, cy + ay + tick_len), color, 2, cv2.LINE_AA)
    cv2.line(image, (cx - ax - tick_len, cy), (cx - ax + 6, cy), color, 2, cv2.LINE_AA)
    cv2.line(image, (cx + ax - 6, cy), (cx + ax + tick_len, cy), color, 2, cv2.LINE_AA)

    return image


def is_point_in_oval(pt: Tuple[float, float], center: Tuple[int, int], axes: Tuple[int, int], tolerance: float = 1.0) -> bool:
    """Kiểm tra tọa độ (x, y) có nằm bên trong khung oval hay không."""
    cx, cy = center
    ax, ay = axes
    if ax <= 0 or ay <= 0:
        return False
    norm_x = (float(pt[0]) - cx) / float(ax * tolerance)
    norm_y = (float(pt[1]) - cy) / float(ay * tolerance)
    return (norm_x ** 2 + norm_y ** 2) <= 1.0


def is_face_in_oval(bbox: Union[List[int], Tuple[int, ...]], center: Tuple[int, int], axes: Tuple[int, int], tolerance: float = 1.08) -> bool:
    """Kiểm tra xem bounding box của khuôn mặt có nằm trong khung oval hay không."""
    x1, y1, x2, y2 = bbox
    face_cx = (x1 + x2) / 2.0
    face_cy = (y1 + y2) / 2.0
    return is_point_in_oval((face_cx, face_cy), center, axes, tolerance=tolerance)


def get_oval_masked_frame(frame: np.ndarray, center: Tuple[int, int], axes: Tuple[int, int], blur_ksize: int = 45, dim_factor: float = 0.35) -> np.ndarray:
    """Tạo bản sao frame với vùng bên ngoài khung oval bị làm mờ mạnh."""
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


def create_rfdetr_small_dashboard(
    img_idx: Any,
    face_info: Optional[Dict[str, Any]] = None,
    num_faces: int = 1,
    pose_info: Optional[Dict[str, Any]] = None,
    pose_valid: bool = True,
    anti_spoof_info: Optional[Dict[str, Any]] = None,
    spoof_iou: float = 0.0,
    blink_passed: bool = True,
    blink_count: int = 0,
    head_movement_passed: bool = True,
    head_action_name: str = "NONE",
    final_pass: bool = True,
    reasons: Optional[List[str]] = None,
    face_crop: Optional[np.ndarray] = None,
    target_height: Optional[int] = None,
    model_id: str = DEFAULT_MODEL_ID,
    width: int = 560
) -> np.ndarray:
    """Tạo bảng Dashboard độc lập chuyên nghiệp cho RF-DETR Small Anti-Spoofing Pipeline."""
    clean_reasons = [remove_vietnamese_accents(r) for r in reasons] if (not final_pass and reasons) else []
    num_reasons = len(clean_reasons)
    extra_h = max(0, num_reasons * 24)

    min_h = 500 + extra_h
    h = max(min_h, target_height) if target_height else min_h
    w = max(500, width)

    canvas = np.full((h, w, 3), (20, 22, 28), dtype=np.uint8)

    # 1. Header Card
    hdr_h = 70
    cv2.rectangle(canvas, (10, 10), (w - 10, hdr_h), (32, 36, 48), -1)
    cv2.rectangle(canvas, (10, 10), (w - 10, hdr_h), (60, 70, 90), 1)

    cv2.putText(canvas, "E-KYC DASHBOARD (RF-DETR SMALL ENGINE)", (24, 38),
                cv2.FONT_HERSHEY_SIMPLEX, 0.58, (0, 230, 255), 2, cv2.LINE_AA)
    model_short = model_id.split("/")[-1] if "/" in model_id else model_id
    session_str = f"Session ID: {img_idx} | Arch: Detection Transformer (~109MB)"
    cv2.putText(canvas, session_str, (24, 58),
                cv2.FONT_HERSHEY_SIMPLEX, 0.40, (170, 180, 195), 1, cv2.LINE_AA)

    if face_crop is not None and face_crop.size > 0:
        try:
            th_size = 50
            thumb = cv2.resize(face_crop, (th_size, th_size))
            tx1 = w - 10 - th_size - 8
            ty1 = 12
            cv2.rectangle(canvas, (tx1 - 2, ty1 - 2), (tx1 + th_size + 2, ty1 + th_size + 2), (0, 230, 255), 1)
            canvas[ty1:ty1 + th_size, tx1:tx1 + th_size] = thumb
        except Exception:
            pass

    cur_y = hdr_h + 12

    def _draw_card(title: str, lines: List[Tuple[str, Tuple[int, int, int], float]], card_h: int):
        nonlocal cur_y
        cv2.rectangle(canvas, (10, cur_y), (w - 10, cur_y + card_h), (27, 30, 40), -1)
        cv2.rectangle(canvas, (10, cur_y), (w - 10, cur_y + card_h), (50, 58, 75), 1)
        cv2.rectangle(canvas, (10, cur_y), (14, cur_y + card_h), (0, 200, 240), -1)

        cv2.putText(canvas, title, (24, cur_y + 20),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.46, (220, 225, 235), 1, cv2.LINE_AA)

        line_y = cur_y + 40
        for text, col, font_scale in lines:
            cv2.putText(canvas, text, (24, line_y),
                        cv2.FONT_HERSHEY_SIMPLEX, font_scale, col, 1, cv2.LINE_AA)
            line_y += 20
        cur_y += card_h + 8

    # Section 1: Face Detection
    if face_info:
        conf = face_info.get("confidence", 0.0)
        if num_faces == 1:
            f_lines = [
                (f"Status: PASS  |  Faces: 1 detected  |  Confidence: {conf*100:.1f}%", (80, 220, 80), 0.42)
            ]
        else:
            f_lines = [
                (f"Status: WARNING | MULTI-FACE ({num_faces} faces detected)", (0, 165, 255), 0.42)
            ]
    else:
        f_lines = [("Status: FAIL  |  NO FACE DETECTED", (70, 70, 240), 0.42)]
    _draw_card("1. FACE DETECTION", f_lines, card_h=52)

    # Section 2: Head Pose 3D
    if pose_info:
        yaw = pose_info.get("yaw", 0.0)
        pitch = pose_info.get("pitch", 0.0)
        roll = pose_info.get("roll", 0.0)
        p_stat = "PASS (Chuan huong thang)" if pose_valid else "FAIL (Goc quay vuot nguong)"
        p_col = (80, 220, 80) if pose_valid else (70, 70, 240)
        p_lines = [
            (f"Angles: Yaw: {yaw:+.1f} deg  |  Pitch: {pitch:+.1f} deg  |  Roll: {roll:+.1f} deg", (200, 210, 220), 0.41),
            (f"Status: {p_stat}", p_col, 0.42)
        ]
    else:
        p_lines = [("Status: UNKNOWN (Khong duoc tinh toan)", (70, 70, 240), 0.42)]
    _draw_card("2. 3D HEAD POSE ESTIMATION", p_lines, card_h=68)

    # Section 3: Anti-Spoofing (RF-DETR Small Engine)
    if anti_spoof_info:
        as_lbl = anti_spoof_info.get("label", "UNKNOWN")
        as_conf = anti_spoof_info.get("confidence", 0.0)
        is_real = anti_spoof_info.get("is_real", False)
        as_col = (80, 220, 80) if is_real else (70, 70, 240)
        iou_str = f"  |  IoU with Face: {spoof_iou:.2f}" if spoof_iou > 0 else ""
        as_lines = [
            (f"Model Verdict: {as_lbl} ({as_conf*100:.1f}%){iou_str}", as_col, 0.44),
            (f"Classification: {'REAL FACE (Hop le)' if is_real else 'FAKE / SPOOF ATTACK (Phat hien gia mao)'}", as_col, 0.41),
            (f"Engine: RF-DETR Small ({model_short})", (160, 180, 200), 0.38)
        ]

        card_h = 96
        cv2.rectangle(canvas, (10, cur_y), (w - 10, cur_y + card_h), (27, 30, 40), -1)
        cv2.rectangle(canvas, (10, cur_y), (w - 10, cur_y + card_h), (50, 58, 75), 1)
        cv2.rectangle(canvas, (10, cur_y), (14, cur_y + card_h), (0, 200, 240), -1)

        cv2.putText(canvas, "3. ANTI-SPOOFING (RF-DETR SMALL ENGINE)", (24, cur_y + 20),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.46, (220, 225, 235), 1, cv2.LINE_AA)

        line_y = cur_y + 38
        for text, col, font_scale in as_lines:
            cv2.putText(canvas, text, (24, line_y),
                        cv2.FONT_HERSHEY_SIMPLEX, font_scale, col, 1, cv2.LINE_AA)
            line_y += 18

        # Thanh tỉ lệ Real vs Spoof
        bar_w = w - 60
        bar_h = 8
        bar_x = 24
        bar_y = cur_y + 82
        cv2.rectangle(canvas, (bar_x, bar_y), (bar_x + bar_w, bar_y + bar_h), (45, 48, 60), -1)
        real_fill = int(bar_w * (as_conf if is_real else (1.0 - as_conf)))
        if real_fill > 0:
            cv2.rectangle(canvas, (bar_x, bar_y), (bar_x + real_fill, bar_y + bar_h), (80, 220, 80), -1)
        if bar_w - real_fill > 0:
            cv2.rectangle(canvas, (bar_x + real_fill, bar_y), (bar_x + bar_w, bar_y + bar_h), (70, 70, 240), -1)

        cur_y += card_h + 8
    else:
        as_lines = [("Status: NO ANTI-SPOOF DATA", (0, 180, 255), 0.42)]
        _draw_card("3. ANTI-SPOOFING (RF-DETR SMALL ENGINE)", as_lines, card_h=52)

    # Section 4: Active Liveness (Blink & Head Action)
    b_stat = f"PASS ({blink_count} blinks)" if blink_passed else f"FAIL ({blink_count} blinks)"
    b_col = (80, 220, 80) if blink_passed else (70, 70, 240)
    h_act = str(head_action_name).upper()
    h_stat = f"PASS [{h_act}]" if head_movement_passed else f"FAIL [{h_act}]"
    h_col = (80, 220, 80) if head_movement_passed else (70, 70, 240)
    l_lines = [
        (f"Eye Blink Liveness      : {b_stat}", b_col, 0.42),
        (f"Head Movement Liveness  : {h_stat}", h_col, 0.42)
    ]
    _draw_card("4. ACTIVE LIVENESS VALIDATION", l_lines, card_h=68)

    # Section 5: Final Decision Card
    dec_h = 60 + extra_h
    dec_bg = (24, 38, 24) if final_pass else (38, 24, 24)
    dec_border = (80, 220, 80) if final_pass else (70, 70, 240)
    cv2.rectangle(canvas, (10, cur_y), (w - 10, cur_y + dec_h), dec_bg, -1)
    cv2.rectangle(canvas, (10, cur_y), (w - 10, cur_y + dec_h), dec_border, 2)

    verdict_text = "FINAL VERDICT: APPROVED (HOP LE)" if final_pass else "FINAL VERDICT: REJECTED (TU CHOI)"
    cv2.putText(canvas, verdict_text, (24, cur_y + 28),
                cv2.FONT_HERSHEY_SIMPLEX, 0.60, dec_border, 2, cv2.LINE_AA)

    if not final_pass and clean_reasons:
        cv2.putText(canvas, "Ly do tu choi / Reject Reasons:", (24, cur_y + 48),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.41, (0, 210, 255), 1, cv2.LINE_AA)
        r_start_y = cur_y + 68
        for idx_r, r_t in enumerate(clean_reasons[:5]):
            if len(r_t) > 62:
                r_t = r_t[:59] + "..."
            cv2.putText(canvas, f"  * {r_t}", (24, r_start_y + idx_r * 22),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.39, (160, 215, 255), 1, cv2.LINE_AA)

    cv2.putText(canvas, "Press [r]: Tiep tuc chup anh tiep theo | [q]: Thoat", (24, h - 12),
                cv2.FONT_HERSHEY_SIMPLEX, 0.35, (100, 110, 130), 1, cv2.LINE_AA)

    return canvas


# =============================================================================
# 3. PIPELINE RF-DETR SMALL WORKFLOW STATE MACHINE
# =============================================================================
class PipelineStage(Enum):
    PREVIEW_ALIGN = 1       # Giai đoạn 1: Mở webcam, canh góc mặt & chờ chụp ảnh trong oval
    RUN_AI_STATIC = 2       # Giai đoạn 2: Chạy Face -> Landmark -> Pose -> Crop 224 -> RF-DETR Small Anti-Spoof
    LIVE_BLINK = 3          # Giai đoạn 3: Active Liveness - Thử thách chớp mắt
    LIVE_HEAD_MOVEMENT = 4  # Giai đoạn 4: Active Liveness - Thử thách quay đầu
    FINAL_DECISION = 5      # Giai đoạn 5: Tổng hợp toàn bộ & lưu vào output/pipeline_rfdetr_small/<id>/


def parse_rfdetr_predictions(response, img_w: int, img_h: int):
    """Trích xuất danh sách bounding box từ kết quả dự đoán RF-DETR Small (lọc bỏ background)."""
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

        if "background" in cls_name:
            continue

        if 0.0 <= cx <= 1.0 and 0.0 <= pw <= 1.0 and img_w > 1:
            cx *= img_w
            cy *= img_h
            pw *= img_w
            ph *= img_h

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
            "label": "REAL" if is_real else "SPOOF",
            "raw_class": cls_name
        })

    return dets


def main_pipeline_rfdetr_small(cam_id=0, skip_liveness=False, model_id=DEFAULT_MODEL_ID):
    # Kiểm tra kích thước weights
    candidate_weights = [
        os.path.join(ROBOFLOW_CACHE_DIR, *model_id.split("/"), "weights.onnx"),
        os.path.join(ROBOFLOW_CACHE_DIR, SHORT_MODEL_ID, "weights.onnx"),
    ]
    weights_path = next((p for p in candidate_weights if os.path.exists(p)), None)
    weights_mb = os.path.getsize(weights_path) / (1024 * 1024) if weights_path else 109.0

    print("\n" + "=" * 80)
    print("   FULL E-KYC PIPELINE RF-DETR SMALL (DETECTION TRANSFORMER ANTI-SPOOFING)")
    print("=" * 80)
    print(f"  * Thư mục lưu ảnh gốc : {DATA_RAW_DIR}")
    print(f"  * Thư mục lưu kết quả : {OUTPUT_DIR}")
    print(f"  * Model RF-DETR ID    : {model_id}")
    print(f"  * Kiến trúc           : RF-DETR Small (Detection Transformer)")
    print(f"  * Weights             : {weights_mb:.1f} MB (weights.onnx)")
    print(f"  * Nhãn                : real, spoof (background_class83422 tự động lọc bỏ)")
    print("  * Nền tảng thực thi   : inference.get_model (Local Offline Cache 100%)")
    print("  * Điều khiển:")
    print("      [SPACE] hoặc [c]  : Chụp ảnh và chạy Full Quy trình (AI + Live Liveness)")
    print("      [s]               : CHỤP NHANH & LƯU NGAY (Chạy AI Model -> Lưu kết quả ngay)")
    print("      [a]               : Bật/Tắt chế độ tự động chụp khi mặt chuẩn trong oval")
    print("      [r]               : Khởi tạo lại phiên eKYC mới")
    print("      [q] hoặc [ESC]    : Thoát")
    print("=" * 80 + "\n")

    os.makedirs(DATA_RAW_DIR, exist_ok=True)
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    # 1. Khởi tạo Models
    print("[INFO] Đang khởi tạo các AI Models...")
    detector = FaceDetector()
    landmark_detector = LandmarkDetector()
    pose_validator = PoseValidator()
    aligner = FaceAligner()
    head_movement_detector = HeadMovementDetector(yaw_threshold=16.0, pitch_threshold=12.0, timeout=7.0)

    print(f"[INFO] Đang nạp mô hình RF-DETR Small ({model_id}) vào RAM...")
    print(f"[INFO] File weights ~{weights_mb:.0f}MB, quá trình nạp có thể mất vài giây...")
    try:
        roboflow_model = get_model(model_id=model_id, api_key=ROBOFLOW_API_KEY)
        print("[OK] Đã nạp thành công với Full Model ID!")
    except Exception as e:
        print(f"[WARN] Thử nạp với Short ID '{SHORT_MODEL_ID}'... ({e})")
        roboflow_model = get_model(model_id=SHORT_MODEL_ID, api_key=ROBOFLOW_API_KEY)
        print("[OK] Đã nạp thành công với Short Model ID!")

    print("[OK] Đã khởi tạo hoàn tất toàn bộ Models!\n")

    cap = cv2.VideoCapture(cam_id)
    if not cap.isOpened():
        print(f"[ERROR] Không thể mở Camera ID {cam_id}!")
        return

    # Trạng thái luồng
    stage = PipelineStage.PREVIEW_ALIGN
    auto_capture_mode = False
    quick_snapshot_mode = False
    consecutive_center_frames = 0
    is_aligned_good = False
    capture_blocked_frames = 0
    is_light_ok = True
    mean_lum = 100.0

    # Dữ liệu phiên hiện tại
    current_img_idx = get_next_image_index(DATA_RAW_DIR)
    captured_frame = None
    captured_img_path = None
    captured_result_dir = None

    # Dữ liệu tĩnh từ ảnh chụp
    primary_face = None
    faces = []
    num_faces = 0
    all_face_crops_info = []
    landmarks_static = None
    pose_dict_static = None
    pose_valid_static = False
    face_crop_static = None
    aligned_img_static = None
    best_spoof_static = None
    spoof_res = []
    primary_spoof_iou = 0.0
    has_any_spoof_in_frame = False

    # Dữ liệu Active Liveness
    blink_passed = False
    blink_counter = 0
    blink_state = False

    head_movement_passed = False
    current_head_action = HeadAction.NONE
    head_action_prompt = ""

    # Dữ liệu Final Decision
    final_pass = False
    reasons = []
    final_record = None
    final_display_img = None

    prev_fps_time = time.time()

    def start_new_session():
        nonlocal stage, auto_capture_mode, quick_snapshot_mode, consecutive_center_frames
        nonlocal is_aligned_good, capture_blocked_frames, current_img_idx, captured_frame
        nonlocal primary_face, faces, num_faces, all_face_crops_info, landmarks_static
        nonlocal pose_dict_static, pose_valid_static, face_crop_static, aligned_img_static
        nonlocal best_spoof_static, spoof_res, primary_spoof_iou, has_any_spoof_in_frame
        nonlocal blink_passed, blink_counter, blink_state, head_movement_passed
        nonlocal current_head_action, head_action_prompt, final_pass, reasons, final_record, final_display_img

        stage = PipelineStage.PREVIEW_ALIGN
        auto_capture_mode = False
        quick_snapshot_mode = False
        consecutive_center_frames = 0
        is_aligned_good = False
        capture_blocked_frames = 0

        current_img_idx = get_next_image_index(DATA_RAW_DIR)
        captured_frame = None

        primary_face = None
        faces = []
        num_faces = 0
        all_face_crops_info = []
        landmarks_static = None
        pose_dict_static = None
        pose_valid_static = False
        face_crop_static = None
        aligned_img_static = None
        best_spoof_static = None
        spoof_res = []
        primary_spoof_iou = 0.0
        has_any_spoof_in_frame = False

        blink_passed = False
        blink_counter = 0
        blink_state = False

        head_movement_passed = False
        head_movement_detector.reset()
        current_head_action = HeadAction.NONE
        head_action_prompt = ""

        final_pass = False
        reasons = []
        final_record = None
        final_display_img = None

        print(f"\n[INFO] Đã khởi tạo phiên mới! Chuẩn bị chụp ảnh ID: {current_img_idx}.jpg")

    while True:
        ret, frame = cap.read()
        if not ret:
            print("[WARN] Mất tín hiệu Webcam.")
            break

        frame = cv2.flip(frame, 1)
        h, w = frame.shape[:2]
        display = frame.copy()
        key_trigger = None

        # Khung oval trung tâm
        oval_cx = int(w * 0.50)
        oval_cy = int(h * 0.50)
        oval_ax = int(w * 0.22)
        oval_ay = int(h * 0.38)
        oval_center = (oval_cx, oval_cy)
        oval_axes = (oval_ax, oval_ay)

        # =====================================================================
        # GIAI ĐOẠN 1: PREVIEW & CĂN CHỈNH KHUÔN MẶT TRONG KHUNG OVAL
        # =====================================================================
        if stage == PipelineStage.PREVIEW_ALIGN:
            masked_preview = get_oval_masked_frame(frame, oval_center, oval_axes)
            landmarks_live = landmark_detector.detect(masked_preview)

            is_in_oval = False
            pose_valid_live = False
            pose_dict_live = None

            if landmarks_live and len(landmarks_live) >= 468:
                xs = [p[0] for p in landmarks_live]
                ys = [p[1] for p in landmarks_live]
                face_center = ((min(xs) + max(xs)) / 2.0, (min(ys) + max(ys)) / 2.0)
                is_in_oval = is_point_in_oval(face_center, oval_center, oval_axes, tolerance=1.12)

                if is_in_oval:
                    pose_valid_live, _, pose_dict_live = pose_validator.validate(landmarks_live, get_landmark_point)

            # Kiểm tra ánh sáng
            lum_bbox = None
            if landmarks_live and is_in_oval:
                xs = [p[0] for p in landmarks_live]
                ys = [p[1] for p in landmarks_live]
                lum_bbox = [int(min(xs)), int(min(ys)), int(max(xs)), int(max(ys))]
            light_metrics = check_illumination_quality(frame, bbox=lum_bbox)
            mean_lum = light_metrics["mean_luminance"]
            is_light_ok = (mean_lum >= 60.0)

            is_aligned_good = (landmarks_live is not None and is_in_oval and pose_valid_live and is_light_ok)

            if is_aligned_good:
                consecutive_center_frames += 1
                guide_color = (0, 255, 127)  # Xanh lục sáng
                align_msg = "MAT CHUAN TRONG OVAL! SAN SANG CHUP"
                align_col = (0, 255, 127)
            else:
                consecutive_center_frames = 0
                guide_color = (0, 140, 255)  # Cam
                if not is_light_ok and landmarks_live is not None:
                    align_msg = f"THIEU SANG (Luminance={mean_lum:.1f} < 60)! VUI LONG BAT DEN"
                    align_col = (0, 100, 255)
                elif not pose_valid_live:
                    align_msg = "VUI LONG NHIN THANG VAO CAMERA"
                    align_col = (70, 70, 240)
                else:
                    align_msg = "CANH CHINH MAT VAO KHUNG OVAL"
                    align_col = (0, 165, 255)

            display = draw_oval_face_guide(display, oval_center, oval_axes,
                                           is_aligned=is_aligned_good,
                                           is_detected=(landmarks_live is not None),
                                           color=guide_color)

            # Header Top Card
            top_card_h = 75
            draw_ui_card(display, 20, 15, w - 40, top_card_h, bg_color=(15, 15, 25), alpha=0.88)
            cv2.putText(display, f"BUOC 1: CAN CHINH MAT VAO OVAL - ID TIEP THEO: {current_img_idx}.jpg",
                        (35, 42), cv2.FONT_HERSHEY_SIMPLEX, 0.58, (0, 230, 255), 2, cv2.LINE_AA)
            cv2.putText(display, f"Huong dan: {align_msg}",
                        (35, 70), cv2.FONT_HERSHEY_SIMPLEX, 0.48, align_col, 1, cv2.LINE_AA)

            if not is_light_ok and landmarks_live is not None:
                light_card_w = min(420, w - 40)
                draw_ui_card(display, (w - light_card_w) // 2, top_card_h + 25, light_card_w, 40, bg_color=(10, 20, 40), alpha=0.92)
                cv2.rectangle(display, ((w - light_card_w) // 2, top_card_h + 25),
                              ((w - light_card_w) // 2 + light_card_w, top_card_h + 65), (0, 140, 255), 2)
                cv2.putText(display, f"! CANH BAO: KHUON MAT BI TOI (Luminance: {mean_lum:.1f}/60) !",
                            ((w - light_card_w) // 2 + 15, top_card_h + 50),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.42, (0, 140, 255), 1, cv2.LINE_AA)

            if capture_blocked_frames > 0:
                capture_blocked_frames -= 1
                warn_w = min(540, w - 40)
                draw_ui_card(display, (w - warn_w) // 2, h // 2 - 30, warn_w, 60, bg_color=(10, 10, 40), alpha=0.92)
                cv2.rectangle(display, ((w - warn_w) // 2, h // 2 - 30),
                              ((w - warn_w) // 2 + warn_w, h // 2 + 30), (0, 0, 255), 2)
                if not is_light_ok:
                    cv2.putText(display, "KHOA CHUP: ANH SANG QUA YEU (L < 60)!",
                                ((w - warn_w) // 2 + 20, h // 2 - 5),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.50, (0, 0, 255), 2, cv2.LINE_AA)
                    cv2.putText(display, "Vui long bat den hoac di chuyen ra noi sang hon!",
                                ((w - warn_w) // 2 + 20, h // 2 + 18),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.40, (0, 180, 255), 1, cv2.LINE_AA)
                else:
                    cv2.putText(display, "KHOA CHUP: MAT CHUA DUNG TRONG KHUNG OVAL!",
                                ((w - warn_w) // 2 + 20, h // 2 - 5),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.50, (0, 0, 255), 2, cv2.LINE_AA)
                    cv2.putText(display, "Vui long nhin thang va dua mat vao giua oval de chup",
                                ((w - warn_w) // 2 + 20, h // 2 + 18),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.40, (0, 180, 255), 1, cv2.LINE_AA)

            bot_card_h = 58
            bot_y = h - bot_card_h - 15
            draw_ui_card(display, 20, bot_y, w - 40, bot_card_h, bg_color=(15, 15, 20), alpha=0.88)

            auto_tag = "[BAT - DANG QUET]" if auto_capture_mode else "[TAT]"
            auto_col = (0, 255, 127) if auto_capture_mode else (170, 170, 170)
            cv2.putText(display, f"Che do Auto-Capture: {auto_tag} (Bam [a] de bat/tat)",
                        (35, bot_y + 24), cv2.FONT_HERSHEY_SIMPLEX, 0.45, auto_col, 1, cv2.LINE_AA)

            if is_aligned_good:
                shortcut_hint = "Bam [SPACE] de Chup Full eKYC  |  [s]: Chup Nhanh  |  [q]: Thoat"
                shortcut_col = (0, 255, 127)
            else:
                shortcut_hint = "Canh mat vao Oval de Mo Khoa Chup | [a]: Auto-Capture | [q]: Thoat"
                shortcut_col = (140, 140, 150)

            cv2.putText(display, shortcut_hint,
                        (28, bot_y + 44), cv2.FONT_HERSHEY_SIMPLEX, 0.38, shortcut_col, 1, cv2.LINE_AA)

            trigger_capture = (auto_capture_mode and consecutive_center_frames >= 25 and is_aligned_good)
            key_trigger = ord(' ') if trigger_capture else None

        # =====================================================================
        # GIAI ĐOẠN 2: CHẠY AI MODEL TRÊN ẢNH CHỤP ĐẾN BƯỚC ANTI-SPOOF
        # =====================================================================
        elif stage == PipelineStage.RUN_AI_STATIC:
            draw_ui_card(display, 20, 20, w - 40, 90, bg_color=(15, 15, 25), alpha=0.9)
            cv2.putText(display, f"DANG CHAY RF-DETR SMALL TREN ANH ID {current_img_idx}.jpg...", (35, 55),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.60, (0, 230, 255), 2)
            cv2.putText(display, "Tien trinh: Face Detect -> Landmark -> Pose 3D -> Crop 224 -> RF-DETR Small Anti-Spoof", (35, 85),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.42, (0, 255, 0), 1)
            cv2.imshow("Full E-KYC Pipeline (RF-DETR Small Anti-Spoof)", display)
            cv2.waitKey(1)

            captured_img_path = os.path.join(DATA_RAW_DIR, f"{current_img_idx}.jpg")
            cv2.imwrite(captured_img_path, captured_frame)
            print(f"\n[1. CHỤP ẢNH GỐC] Đã lưu ảnh vào: {captured_img_path}")

            captured_result_dir = os.path.join(OUTPUT_DIR, str(current_img_idx))
            os.makedirs(captured_result_dir, exist_ok=True)
            all_faces_dir = os.path.join(captured_result_dir, "all_faces_cropped")
            os.makedirs(all_faces_dir, exist_ok=True)

            raw_faces = detector.detect(captured_frame)
            faces = [f for f in raw_faces if is_face_in_oval(f["bbox"], oval_center, oval_axes)]
            ignored_faces = [f for f in raw_faces if not is_face_in_oval(f["bbox"], oval_center, oval_axes)]
            num_faces = len(faces)
            print(f"[2. Face Detection] Tổng phát hiện: {len(raw_faces)} khuôn mặt.")
            print(f"  -> Trong khung oval (xác thực): {num_faces} mặt.")

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

            captured_masked = get_oval_masked_frame(captured_frame, oval_center, oval_axes)
            landmarks_static = landmark_detector.detect(captured_masked)
            if landmarks_static and len(landmarks_static) >= 468:
                xs = [p[0] for p in landmarks_static]
                ys = [p[1] for p in landmarks_static]
                if not is_point_in_oval(((min(xs) + max(xs)) / 2.0, (min(ys) + max(ys)) / 2.0), oval_center, oval_axes, tolerance=1.15):
                    landmarks_static = None
            print(f"[3. Landmarks] Trích xuất được {len(landmarks_static) if landmarks_static else 0} điểm.")

            pose_valid_static = False
            pose_dict_static = None
            if landmarks_static:
                pose_valid_static, _, pose_dict_static = pose_validator.validate(landmarks_static, get_landmark_point)
                if pose_dict_static:
                    print(f"[4. Head Pose 3D] Y={pose_dict_static['yaw']:+.1f}° | P={pose_dict_static['pitch']:+.1f}° | R={pose_dict_static['roll']:+.1f}° -> {'PASS' if pose_valid_static else 'FAIL'}")

            aligned_img_static = None
            face_crop_static = None

            if primary_face is not None:
                face_crop_static = aligner.crop_face(
                    captured_frame,
                    bbox=primary_face["bbox"],
                    padding=25,
                    output_size=(224, 224),
                    mode="bbox"
                )
                print(f"[5. Face Crop] Cắt ảnh chuẩn thẳng đứng tự nhiên từ YOLO BBox (224x224).")
            elif landmarks_static:
                face_crop_static = aligner.crop_face(
                    captured_frame,
                    landmarks=landmarks_static,
                    padding=25,
                    output_size=(224, 224)
                )

            if landmarks_static:
                aligned_img_static = aligner.align_face(captured_frame, landmarks_static)

            if aligned_img_static is None:
                aligned_img_static = captured_frame.copy()

            if face_crop_static is not None:
                cv2.imwrite(os.path.join(captured_result_dir, "2_face_crop_224.jpg"), face_crop_static)

            if aligned_img_static is not None:
                cv2.imwrite(os.path.join(captured_result_dir, "3_aligned_full.jpg"), aligned_img_static)

            # 7. RF-DETR Small Anti-Spoofing Inference
            captured_light = check_illumination_quality(captured_frame, bbox=primary_face["bbox"] if primary_face else None)
            if captured_light["mean_luminance"] < 80.0:
                input_spoof = enhance_low_light(captured_frame)
                print(f"[Anti-Spoof Preprocess] Độ sáng L={captured_light['mean_luminance']:.1f}. Áp dụng CLAHE tăng cường vi vân da.")
            else:
                input_spoof = captured_frame

            print(f"[6. Anti-Spoofing RF-DETR Small] Đang chạy suy luận mô hình {model_id}...")
            t0 = time.time()
            rf_res = roboflow_model.infer(image=input_spoof)
            rf_time = (time.time() - t0) * 1000.0
            print(f"[OK] RF-DETR Small hoàn thành trong {rf_time:.1f}ms (Offline 100%)")

            spoof_res = parse_rfdetr_predictions(rf_res, w_f, h_f)

            has_any_spoof_in_frame = any(not d["is_real"] for d in spoof_res)
            best_spoof_static = None
            primary_spoof_iou = 0.0

            if primary_face is not None and spoof_res:
                p_box = primary_face["bbox"]
                best_iou = -1.0
                best_d = None
                for d in spoof_res:
                    iou = calculate_iou(p_box, d["bbox"])
                    if iou > best_iou:
                        best_iou = iou
                        best_d = d
                if best_iou > 0.10:
                    best_spoof_static = best_d
                    primary_spoof_iou = best_iou
                else:
                    best_spoof_static = max(spoof_res, key=lambda x: x["confidence"])
            elif spoof_res:
                best_spoof_static = max(spoof_res, key=lambda x: x["confidence"])

            if best_spoof_static:
                stat_str = "REAL (THẬT)" if best_spoof_static["is_real"] else "SPOOF (GIẢ MẠO)"
                print(f"  -> Kết quả RF-DETR: {stat_str} | Độ tin cậy: {best_spoof_static['confidence']*100:.2f}% | IoU: {primary_spoof_iou:.2f}")

            # Điều hướng giai đoạn tiếp theo
            if skip_liveness or quick_snapshot_mode:
                print("\n[CHẾ ĐỘ STATIC] Bỏ qua Active Liveness. Đưa ra phán quyết ngay lập tức!")
                blink_passed = True
                head_movement_passed = True
                stage = PipelineStage.FINAL_DECISION
            else:
                stage = PipelineStage.LIVE_BLINK
                print("\n[BƯỚC TIẾP THEO] Bắt đầu Active Liveness: Thử thách chớp mắt...")

        # =====================================================================
        # GIAI ĐOẠN 3: ACTIVE LIVENESS - THỬ THÁCH CHỚP MẮT (BLINK)
        # =====================================================================
        elif stage == PipelineStage.LIVE_BLINK:
            masked_blink = get_oval_masked_frame(frame, oval_center, oval_axes)
            landmarks_blink = landmark_detector.detect(masked_blink)

            ear_avg = 0.0
            if landmarks_blink and len(landmarks_blink) >= 468:
                xs = [p[0] for p in landmarks_blink]
                ys = [p[1] for p in landmarks_blink]
                if is_point_in_oval(((min(xs) + max(xs)) / 2.0, (min(ys) + max(ys)) / 2.0), oval_center, oval_axes, tolerance=1.15):
                    ear_avg, _, _ = compute_eye_aspect_ratio(landmarks_blink)

            EAR_THRESH = 0.20
            if ear_avg > 0.0:
                if ear_avg < EAR_THRESH:
                    if not blink_state:
                        blink_state = True
                else:
                    if blink_state:
                        blink_counter += 1
                        blink_state = False
                        print(f"[LIVE BLINK] Đã ghi nhận {blink_counter} lần chớp mắt!")

            if blink_counter >= 1:
                blink_passed = True
                stage = PipelineStage.LIVE_HEAD_MOVEMENT
                current_head_action = head_movement_detector.start_challenge()
                head_action_prompt = head_movement_detector.get_prompt()
                print(f"\n[LIVE BLINK] Hoàn thành chớp mắt ({blink_counter} lần)!")
                print(f"[LIVE HEAD] Thử thách quay đầu: {head_action_prompt} ({current_head_action.value})")

            draw_oval_face_guide(display, oval_center, oval_axes, is_aligned=True, is_detected=True, color=(0, 230, 255))
            draw_ui_card(display, 20, 20, w - 40, 95, bg_color=(15, 15, 25), alpha=0.9)
            cv2.putText(display, "THU THACH 1: VUI LONG CHOP MAT (BLINK DETECTION)", (35, 50),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.60, (0, 230, 255), 2)
            cv2.putText(display, f"So lan chop mat: {blink_counter}/1  |  EAR: {ear_avg:.2f} (Nguong: < {EAR_THRESH})", (35, 80),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.46, (0, 255, 127) if blink_counter >= 1 else (200, 200, 200), 1)
            cv2.putText(display, "[s]: Bo qua va Luu ngay  |  [q]: Thoat", (35, 102),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.38, (150, 150, 150), 1)

        # =====================================================================
        # GIAI ĐOẠN 4: ACTIVE LIVENESS - THỬ THÁCH QUAY ĐẦU (HEAD MOVEMENT)
        # =====================================================================
        elif stage == PipelineStage.LIVE_HEAD_MOVEMENT:
            masked_head = get_oval_masked_frame(frame, oval_center, oval_axes)
            landmarks_head = landmark_detector.detect(masked_head)

            h_status = None
            if landmarks_head and len(landmarks_head) >= 468:
                xs = [p[0] for p in landmarks_head]
                ys = [p[1] for p in landmarks_head]
                if is_point_in_oval(((min(xs) + max(xs)) / 2.0, (min(ys) + max(ys)) / 2.0), oval_center, oval_axes, tolerance=1.15):
                    _, _, pose_dict_live = pose_validator.validate(landmarks_head, get_landmark_point)
                    if pose_dict_live:
                        h_status = head_movement_detector.update(pose_dict_live)

            draw_oval_face_guide(display, oval_center, oval_axes, is_aligned=True, is_detected=True, color=(0, 230, 255))
            draw_ui_card(display, 20, 20, w - 40, 105, bg_color=(15, 15, 25), alpha=0.9)

            prompt_no_accent = remove_vietnamese_accents(head_action_prompt)
            cv2.putText(display, f"THU THACH 2: {prompt_no_accent.upper()}", (35, 50),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.60, (0, 230, 255), 2)

            prog = h_status.get("progress", 0.0) if h_status else 0.0
            t_left = h_status.get("time_left", 0.0) if h_status else 0.0
            cv2.putText(display, f"Tien trinh: {prog*100:.0f}%  |  Thoi gian con lai: {t_left:.1f}s", (35, 80),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.48, (0, 255, 127) if prog >= 1.0 else (200, 200, 200), 1)
            cv2.putText(display, "[s]: Bo qua va Luu ngay  |  [q]: Thoat", (35, 106),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.38, (150, 150, 150), 1)

            if h_status:
                if h_status.get("passed", False):
                    head_movement_passed = True
                    stage = PipelineStage.FINAL_DECISION
                    print(f"[LIVE HEAD] Thử thách quay đầu '{current_head_action.value}' HOÀN TẤT THÀNH CÔNG!")
                elif str(h_status.get("state", "")).upper() == "FAILED":
                    head_movement_passed = False
                    stage = PipelineStage.FINAL_DECISION
                    print(f"[LIVE HEAD] Hết thời gian thử thách quay đầu!")

        # =====================================================================
        # GIAI ĐOẠN 5: TỔNG HỢP VÀ RA QUYẾT ĐỊNH CUỐI CÙNG (FINAL DECISION)
        # =====================================================================
        elif stage == PipelineStage.FINAL_DECISION:
            if final_display_img is None:
                reasons = []

                if num_faces == 0:
                    reasons.append("Khong phat hien khuon mat hop le trong oval")
                elif num_faces > 1:
                    reasons.append(f"Phat hien nhieu khuon mat ({num_faces} mat trong oval)")

                if not pose_valid_static:
                    reasons.append("Tu the khuon mat khong thang hoac vuot nguong")

                if best_spoof_static is None:
                    reasons.append("Khong co du lieu Anti-Spoof tu mo hinh RF-DETR Small")
                else:
                    if not best_spoof_static["is_real"]:
                        reasons.append(f"Phat hien gia mao boi RF-DETR Small ({best_spoof_static['confidence']*100:.1f}%)")
                    elif best_spoof_static["confidence"] < 0.60:
                        reasons.append(f"Do tin cay mat that RF-DETR Small qua thap ({best_spoof_static['confidence']*100:.1f}% < 60%)")

                if has_any_spoof_in_frame:
                    reasons.append("Phat hien doi tuong gia mao trong khung hinh")

                if not blink_passed:
                    reasons.append("Khong vuot qua thu thach chop mat (Liveness)")

                if not head_movement_passed:
                    reasons.append(f"Khong vuot qua thu thach quay dau ({current_head_action.value})")

                final_pass = (len(reasons) == 0)

                # Vẽ ảnh kết quả sạch
                clean_annotated = captured_frame.copy()
                if primary_face:
                    px1, py1, px2, py2 = primary_face["bbox"]
                    box_color = (0, 255, 0) if final_pass else (0, 50, 255)
                    cv2.rectangle(clean_annotated, (px1, py1), (px2, py2), box_color, 2)

                    c_len = min(22, (px2 - px1) // 4, (py2 - py1) // 4)
                    cv2.line(clean_annotated, (px1, py1), (px1 + c_len, py1), box_color, 3)
                    cv2.line(clean_annotated, (px1, py1), (px1, py1 + c_len), box_color, 3)
                    cv2.line(clean_annotated, (px2, py1), (px2 - c_len, py1), box_color, 3)
                    cv2.line(clean_annotated, (px2, py1), (px2, py1 + c_len), box_color, 3)
                    cv2.line(clean_annotated, (px1, py2), (px1 + c_len, py2), box_color, 3)
                    cv2.line(clean_annotated, (px1, py2), (px1, py2 - c_len), box_color, 3)
                    cv2.line(clean_annotated, (px2, py2), (px2 - c_len, py2), box_color, 3)
                    cv2.line(clean_annotated, (px2, py2), (px2, py2 - c_len), box_color, 3)

                    tag_txt = f"{'REAL' if final_pass else 'SPOOF'} {best_spoof_static['confidence']*100:.1f}%" if best_spoof_static else "FACE"
                    (tw, th_t), _ = cv2.getTextSize(tag_txt, cv2.FONT_HERSHEY_SIMPLEX, 0.65, 2)
                    cv2.rectangle(clean_annotated, (px1, max(0, py1 - th_t - 12)), (px1 + tw + 16, py1), box_color, -1)
                    cv2.putText(clean_annotated, tag_txt, (px1 + 8, py1 - 5),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.65, (255, 255, 255), 2, cv2.LINE_AA)

                if landmarks_static:
                    draw_landmarks(clean_annotated, landmarks_static)

                if pose_dict_static:
                    draw_pose_info(clean_annotated, pose_dict_static, "Head Pose", pose_valid_static)

                dashboard_img = create_rfdetr_small_dashboard(
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
                    head_action_name=current_head_action.value if hasattr(current_head_action, 'value') else str(current_head_action),
                    final_pass=final_pass,
                    reasons=reasons,
                    face_crop=face_crop_static,
                    target_height=clean_annotated.shape[0],
                    model_id=model_id,
                    width=560
                )

                sbs_result = create_side_by_side_result(clean_annotated, dashboard_img)
                final_display_img = sbs_result

                # Lưu ảnh kết quả
                cv2.imwrite(os.path.join(captured_result_dir, "1_pipeline_result.jpg"), sbs_result)
                cv2.imwrite(os.path.join(captured_result_dir, "1_pipeline_result_clean.jpg"), clean_annotated)
                cv2.imwrite(os.path.join(captured_result_dir, "1_dashboard_panel.jpg"), dashboard_img)

                # Báo cáo JSON
                final_record = {
                    "image_id": current_img_idx,
                    "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
                    "final_verdict": "APPROVED" if final_pass else "REJECTED",
                    "decision_reasons": reasons,
                    "camera_index": cam_id,
                    "quick_snapshot_mode": quick_snapshot_mode,
                    "pipeline_architecture": "RF-DETR Small Detection Transformer",
                    "face_detection": {
                        "num_faces": num_faces,
                        "primary_face_bbox": primary_face["bbox"] if primary_face else None,
                        "confidence": float(primary_face["confidence"]) if primary_face else 0.0,
                        "all_faces": all_face_crops_info
                    },
                    "pose_validation": {
                        "is_valid": pose_valid_static,
                        "angles": pose_dict_static
                    },
                    "anti_spoof_rfdetr_small": {
                        "model_id": model_id,
                        "model_architecture": "RF-DETR Small (Detection Transformer)",
                        "weights_size_mb": round(weights_mb, 1),
                        "class_names": ["real", "spoof"],
                        "label": best_spoof_static["label"] if best_spoof_static else "UNKNOWN",
                        "confidence": float(best_spoof_static["confidence"]) if best_spoof_static else 0.0,
                        "is_real": best_spoof_static["is_real"] if best_spoof_static else False,
                        "iou_with_primary_face": round(float(primary_spoof_iou), 4),
                        "all_detections_in_frame": spoof_res
                    },
                    "active_liveness": {
                        "blink_challenge": {
                            "passed": blink_passed,
                            "blink_count": blink_counter
                        },
                        "head_movement_challenge": {
                            "passed": head_movement_passed,
                            "head_action": current_head_action.value if hasattr(current_head_action, 'value') else str(current_head_action)
                        }
                    }
                }

                out_json_path = os.path.join(captured_result_dir, "4_report.json")
                with open(out_json_path, "w", encoding="utf-8") as f:
                    json.dump(final_record, f, ensure_ascii=False, indent=2, default=json_serialize_helper)

                # Báo cáo CSV
                batch_csv_path = os.path.join(OUTPUT_DIR, "batch_summary_rfdetr_small.csv")
                file_exists = os.path.exists(batch_csv_path)
                with open(batch_csv_path, "a", newline="", encoding="utf-8-sig") as f:
                    writer = csv.writer(f)
                    if not file_exists:
                        writer.writerow([
                            "Image ID", "Verdict", "Num Faces", "RF-DETR Small Label", "Confidence",
                            "IoU", "Pose Valid", "Blink", "Head Movement", "Reasons", "Output Folder"
                        ])
                    writer.writerow([
                        f"{current_img_idx}.jpg",
                        final_record["final_verdict"],
                        num_faces,
                        final_record["anti_spoof_rfdetr_small"]["label"],
                        final_record["anti_spoof_rfdetr_small"]["confidence"],
                        f"{primary_spoof_iou:.2f}",
                        "PASS" if final_record["pose_validation"]["is_valid"] else "FAIL",
                        "PASS" if blink_passed else "FAIL",
                        f"PASS ({final_record['active_liveness']['head_movement_challenge']['head_action']})" if head_movement_passed else f"FAIL ({final_record['active_liveness']['head_movement_challenge']['head_action']})",
                        "; ".join(reasons) if reasons else "None",
                        captured_result_dir
                    ])

                print("\n" + "=" * 65)
                print(f"  [HOÀN TẤT eKYC RF-DETR SMALL ID: {current_img_idx}] Kết quả: {final_record['final_verdict']}")
                print(f"  * Ảnh gốc đã lưu      : {captured_img_path}")
                print(f"  * Thư mục kết quả     : {captured_result_dir}")
                print(f"  * Chi tiết 4_report   : {out_json_path}")
                print("=" * 65 + "\n")

            display = final_display_img.copy()
            disp_h, disp_w = display.shape[:2]
            draw_ui_card(display, 20, disp_h - 70, disp_w - 40, 50, bg_color=(15, 15, 20), alpha=0.85)
            cv2.putText(display, "[r]: Tiep tuc chup anh tiep theo | [q]: Thoat", (35, disp_h - 38),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 230, 255), 2)

        curr_time = time.time()
        fps = 1.0 / (curr_time - prev_fps_time) if curr_time > prev_fps_time else 0.0
        prev_fps_time = curr_time
        cv2.putText(display, f"FPS: {fps:.1f}", (w - 120, 45), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 200, 200), 1)

        cv2.imshow("Full E-KYC Pipeline (RF-DETR Small Anti-Spoof)", display)

        key = cv2.waitKey(1) & 0xFF
        if key in (27, ord('q'), ord('Q')):
            break

        elif key in (ord('r'), ord('R')):
            start_new_session()

        elif key in (ord('a'), ord('A')) and stage == PipelineStage.PREVIEW_ALIGN:
            auto_capture_mode = not auto_capture_mode
            print(f"[INFO] Chế độ Auto-Capture: {'BẬT' if auto_capture_mode else 'TẮT'}")

        elif key in (ord('s'), ord('S')):
            if stage == PipelineStage.PREVIEW_ALIGN:
                if not is_aligned_good:
                    capture_blocked_frames = 40
                    if not is_light_ok:
                        print(f"\n[CHẶN CHỤP] Ánh sáng quá yếu (Luminance={mean_lum:.1f} < 60.0)! Vui lòng bật đèn.")
                    else:
                        print("\n[CHẶN CHỤP] Không thể chụp! Vui lòng đưa khuôn mặt vào giữa khung oval và nhìn thẳng.")
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

        elif stage == PipelineStage.PREVIEW_ALIGN and (key in (32, ord('c'), ord('C')) or key_trigger == ord(' ')):
            if not is_aligned_good:
                capture_blocked_frames = 40
                if not is_light_ok:
                    print(f"\n[CHẶN CHỤP] Ánh sáng quá yếu (Luminance={mean_lum:.1f} < 60.0)! Vui lòng bật đèn.")
                else:
                    print("\n[CHẶN CHỤP] Không thể chụp! Vui lòng đưa khuôn mặt vào giữa khung oval và nhìn thẳng.")
            else:
                captured_frame = frame.copy()
                quick_snapshot_mode = False
                stage = PipelineStage.RUN_AI_STATIC
                print(f"\n[TRIGGER] Đã kích hoạt chụp ảnh Full Quy trình cho ID: {current_img_idx}!")

    cap.release()
    cv2.destroyAllWindows()
    print("[INFO] Đã đóng chương trình Pipeline RF-DETR Small an toàn.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Full E-KYC Pipeline RF-DETR Small (Detection Transformer Anti-Spoofing)")
    parser.add_argument("--cam", "--camera", type=int, default=0, help="Camera device index (mặc định 0)")
    parser.add_argument("--static", "--skip-liveness", "--quick", action="store_true", help="Chế độ chụp và lưu AI nhanh, bỏ qua thử thách Liveness")
    parser.add_argument("--model-id", type=str, default=DEFAULT_MODEL_ID, help=f"RF-DETR Model ID (mặc định: {DEFAULT_MODEL_ID})")
    args = parser.parse_args()

    try:
        main_pipeline_rfdetr_small(
            cam_id=args.cam,
            skip_liveness=getattr(args, 'static', False),
            model_id=args.model_id
        )
    except KeyboardInterrupt:
        print("\n[INFO] Đã dừng pipeline theo yêu cầu của người dùng.")
