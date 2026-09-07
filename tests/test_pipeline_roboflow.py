# -*- coding: utf-8 -*-
"""
=============================================================================
Full E-KYC Pipeline: Roboflow Anti-Spoofing + Active Liveness Verification
=============================================================================
Quy trình thực hiện toàn diện (End-to-End eKYC Verification Pipeline):
  1. Mở Webcam: Hiển thị giao diện xem trước & khung oval bán nguyệt căn chỉnh khuôn mặt.
     - Vùng bên ngoài khung oval được làm mờ (Gaussian Blur) và giảm sáng (Bokeh effect).
     - Khóa chụp ảnh (Capture Lock): Nếu khuôn mặt chưa đưa vào đúng khung oval hoặc
       chưa nhìn thẳng, hệ thống sẽ chặn không cho chụp và hiện cảnh báo.
  2. Chụp ảnh (Phím SPACE / 'c' hoặc Tự động khi mặt chuẩn trong oval):
     - Lưu ảnh gốc vào data_raw/<id>.jpg (đánh số tăng dần tiếp theo).
  3. Chạy AI Models trên ảnh vừa chụp:
     - Face Detection (YOLOv8) -> Landmarks (MediaPipe) -> Pose 3D -> Face Align & Crop 224x224.
     - Roboflow Anti-Spoofing Model (chạy cục bộ bằng inference.get_model, không cần Docker):
       + Model ID: face-spoof-detection-liika-qopyy/1
       + Tự động quét và phân loại khuôn mặt: Real vs Spoof Attack.
  4. Bắt đầu Active Liveness trên luồng Live Webcam:
     - Blink Detection: Yêu cầu người dùng chớp mắt (đo EAR).
     - Head Movement Challenge: Thử thách quay đầu ngẫu nhiên (Trái/Phải).
  5. Tổng hợp toàn bộ dữ liệu & Đưa ra quyết định cuối cùng (Final eKYC Decision).
  6. Hiển thị giao diện kết quả:
     - Chỉ hiển thị 1 khung nhận diện có tỉ lệ cao nhất (ẩn các khung tỉ lệ thấp hơn / trùng lặp).
     - Giao diện song song (Side-by-Side): Ảnh khuôn mặt bên trái + Dashboard thông số bên phải,
       hoàn toàn không che khuất khuôn mặt.
  7. Lưu toàn bộ kết quả vào output/pipeline_roboflow/<id>/ gồm:
     - 1_pipeline_result.jpg (Ảnh song song Side-by-Side)
     - 1_pipeline_result_clean.jpg (Ảnh khuôn mặt sạch)
     - 1_dashboard_panel.jpg (Bảng Dashboard độc lập)
     - 2_face_crop_224.jpg
     - 3_aligned_full.jpg
     - 4_report.json
     - Cập nhật batch_summary_roboflow.csv.

Phím điều khiển:
  - SPACE / 'c' : Chụp ảnh và bắt đầu quy trình eKYC (yêu cầu mặt trong oval)
  - 's'         : CHỤP NHANH & LƯU NGAY (Bỏ qua Active Liveness, yêu cầu mặt trong oval)
  - 'a'         : Bật / Tắt chế độ tự động chụp khi khuôn mặt đạt chuẩn trong oval
  - 'r'         : Khởi tạo lại phiên eKYC mới (tiếp tục ảnh mới)
  - 'q' / ESC   : Thoát chương trình
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

# Trỏ cache của Roboflow về thư mục models/roboflow của repo (đảm bảo chạy offline 100%)
ROBOFLOW_CACHE_DIR = os.path.join(BASE_DIR, "models", "roboflow")
os.environ["MODEL_CACHE_DIR"] = ROBOFLOW_CACHE_DIR

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
OUTPUT_DIR = os.path.join(CURRENT_DIR, "output", "pipeline_roboflow")
os.makedirs(DATA_RAW_DIR, exist_ok=True)
os.makedirs(OUTPUT_DIR, exist_ok=True)

ROBOFLOW_MODEL_ID = "face-spoof-detection-liika-qopyy/1"
ROBOFLOW_API_KEY = "LiYT7osRW01duX3ao91S"


# =============================================================================
# 1. HELPER FUNCTIONS: FONT, EAR & FILE INDEX MANAGEMENT
# =============================================================================
def remove_vietnamese_accents(text: str) -> str:
    """Chuyển đổi văn bản tiếng Việt có dấu thành không dấu để OpenCV cv2.putText hiển thị đẹp, không bị lỗi phông"""
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
    """
    Lọc các khung nhận diện bị trùng lặp hoặc đè lên nhau (IoU > iou_thresh).
    Chỉ giữ lại khung có tỉ lệ confidence cao nhất, ẩn hoàn toàn các khung có tỉ lệ thấp hơn.
    """
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

    l_top = (calc_dist(landmarks[160], landmarks[144]) + calc_dist(landmarks[158], landmarks[153])) / 2.0
    l_width = calc_dist(landmarks[33], landmarks[133])
    ear_left = (l_top / l_width) if l_width > 0 else 0.0

    r_top = (calc_dist(landmarks[385], landmarks[380]) + calc_dist(landmarks[387], landmarks[373])) / 2.0
    r_width = calc_dist(landmarks[362], landmarks[263])
    ear_right = (r_top / r_width) if r_width > 0 else 0.0

    ear_avg = (ear_left + ear_right) / 2.0
    return ear_left, ear_right, ear_avg


def get_next_image_index(data_dir=DATA_RAW_DIR):
    """Tìm số thứ tự tiếp theo cho ảnh mới trong thư mục data_raw"""
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
    """
    Vẽ khung oval bán nguyệt/elip ngay giữa màn hình để người dùng đưa khuôn mặt vào trước khi chụp.
    - Làm mờ nhòe (Gaussian Blur bokeh) và giảm sáng toàn bộ các vùng bên ngoài oval để tập trung sự chú ý vào khuôn mặt.
    - Vẽ viền oval phản hồi động theo trạng thái khuôn mặt kèm 4 vạch căn chỉnh công nghệ cao (biometric ticks).
    """
    h, w = image.shape[:2]
    cx, cy = center
    ax, ay = axes

    # 1. Tạo mask oval
    mask = np.zeros((h, w), dtype=np.uint8)
    cv2.ellipse(mask, (cx, cy), (ax, ay), 0, 0, 360, 255, -1)
    outside_mask = (mask == 0)

    # Làm mờ nhòe các vùng bên ngoài khung oval bằng Gaussian Blur
    blurred = cv2.GaussianBlur(image, (35, 35), 0)
    # Kết hợp làm mờ và giảm độ sáng (60% độ sáng) cho các vùng ngoài oval
    image[outside_mask] = (blurred[outside_mask] * 0.60).astype(np.uint8)

    # 2. Vẽ viền ngoài mỏng tạo hiệu ứng phát sáng (glow effect)
    glow_color = (int(color[0] * 0.35), int(color[1] * 0.35), int(color[2] * 0.35))
    cv2.ellipse(image, (cx, cy), (ax + 3, ay + 3), 0, 0, 360, glow_color, 1, cv2.LINE_AA)
    cv2.ellipse(image, (cx, cy), (max(10, ax - 3), max(10, ay - 3)), 0, 0, 360, glow_color, 1, cv2.LINE_AA)

    # 3. Vẽ đường viền oval chính
    thickness = 3 if is_aligned else 2
    cv2.ellipse(image, (cx, cy), (ax, ay), 0, 0, 360, color, thickness, cv2.LINE_AA)

    # 4. Vẽ 4 vạch căn chỉnh thước đo (Biometric ticks) ở 4 cực trên, dưới, trái, phải
    tick_len = 16
    cv2.line(image, (cx, cy - ay - tick_len), (cx, cy - ay + 6), color, 2, cv2.LINE_AA)
    cv2.line(image, (cx, cy + ay - 6), (cx, cy + ay + tick_len), color, 2, cv2.LINE_AA)
    cv2.line(image, (cx - ax - tick_len, cy), (cx - ax + 6, cy), color, 2, cv2.LINE_AA)
    cv2.line(image, (cx + ax - 6, cy), (cx + ax + tick_len, cy), color, 2, cv2.LINE_AA)

    return image


def create_roboflow_pipeline_dashboard(
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
    width: int = 560
) -> np.ndarray:
    """
    Tạo bảng Dashboard độc lập chuyên nghiệp cho Roboflow Anti-Spoofing Pipeline.
    Giao diện Dark Slate đồng bộ chuẩn mực với Pipeline Full.
    """
    clean_reasons = [remove_vietnamese_accents(r) for r in reasons] if (not final_pass and reasons) else []
    num_reasons = len(clean_reasons)
    extra_h = max(0, num_reasons * 24)

    min_h = 500 + extra_h
    h = max(min_h, target_height) if target_height else min_h
    w = max(500, width)

    # Nền Dark Slate cao cấp
    canvas = np.full((h, w, 3), (20, 22, 28), dtype=np.uint8)

    # 1. Header Card
    hdr_h = 70
    cv2.rectangle(canvas, (10, 10), (w - 10, hdr_h), (32, 36, 48), -1)
    cv2.rectangle(canvas, (10, 10), (w - 10, hdr_h), (60, 70, 90), 1)

    cv2.putText(canvas, "E-KYC VERIFICATION DASHBOARD", (24, 38),
                cv2.FONT_HERSHEY_SIMPLEX, 0.65, (0, 230, 255), 2, cv2.LINE_AA)
    session_str = f"Session ID: {img_idx} | Engine: Roboflow Inference"
    cv2.putText(canvas, session_str, (24, 58),
                cv2.FONT_HERSHEY_SIMPLEX, 0.40, (170, 180, 195), 1, cv2.LINE_AA)

    # Thumbnail khuôn mặt chuẩn hóa ở góc phải Header
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

    # Helper vẽ từng thẻ nội dung
    def _draw_card(title: str, lines: List[Tuple[str, Tuple[int, int, int], float]], card_h: int):
        nonlocal cur_y
        cv2.rectangle(canvas, (10, cur_y), (w - 10, cur_y + card_h), (27, 30, 40), -1)
        cv2.rectangle(canvas, (10, cur_y), (w - 10, cur_y + card_h), (50, 58, 75), 1)
        # Accent bar bên trái
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

    # Section 3: Anti-Spoofing (Roboflow Engine)
    if anti_spoof_info:
        as_lbl = anti_spoof_info.get("label", "UNKNOWN")
        as_conf = anti_spoof_info.get("confidence", 0.0)
        is_real = anti_spoof_info.get("is_real", False)
        as_col = (80, 220, 80) if is_real else (70, 70, 240)
        iou_str = f"  |  IoU with Face: {spoof_iou:.2f}" if spoof_iou > 0 else ""
        as_lines = [
            (f"Model Verdict: {as_lbl} ({as_conf*100:.1f}%){iou_str}", as_col, 0.44),
            (f"Classification: {'REAL FACE (Hop le)' if is_real else 'FAKE / SPOOF ATTACK (Phat hien gia mao)'}", as_col, 0.41),
            (f"Model ID: {ROBOFLOW_MODEL_ID}", (160, 180, 200), 0.38)
        ]

        card_h = 96
        cv2.rectangle(canvas, (10, cur_y), (w - 10, cur_y + card_h), (27, 30, 40), -1)
        cv2.rectangle(canvas, (10, cur_y), (w - 10, cur_y + card_h), (50, 58, 75), 1)
        cv2.rectangle(canvas, (10, cur_y), (14, cur_y + card_h), (0, 200, 240), -1)

        cv2.putText(canvas, "3. ANTI-SPOOFING (ROBOFLOW ENGINE)", (24, cur_y + 20),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.46, (220, 225, 235), 1, cv2.LINE_AA)

        line_y = cur_y + 38
        for text, col, font_scale in as_lines:
            cv2.putText(canvas, text, (24, line_y),
                        cv2.FONT_HERSHEY_SIMPLEX, font_scale, col, 1, cv2.LINE_AA)
            line_y += 18

        # Thanh tỉ lệ Real vs Fake
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
        _draw_card("3. ANTI-SPOOFING (ROBOFLOW ENGINE)", as_lines, card_h=52)

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

    # Footer note
    cv2.putText(canvas, "Press [r]: Tiep tuc chup anh tiep theo | [q]: Thoat", (24, h - 12),
                cv2.FONT_HERSHEY_SIMPLEX, 0.35, (100, 110, 130), 1, cv2.LINE_AA)

    return canvas


# =============================================================================
# 3. PIPELINE ROBOFLOW WORKFLOW STATE MACHINE
# =============================================================================
class PipelineStage(Enum):
    PREVIEW_ALIGN = 1       # Giai đoạn 1: Mở webcam, canh góc mặt & chờ chụp ảnh trong oval
    RUN_AI_STATIC = 2       # Giai đoạn 2: Chạy Face -> Landmark -> Pose -> Crop 224 -> Roboflow Anti-Spoof
    LIVE_BLINK = 3          # Giai đoạn 3: Active Liveness - Thử thách chớp mắt
    LIVE_HEAD_MOVEMENT = 4  # Giai đoạn 4: Active Liveness - Thử thách quay đầu
    FINAL_DECISION = 5      # Giai đoạn 5: Tổng hợp toàn bộ & lưu vào output/pipeline_roboflow/<id>/


def parse_roboflow_predictions(response, img_w, img_h):
    """Trích xuất kết quả nhận diện từ response của mô hình Roboflow"""
    dets = []
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


def main_pipeline_roboflow(cam_id=0, skip_liveness=False):
    print("\n" + "=" * 80)
    print("      FULL E-KYC PIPELINE (ROBOFLOW LOCAL ANTI-SPOOFING INFERENCE)")
    print("=" * 80)
    print(f"  * Thư mục lưu ảnh gốc : {DATA_RAW_DIR}")
    print(f"  * Thư mục lưu kết quả : {OUTPUT_DIR}")
    print(f"  * Model Roboflow ID   : {ROBOFLOW_MODEL_ID}")
    print("  * Nền tảng thực thi   : inference (Local Cache, Offline, Không cần Docker)")
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

    print(f"[INFO] Đang nạp mô hình Roboflow ({ROBOFLOW_MODEL_ID}) vào RAM...")
    roboflow_model = get_model(model_id=ROBOFLOW_MODEL_ID, api_key=ROBOFLOW_API_KEY)
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

    # Dữ liệu động từ Live Active Liveness
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
        nonlocal primary_face, faces, num_faces, all_face_crops_info, landmarks_static, pose_dict_static, pose_valid_static
        nonlocal face_crop_static, aligned_img_static, best_spoof_static, spoof_res
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

        # =====================================================================
        # GIAI ĐOẠN 1: PREVIEW & CANH CHỈNH KHUÔN MẶT TRONG KHUNG OVAL
        # =====================================================================
        if stage == PipelineStage.PREVIEW_ALIGN:
            oval_cx = w // 2
            oval_cy = int(h * 0.505)
            oval_ay = int(h * 0.38)
            oval_ax = int(oval_ay * 0.65)
            oval_center = (oval_cx, oval_cy)
            oval_axes = (oval_ax, oval_ay)

            landmarks_live = landmark_detector.detect(frame)
            pose_valid_live = False
            pose_dict_live = None

            face_in_oval = False
            is_too_far = False
            is_too_close = False
            is_off_center = False
            off_center_hint = ""

            if landmarks_live and len(landmarks_live) >= 468:
                pose_valid_live, _, pose_dict_live = pose_validator.validate(landmarks_live, get_landmark_point)
                display = draw_landmarks(display, landmarks_live)

                xs = [p[0] for p in landmarks_live]
                ys = [p[1] for p in landmarks_live]
                face_min_x, face_max_x = min(xs), max(xs)
                face_min_y, face_max_y = min(ys), max(ys)
                face_cx = (face_min_x + face_max_x) / 2.0
                face_cy = (face_min_y + face_max_y) / 2.0
                face_w = face_max_x - face_min_x
                face_h = face_max_y - face_min_y

                dx_norm = abs(face_cx - oval_cx) / float(oval_ax)
                dy_norm = abs(face_cy - oval_cy) / float(oval_ay)
                oval_total_h = 2 * oval_ay
                face_h_ratio = face_h / float(oval_total_h)

                if face_h_ratio < 0.46 or face_h < 150:
                    is_too_far = True
                elif face_h_ratio > 1.15 or face_w > oval_ax * 2.2:
                    is_too_close = True
                elif dx_norm > 0.32 or dy_norm > 0.32:
                    is_off_center = True
                    if face_cx < oval_cx - oval_ax * 0.25:
                        off_center_hint = "Di chuyen mat sang PHAI vao giua oval"
                    elif face_cx > oval_cx + oval_ax * 0.25:
                        off_center_hint = "Di chuyen mat sang TRAI vao giua oval"
                    elif face_cy < oval_cy - oval_ay * 0.25:
                        off_center_hint = "Di chuyen mat xuong DUOI vao giua oval"
                    else:
                        off_center_hint = "Di chuyen mat len TREN vao giua oval"
                else:
                    face_in_oval = True

            # Kiểm tra chất lượng ánh sáng khuôn mặt trong khung oval
            face_bbox_live = [int(face_min_x), int(face_min_y), int(face_max_x), int(face_max_y)] if (landmarks_live and len(landmarks_live) >= 468) else None
            light_res = check_illumination_quality(frame, bbox=face_bbox_live, dark_threshold=60.0)
            is_light_ok = light_res["is_acceptable"]
            mean_lum = light_res["mean_luminance"]

            # Đánh giá toàn diện: Có mặt trong oval + Góc nhìn 3D chuẩn + ĐỦ ÁNH SÁNG
            is_aligned_good = (landmarks_live is not None and face_in_oval and pose_valid_live and is_light_ok)

            if landmarks_live is None:
                guide_color = (200, 200, 200)
                align_msg = "VUI LONG DUA KHUON MAT VAO KHUNG OVAL"
                align_col = (220, 220, 220)
                consecutive_center_frames = max(0, consecutive_center_frames - 1)
            elif not is_light_ok:
                guide_color = (0, 140, 255)  # Màu cam đậm cảnh báo thiếu sáng
                align_msg = f"ANH SANG YEU (L:{mean_lum:.0f}/60) - VUI LONG BAT DEN HOAC TIEN VE PHIA SANG!"
                align_col = (0, 140, 255)
                consecutive_center_frames = max(0, consecutive_center_frames - 1)
            elif is_aligned_good:
                guide_color = (0, 255, 127)
                consecutive_center_frames += 1
                if auto_capture_mode:
                    pct = min(100, int(consecutive_center_frames / 25 * 100))
                    align_msg = f"MAT CHUAN TRONG KHUNG OVAL! DANG CHUP... ({pct}%)"
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
                    align_msg = f"ANH SANG YEU (L:{mean_lum:.0f}/60) - KHOA CHUP! VUI LONG BAT DEN."
                else:
                    align_msg = "CHUA DUA MAT VAO OVAL - KHONG THE NHAN CHUP!"
                align_col = (0, 0, 255)
                guide_color = (0, 0, 255)

            display = draw_oval_face_guide(
                display,
                center=oval_center,
                axes=oval_axes,
                is_aligned=is_aligned_good,
                is_detected=(landmarks_live is not None),
                color=guide_color
            )

            # Banner tiêu đề trên cùng
            draw_ui_card(display, 15, 8, w - 30, 48, bg_color=(15, 15, 25), alpha=0.85)
            cv2.putText(display, f"E-KYC ROBOFLOW: CANH CHINH KHUON MAT (ID: {current_img_idx}.jpg)", (28, 28),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.52, (0, 230, 255), 2, cv2.LINE_AA)
            mode_str = f"Auto-Capture: {'BAT' if auto_capture_mode else 'TAT'} | Sang (Luminance): {mean_lum:.0f}/255"
            cv2.putText(display, mode_str, (28, 46),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.38, (180, 180, 180), 1, cv2.LINE_AA)

            # Banner thông báo hướng dẫn dưới cùng
            bot_y = h - 56
            draw_ui_card(display, 15, bot_y, w - 30, 48, bg_color=(15, 15, 25), alpha=0.88)
            cv2.putText(display, align_msg, (28, bot_y + 22),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.48, align_col, 2, cv2.LINE_AA)

            if is_aligned_good:
                shortcut_hint = "[SPACE]/[c]: SAN SANG CHUP | [s]: Luu ngay | [a]: Auto | [r]: Reset | [q]: Thoat"
                shortcut_col = (0, 255, 127)
            elif not is_light_ok:
                shortcut_hint = f"[KHOA CHUP: THIEU SANG L:{mean_lum:.0f}/60] Vui long bat den de mo khoa chup"
                shortcut_col = (0, 140, 255)
            else:
                shortcut_hint = "[SPACE]/[c]: KHOA CHUP (Canh mat vao oval de mo khoa) | [a]: Auto | [q]: Thoat"
                shortcut_col = (150, 150, 150)

            cv2.putText(display, shortcut_hint,
                        (28, bot_y + 40), cv2.FONT_HERSHEY_SIMPLEX, 0.38, shortcut_col, 1, cv2.LINE_AA)

            if auto_capture_mode and consecutive_center_frames >= 25 and is_aligned_good:
                trigger_capture = True
            else:
                trigger_capture = False

            key_trigger = ord(' ') if trigger_capture else None

        # =====================================================================
        # GIAI ĐOẠN 2: CHẠY AI MODEL TRÊN ẢNH CHỤP ĐẾN BƯỚC ANTI-SPOOF
        # =====================================================================
        elif stage == PipelineStage.RUN_AI_STATIC:
            draw_ui_card(display, 20, 20, w - 40, 90, bg_color=(15, 15, 25), alpha=0.9)
            cv2.putText(display, f"DANG CHAY AI MODEL TREN ANH ID {current_img_idx}.jpg...", (35, 55),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.65, (0, 230, 255), 2)
            cv2.putText(display, "Tien trinh: Face Detect -> Landmark -> Pose 3D -> Crop 224 -> Roboflow Anti-Spoof", (35, 85),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 255, 0), 1)
            cv2.imshow("Full E-KYC Pipeline (Roboflow Anti-Spoof)", display)
            cv2.waitKey(1)

            # 1. Lưu ảnh gốc
            captured_img_path = os.path.join(DATA_RAW_DIR, f"{current_img_idx}.jpg")
            cv2.imwrite(captured_img_path, captured_frame)
            print(f"\n[1. CHỤP ẢNH GỐC] Đã lưu ảnh vào: {captured_img_path}")

            # 2. Tạo thư mục output/pipeline_roboflow/<id>/
            captured_result_dir = os.path.join(OUTPUT_DIR, str(current_img_idx))
            os.makedirs(captured_result_dir, exist_ok=True)
            all_faces_dir = os.path.join(captured_result_dir, "all_faces_cropped")
            os.makedirs(all_faces_dir, exist_ok=True)

            # 3. Chạy Face Detection
            faces = detector.detect(captured_frame)
            num_faces = len(faces)
            print(f"[2. Face Detection] Tìm thấy {num_faces} khuôn mặt trong khung hình.")

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

            # Chọn Primary Face
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
            landmarks_static = landmark_detector.detect(captured_frame)
            print(f"[3. Landmarks] Trích xuất được {len(landmarks_static) if landmarks_static else 0} điểm.")

            # 5. Pose 3D
            pose_valid_static = False
            pose_dict_static = None
            if landmarks_static:
                pose_valid_static, _, pose_dict_static = pose_validator.validate(landmarks_static, get_landmark_point)
                if pose_dict_static:
                    print(f"[4. Head Pose 3D] Y={pose_dict_static['yaw']:+.1f}° | P={pose_dict_static['pitch']:+.1f}° | R={pose_dict_static['roll']:+.1f}° -> {'PASS' if pose_valid_static else 'FAIL'}")

            # 6. Align & Crop 224x224
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

            # 7. Roboflow Anti-Spoofing Inference
            captured_light = check_illumination_quality(captured_frame, bbox=primary_face["bbox"] if primary_face else None)
            if captured_light["mean_luminance"] < 80.0:
                input_spoof = enhance_low_light(captured_frame)
                print(f"[Anti-Spoof Preprocess] Độ sáng L={captured_light['mean_luminance']:.1f}. Đã tự động áp dụng CLAHE tăng cường vi vân da mặt.")
            else:
                input_spoof = captured_frame

            print(f"[6. Anti-Spoofing Roboflow] Đang chạy suy luận mô hình {ROBOFLOW_MODEL_ID}...")
            rf_res = roboflow_model.infer(image=input_spoof)
            raw_spoof_res = parse_roboflow_predictions(rf_res, w_f, h_f)

            # Nếu không tìm thấy bbox trên toàn ảnh nhưng có primary_face, thử infer trên crop
            if not raw_spoof_res and face_crop_static is not None:
                rf_crop_res = roboflow_model.infer(image=face_crop_static)
                crop_preds = parse_roboflow_predictions(rf_crop_res, 224, 224)
                if crop_preds and primary_face:
                    # Map ngược lại bbox của primary_face
                    best_crop = max(crop_preds, key=lambda x: x["confidence"])
                    raw_spoof_res.append({
                        "bbox": primary_face["bbox"],
                        "confidence": best_crop["confidence"],
                        "class_name": best_crop["class_name"],
                        "is_real": best_crop["is_real"],
                        "label": best_crop["label"],
                        "raw_class": best_crop["raw_class"]
                    })

            # Lọc chỉ giữ khung có tỉ lệ cao nhất
            spoof_res = filter_highest_confidence_boxes(raw_spoof_res, iou_thresh=0.25)
            print(f"  -> Tìm thấy {len(raw_spoof_res)} vùng -> Lọc còn {len(spoof_res)} khung có tỉ lệ cao nhất.")

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
                print(f"  -> Kết quả Anti-Spoof Roboflow: {best_spoof_static['label']} ({best_spoof_static['confidence']*100:.1f}%) | IoU={primary_spoof_iou:.2f} | Real={best_spoof_static['is_real']}")

            if quick_snapshot_mode or skip_liveness:
                print("\n[INFO] Chế độ Quick Save / Skip Liveness -> Chuyển ngay đến Lưu Kết quả Final...")
                blink_passed = True
                head_movement_passed = True
                stage = PipelineStage.FINAL_DECISION
            else:
                print("\n[INFO] Chuyển sang giai đoạn Live Active Liveness (Blink & Head Movement)...")
                stage = PipelineStage.LIVE_BLINK
                blink_counter = 0
                blink_state = False
                blink_passed = False

        # =====================================================================
        # GIAI ĐOẠN 3: ACTIVE LIVENESS - BLINK DETECTION
        # =====================================================================
        elif stage == PipelineStage.LIVE_BLINK:
            landmarks_live = landmark_detector.detect(frame)
            ear_l, ear_r, ear_avg = compute_eye_aspect_ratio(landmarks_live) if landmarks_live else (0.0, 0.0, 0.0)

            if landmarks_live:
                display = draw_landmarks(display, landmarks_live)

            if ear_avg > 0.05 and ear_avg < 0.18:
                if not blink_state:
                    blink_state = True
            elif ear_avg >= 0.22:
                if blink_state:
                    blink_counter += 1
                    blink_state = False

            if blink_counter >= 1:
                blink_passed = True
                print(f"[LIVENESS 1: BLINK] ĐÃ XÁC NHẬN CHỚP MẮT ({blink_counter} lần) -> PASS!")
                stage = PipelineStage.LIVE_HEAD_MOVEMENT
                current_head_action = head_movement_detector.start_challenge()
                head_action_prompt = head_movement_detector.get_prompt()
                print(f"[LIVENESS 2: HEAD MOVEMENT] Thử thách: {current_head_action.value} -> {head_action_prompt}")

            draw_ui_card(display, 20, 20, w - 40, 110, bg_color=(20, 20, 25), alpha=0.85)
            cv2.putText(display, f"E-KYC BUOC 1/2: THU THACH CHOP MAT (ID: {current_img_idx})", (35, 48),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.62, (0, 230, 255), 2)
            cv2.putText(display, f"VUI LONG CHOP MAT (EAR: {ear_avg:.2f} | Blinks: {blink_counter}/1)", (35, 80),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)

            b_prog = 1.0 if blink_counter >= 1 else (0.5 if blink_state else 0.0)
            bar_w = w - 110
            cv2.rectangle(display, (35, 95), (35 + bar_w, 107), (50, 50, 50), -1)
            if b_prog > 0:
                cv2.rectangle(display, (35, 95), (35 + int(bar_w * b_prog), 107), (0, 255, 0), -1)
            cv2.rectangle(display, (35, 95), (35 + bar_w, 107), (120, 120, 120), 1)

        # =====================================================================
        # GIAI ĐOẠN 4: ACTIVE LIVENESS - HEAD MOVEMENT CHALLENGE
        # =====================================================================
        elif stage == PipelineStage.LIVE_HEAD_MOVEMENT:
            landmarks_live = landmark_detector.detect(frame)
            pose_dict_live = None
            if landmarks_live:
                _, _, pose_dict_live = pose_validator.validate(landmarks_live, get_landmark_point)
                display = draw_landmarks(display, landmarks_live)

            hm_status = head_movement_detector.update(pose_dict_live)
            prompt_str = hm_status.get("prompt", "")
            time_left = hm_status.get("time_left", 0.0)
            progress_val = hm_status.get("progress", 0.0)

            if hm_status["passed"]:
                head_movement_passed = True
                print(f"[LIVENESS 2: HEAD MOVEMENT] ĐÃ HOÀN THÀNH CỬ ĐỘNG ĐẦU [{current_head_action.value}] -> PASS!")
                stage = PipelineStage.FINAL_DECISION

            elif hm_status["state"] == "FAILED":
                head_movement_passed = False
                print(f"[LIVENESS 2: HEAD MOVEMENT] HẾT THỜI GIAN THỰC HIỆN -> FAIL!")
                stage = PipelineStage.FINAL_DECISION

            draw_ui_card(display, 20, 20, w - 40, 110, bg_color=(20, 20, 25), alpha=0.85)
            cv2.putText(display, f"E-KYC BUOC 2/2: THU THACH CU DONG DAU (ID: {current_img_idx})", (35, 48),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.62, (0, 230, 255), 2)

            hm_color = (0, 255, 0) if hm_status["passed"] else (0, 255, 255)
            cv2.putText(display, f"{prompt_str.upper()} ({time_left:.1f}s)", (35, 80),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.58, hm_color, 2)

            bar_w = w - 110
            cv2.rectangle(display, (35, 95), (35 + bar_w, 107), (50, 50, 50), -1)
            fill_w = int(bar_w * progress_val)
            if fill_w > 0:
                cv2.rectangle(display, (35, 95), (35 + fill_w, 107), (0, 255, 0), -1)
            cv2.rectangle(display, (35, 95), (35 + bar_w, 107), (120, 120, 120), 1)

        # =====================================================================
        # GIAI ĐOẠN 5: TỔNG HỢP KẾT QUẢ & LƯU VÀO OUTPUT/PIPELINE_ROBOFLOW/<ID>/
        # =====================================================================
        elif stage == PipelineStage.FINAL_DECISION:
            if final_record is None:
                c_face = (primary_face is not None)
                c_single = (num_faces == 1)
                c_pose = bool(pose_valid_static)
                c_spoof = bool(best_spoof_static["is_real"]) if best_spoof_static else False
                c_blink = bool(blink_passed)
                c_head = bool(head_movement_passed)

                reasons.clear()
                if not c_face:
                    reasons.append("Không tìm thấy khuôn mặt trong ảnh")
                elif not c_single:
                    reasons.append(f"Phát hiện {num_faces} người trong khung hình (Yêu cầu 1 người duy nhất)")

                if not c_pose:
                    reasons.append("Góc mặt ảnh chụp bị nghiêng/lệch")

                if not c_spoof:
                    reasons.append("Phát hiện giả mạo qua Roboflow Anti-Spoofing Model")
                elif has_any_spoof_in_frame:
                    print("  [CẢNH BÁO BỐI CẢNH] Phát hiện vật thể nghi ngờ ở nền xung quanh, nhưng khuôn mặt chính đạt chuẩn REAL.")

                if not c_blink:
                    reasons.append("Chưa hoàn thành chớp mắt (Blink)")
                if not c_head:
                    reasons.append("Chưa hoàn thành cử động đầu (Head Movement)")

                final_pass = (c_face and c_single and c_pose and c_spoof and c_blink and c_head)

                res_img = captured_frame.copy()

                if landmarks_static:
                    res_img = draw_landmarks(res_img, landmarks_static)

                # CHỈ HIỂN THỊ 1 KHUNG NHẬN DIỆN CÓ TỈ LỆ CAO NHẤT
                drawn_spoof_bboxes = []
                if spoof_res:
                    for sd in spoof_res:
                        sx1, sy1, sx2, sy2 = sd["bbox"]
                        s_col = (0, 255, 0) if sd["is_real"] else (0, 0, 255)
                        cv2.rectangle(res_img, (sx1, sy1), (sx2, sy2), s_col, 2)
                        cv2.putText(res_img, f"{sd['label']} {sd['confidence']*100:.1f}%",
                                    (sx1, max(25, sy1 - 10)), cv2.FONT_HERSHEY_SIMPLEX, 0.65, s_col, 2)
                        drawn_spoof_bboxes.append(sd["bbox"])

                # Vẽ khuôn mặt phụ (nếu có), ẩn khung primary face nếu đã có box anti-spoof
                for f_it in faces:
                    bx1, by1, bx2, by2 = f_it["bbox"]
                    is_p = (primary_face and f_it["bbox"] == primary_face["bbox"])
                    has_spoof_box = any(calculate_iou(f_it["bbox"], sb) > 0.20 for sb in drawn_spoof_bboxes)
                    if is_p and has_spoof_box:
                        continue
                    box_c = (0, 255, 0) if is_p else (180, 180, 180)
                    box_thick = 2 if is_p else 1
                    cv2.rectangle(res_img, (bx1, by1), (bx2, by2), box_c, box_thick)
                    lbl_tag = "PRIMARY FACE" if is_p else "EXTRA FACE"
                    cv2.putText(res_img, lbl_tag, (bx1, max(15, by1 - 5)), cv2.FONT_HERSHEY_SIMPLEX, 0.45, box_c, 1)

                # Tạo bảng Dashboard thông số độc lập
                dashboard_img = create_roboflow_pipeline_dashboard(
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
                    head_action_name=current_head_action.value if hasattr(current_head_action, "value") else str(current_head_action),
                    final_pass=final_pass,
                    reasons=reasons,
                    face_crop=face_crop_static,
                    target_height=h
                )

                # Ảnh kết quả sạch
                clean_img = res_img.copy()
                verdict_badge = "eKYC: APPROVED" if final_pass else "eKYC: REJECTED"
                badge_col = (0, 255, 0) if final_pass else (0, 0, 255)
                cv2.rectangle(clean_img, (w - 240, 15), (w - 15, 55), (15, 15, 20), -1)
                cv2.rectangle(clean_img, (w - 240, 15), (w - 15, 55), badge_col, 2)
                cv2.putText(clean_img, verdict_badge, (w - 225, 42),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.62, badge_col, 2)

                side_by_side_img = create_side_by_side_result(clean_img, dashboard_img)

                # Lưu các file vào thư mục output/pipeline_roboflow/<id>/
                out_clean_path = os.path.join(captured_result_dir, "1_pipeline_result_clean.jpg")
                cv2.imwrite(out_clean_path, clean_img)

                out_dash_path = os.path.join(captured_result_dir, "1_dashboard_panel.jpg")
                cv2.imwrite(out_dash_path, dashboard_img)

                out_sbs_path = os.path.join(captured_result_dir, "1_pipeline_side_by_side.jpg")
                cv2.imwrite(out_sbs_path, side_by_side_img)

                out_res_path = os.path.join(captured_result_dir, "1_pipeline_result.jpg")
                cv2.imwrite(out_res_path, side_by_side_img)

                final_display_img = side_by_side_img

                if face_crop_static is not None:
                    out_crop_path = os.path.join(captured_result_dir, "2_face_crop_224.jpg")
                    cv2.imwrite(out_crop_path, face_crop_static)

                if aligned_img_static is not None:
                    out_align_path = os.path.join(captured_result_dir, "3_aligned_full.jpg")
                    cv2.imwrite(out_align_path, aligned_img_static)

                # File báo cáo JSON
                final_record = {
                    "image_id": current_img_idx,
                    "image_name": f"{current_img_idx}.jpg",
                    "raw_image_path": captured_img_path,
                    "output_folder": captured_result_dir,
                    "model_type": f"Roboflow ({ROBOFLOW_MODEL_ID})",
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
                    "anti_spoof_roboflow": {
                        "model_id": ROBOFLOW_MODEL_ID,
                        "label": best_spoof_static["label"] if best_spoof_static else "NONE",
                        "is_real": bool(best_spoof_static["is_real"]) if best_spoof_static else False,
                        "confidence": round(best_spoof_static["confidence"], 4) if best_spoof_static else 0.0,
                        "matched_iou": round(primary_spoof_iou, 4),
                        "has_global_spoof_in_frame": bool(has_any_spoof_in_frame),
                        "all_spoof_detections": spoof_res
                    },
                    "active_liveness": {
                        "blink_passed": bool(blink_passed),
                        "blink_count": int(blink_counter),
                        "head_movement_passed": bool(head_movement_passed),
                        "head_action": current_head_action.value if hasattr(current_head_action, "value") else str(current_head_action),
                    },
                    "final_verdict": "APPROVED" if final_pass else "REJECTED",
                    "reasons": reasons
                }

                out_json_path = os.path.join(captured_result_dir, "4_report.json")
                with open(out_json_path, "w", encoding="utf-8") as f:
                    json.dump(final_record, f, ensure_ascii=False, indent=2, default=json_serialize_helper)

                # Báo cáo CSV
                batch_csv_path = os.path.join(OUTPUT_DIR, "batch_summary_roboflow.csv")
                file_exists = os.path.exists(batch_csv_path)
                with open(batch_csv_path, "a", newline="", encoding="utf-8-sig") as f:
                    writer = csv.writer(f)
                    if not file_exists:
                        writer.writerow([
                            "Image ID", "Verdict", "Num Faces", "Roboflow Label", "Confidence",
                            "IoU", "Pose Valid", "Blink", "Head Movement", "Reasons", "Output Folder"
                        ])
                    writer.writerow([
                        f"{current_img_idx}.jpg",
                        final_record["final_verdict"],
                        num_faces,
                        final_record["anti_spoof_roboflow"]["label"],
                        final_record["anti_spoof_roboflow"]["confidence"],
                        f"{primary_spoof_iou:.2f}",
                        "PASS" if final_record["pose_validation"]["is_valid"] else "FAIL",
                        "PASS" if blink_passed else "FAIL",
                        f"PASS ({final_record['active_liveness']['head_action']})" if head_movement_passed else f"FAIL ({final_record['active_liveness']['head_action']})",
                        "; ".join(reasons) if reasons else "None",
                        captured_result_dir
                    ])

                print("\n" + "=" * 65)
                print(f"  [HOÀN TẤT eKYC ROBOFLOW ID: {current_img_idx}] Kết quả: {final_record['final_verdict']}")
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

        cv2.imshow("Full E-KYC Pipeline (Roboflow Anti-Spoof)", display)

        key = cv2.waitKey(1) & 0xFF
        if key == 27 or key == ord('q') or key == ord('Q'):
            break

        elif key == ord('r') or key == ord('R'):
            start_new_session()

        elif (key == ord('a') or key == ord('A')) and stage == PipelineStage.PREVIEW_ALIGN:
            auto_capture_mode = not auto_capture_mode
            print(f"[INFO] Chế độ Auto-Capture: {'BẬT' if auto_capture_mode else 'TẮT'}")

        elif (key == ord('s') or key == ord('S')):
            if stage == PipelineStage.PREVIEW_ALIGN:
                if not is_aligned_good:
                    capture_blocked_frames = 40
                    if not is_light_ok:
                        print(f"\n[CHẶN CHỤP] Ánh sáng quá yếu (Luminance={mean_lum:.1f} < 60.0)! Vui lòng bật đèn hoặc di chuyển ra nơi đủ sáng.")
                    else:
                        print("\n[CHẶN CHỤP] Không thể chụp! Vui lòng đưa khuôn mặt vào giữa khung oval và nhìn thẳng trước.")
                else:
                    captured_frame = frame.copy()
                    quick_snapshot_mode = True
                    stage = PipelineStage.RUN_AI_STATIC
                    print(f"\n[QUICK SAVE] Đã kích hoạt Chụp nhanh & Lưu ngay cho ID: {current_img_idx}!")
            elif stage in (PipelineStage.LIVE_BLINK, PipelineStage.LIVE_HEAD_MOVEMENT):
                print("\n[QUICK SAVE] Bỏ qua các bước Liveness tiếp theo và Lưu kết quả ngay lập tức!")
                blink_passed = True
                head_movement_passed = True
                stage = PipelineStage.FINAL_DECISION

        elif (key == 32 or key == ord('c') or key == ord('C') or key_trigger == ord(' ')) and stage == PipelineStage.PREVIEW_ALIGN:
            if not is_aligned_good:
                capture_blocked_frames = 40
                if not is_light_ok:
                    print(f"\n[CHẶN CHỤP] Ánh sáng quá yếu (Luminance={mean_lum:.1f} < 60.0)! Vui lòng bật đèn hoặc di chuyển ra nơi đủ sáng.")
                else:
                    print("\n[CHẶN CHỤP] Không thể chụp! Vui lòng đưa khuôn mặt vào giữa khung oval và nhìn thẳng trước.")
            else:
                captured_frame = frame.copy()
                quick_snapshot_mode = False
                stage = PipelineStage.RUN_AI_STATIC
                print(f"\n[TRIGGER] Đã kích hoạt chụp ảnh Full Quy trình cho ID: {current_img_idx}!")

    cap.release()
    cv2.destroyAllWindows()
    print("[INFO] Đã đóng chương trình Pipeline Roboflow an toàn.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Full E-KYC Pipeline (Roboflow Anti-Spoofing Local Inference)")
    parser.add_argument("--cam", "--camera", type=int, default=0, help="Camera device index (mặc định 0)")
    parser.add_argument("--static", "--skip-liveness", "--quick", action="store_true", help="Chế độ chụp và lưu AI nhanh, bỏ qua thử thách Liveness")
    args = parser.parse_args()

    try:
        main_pipeline_roboflow(
            cam_id=args.cam,
            skip_liveness=getattr(args, 'static', False)
        )
    except KeyboardInterrupt:
        print("\n[INFO] Đã dừng pipeline theo yêu cầu của người dùng.")
