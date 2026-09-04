"""
=============================================================================
Full E-KYC Pipeline: MobileNetV2 Anti-Spoofing & Active Liveness Detection
=============================================================================
Quy trình chuẩn eKYC tương tác trực quan (chuẩn quy trình pipeline_full):
  1. Giao diện Webcam Preview & Canh chỉnh khuôn mặt (Live Preview):
     - Kiểm tra khoảng cách, góc xoay đầu 3D (Yaw, Pitch, Roll).
     - Hỗ trợ phím [SPACE] / [c] chụp thủ công hoặc [a] chụp tự động khi mặt chuẩn.
  2. Chụp ảnh & Lưu dữ liệu gốc:
     - Lưu ảnh chụp chất lượng cao vào data_raw/<id>.jpg (đánh số tự động tăng dần).
  3. Chạy AI Model trên ảnh chụp (Tĩnh):
     - Face Detection (YOLO Face_Detection.pt) -> Crop toàn bộ khuôn mặt trong khung hình.
     - Landmark Detection (MediaPipe 478 điểm mốc).
     - 3D Head Pose Validation.
     - Face Alignment & Face Crop chuẩn hóa kích thước 224x224.
     - Passive Anti-Spoofing với MobileNetV2 (Hugging Face / Safetensors 224x224, ngưỡng mặc định 0.6).
  4. Active Liveness trên luồng Live Webcam:
     - Bước 1/2: Thử thách chớp mắt tự nhiên (đo chỉ số EAR).
     - Bước 2/2: Thử thách cử động đầu ngẫu nhiên (Quay trái, Quay phải).
  5. Tổng hợp quyết định eKYC cuối cùng (Final eKYC Decision Engine):
     - Đánh giá tổng thể 6 tiêu chí an toàn: Có mặt, Đơn nhân, Pose chuẩn, Anti-Spoof REAL, Chớp mắt, Quay đầu.
     - Xuất báo cáo chi tiết vào output/pipeline_mobilenet/<id>/:
       + 1_pipeline_result.jpg (Ảnh kết quả kèm HUD dashboard)
       + 1_pipeline_result_clean.jpg (Ảnh sạch chỉ có badge kết quả)
       + 2_face_crop_224.jpg (Ảnh crop chuẩn 224x224 đưa vào MobileNetV2)
       + 3_aligned_full.jpg (Ảnh đã xoay thẳng trục mắt)
       + 4_report.json (Chi tiết kỹ thuật toàn bộ các bước)
     - Cập nhật tự động batch_summary_mobilenet.csv và batch_summary_mobilenet.json.

Phím tắt điều khiển (Webcam Mode):
  - [SPACE] hoặc [c] : Chụp ảnh và bắt đầu chu trình eKYC đầy đủ
  - [s]              : Chụp nhanh & Lưu ngay (Bỏ qua thử thách Liveness)
  - [a]              : Bật/Tắt chế độ tự động chụp khi khuôn mặt chuẩn
  - [r]              : Bắt đầu phiên eKYC mới (ảnh ID tiếp theo)
  - [q] hoặc [ESC]   : Thoát chương trình
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
from enum import Enum
from pathlib import Path
from typing import Optional, List, Dict, Any, Tuple

if hasattr(sys.stdout, "reconfigure"):
    try:
        sys.stdout.reconfigure(encoding="utf-8")
        sys.stderr.reconfigure(encoding="utf-8")
    except Exception:
        pass

# Kiểm tra và xử lý lỗi thiếu thư viện trước khi import
try:
    import cv2
    import numpy as np
    import torch
except ModuleNotFoundError as err:
    print("\n" + "=" * 75)
    print(f" [LỖI MÔI TRƯỜNG PYTHON] : {err}")
    print("=" * 75)
    print(f" Python hiện tại đang chạy : {sys.executable}")
    print(" Môi trường Python này chưa có thư viện OpenCV (cv2) hoặc PyTorch.")
    print(" Gợi ý cách chạy chính xác:")
    print("   👉 Cách 1 (Khuyên dùng): Dùng trình khởi chạy 'py' của Windows:")
    print("        py tests/test_pipeline_mobilenet.py")
    print("   👉 Cách 2: Dùng trực tiếp Python 3.11 đã cài sẵn thư viện:")
    print("        & \"C:/Users/HP/AppData/Local/Programs/Python/Python311/python.exe\" tests/test_pipeline_mobilenet.py")
    print("=" * 75 + "\n")
    sys.exit(1)

# Thiết lập đường dẫn import
CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
BASE_DIR = os.path.dirname(CURRENT_DIR)  # Face-Project/

if BASE_DIR not in sys.path:
    sys.path.insert(0, BASE_DIR)

from src.face_detection import FaceDetector
from src.landmark_detection import LandmarkDetector
from src.landmark_detection.draw_landmarks import draw_landmarks
from src.landmark_detection.utils import get_landmark_point
from src.pose_validation import PoseValidator
from src.pose_validation.draw_pose import draw_pose_info
from src.face_alignment_crop import FaceAligner
from src.head_movement import HeadMovementDetector, HeadAction, ChallengeState
from src.anti_spoof.mobilenetv2 import AntiSpoofMobileNetV2, resolve_mobilenetv2_paths

DATA_RAW_DIR = os.path.join(BASE_DIR, "data_raw")
DEFAULT_OUTPUT_DIR = os.path.join(CURRENT_DIR, "output", "pipeline_mobilenet")
IMAGE_EXTENSIONS = {'.jpg', '.jpeg', '.png', '.bmp', '.webp', '.tiff'}


def remove_vietnamese_accents(text: str) -> str:
    """Chuyển đổi văn bản tiếng Việt có dấu thành không dấu để OpenCV cv2.putText hiển thị đẹp, không bị lỗi phông"""
    if not text:
        return ""
    text = str(text)
    text = text.replace("đ", "d").replace("Đ", "D")
    nfkd = unicodedata.normalize("NFKD", text)
    return "".join([c for c in nfkd if not unicodedata.combining(c)])


# =============================================================================
# 1. HELPER FUNCTIONS: EAR, QUẢN LÝ THỨ TỰ ẢNH & JSON SERIALIZER
# =============================================================================
def calc_dist(p1, p2):
    """Tính khoảng cách Euclidean giữa 2 điểm (x, y)"""
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


def compute_mouth_aspect_ratio(landmarks):
    """Tính MAR (Mouth Aspect Ratio) từ MediaPipe 478 landmarks"""
    if not landmarks or len(landmarks) < 468:
        return 0.0

    vertical = calc_dist(landmarks[13], landmarks[14])
    horizontal = calc_dist(landmarks[61], landmarks[291])
    mar = (vertical / horizontal) if horizontal > 0 else 0.0
    return mar


def get_next_image_index(data_dir: str) -> int:
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
# 2. GIAO DIỆN HUD & VISUALIZATION UTILITIES (Chuẩn UI pipeline_full)
# =============================================================================
def draw_ui_card(image: np.ndarray, x: int, y: int, w: int, h: int, bg_color=(15, 15, 20), alpha=0.85):
    """Vẽ khung card bán trong suốt làm nền HUD"""
    overlay = image.copy()
    cv2.rectangle(overlay, (x, y), (x + w, y + h), bg_color, -1)
    cv2.addWeighted(overlay, alpha, image, 1 - alpha, 0, image)
    cv2.rectangle(image, (x, y), (x + w, y + h), (100, 100, 100), 1)


def draw_mobilenet_prob_bar(image: np.ndarray, x: int, y: int, w: int, real_score: float, fake_score: float):
    """Vẽ thanh đo xác suất Real vs Fake của MobileNetV2"""
    bar_h = 12
    cv2.rectangle(image, (x, y), (x + w, y + bar_h), (35, 35, 40), -1)

    real_w = int(w * min(1.0, max(0.0, real_score)))
    if real_w > 0:
        cv2.rectangle(image, (x, y), (x + real_w, y + bar_h), (46, 204, 113), -1)
    if real_w < w:
        cv2.rectangle(image, (x + real_w, y), (x + w, y + bar_h), (60, 76, 231), -1)

    cv2.rectangle(image, (x, y), (x + w, y + bar_h), (120, 120, 120), 1)


def draw_pipeline_mobilenet_result_hud(
    image: np.ndarray,
    img_idx: int,
    face_info: Optional[dict],
    num_faces: int,
    pose_info: Optional[dict],
    pose_valid: bool,
    spoof_info: Optional[dict],
    blink_passed: bool,
    blink_count: int,
    head_movement_passed: bool,
    head_action_name: str,
    final_pass: bool,
    reasons: List[str],
    has_spoof_face: bool = False,
    spoof_faces_count: int = 0
) -> np.ndarray:
    """Vẽ bảng dashboard báo cáo chi tiết kết quả eKYC lên frame (chuẩn phong cách pipeline_full)"""
    h, w = image.shape[:2]
    vis = image.copy()

    clean_reasons = [remove_vietnamese_accents(r) for r in reasons] if (not final_pass and reasons) else []
    num_reasons = len(clean_reasons)
    extra_h = max(0, num_reasons * 22) if num_reasons > 0 else 0

    card_w = min(560, w - 20)
    card_h = min(h - 25, 255 + extra_h)
    draw_ui_card(vis, 15, 15, card_w, card_h, bg_color=(15, 15, 20), alpha=0.88)

    # Tiêu đề card
    cv2.putText(vis, f"E-KYC MOBILENETV2 REPORT (ID: {img_idx})", (25, 42),
                cv2.FONT_HERSHEY_SIMPLEX, 0.62, (0, 230, 255), 2, cv2.LINE_AA)
    cv2.line(vis, (25, 50), (15 + card_w - 20, 50), (80, 80, 80), 1)

    # 1. Face Detection & Single Person Rule
    if face_info:
        if num_faces == 1:
            f_txt = f"1. Face Detect   : 1 FACE (CONF: {face_info['confidence']:.2f}) -> PASS"
            f_col = (0, 255, 0)
        else:
            f_txt = f"1. Face Detect   : MULTI-FACE ({num_faces} FACES) -> REJECT"
            f_col = (0, 0, 255)
    else:
        f_txt = "1. Face Detect   : NO FACE DETECTED -> FAIL"
        f_col = (0, 0, 255)
    cv2.putText(vis, f_txt, (25, 72), cv2.FONT_HERSHEY_SIMPLEX, 0.44, f_col, 1, cv2.LINE_AA)

    # 2. Pose 3D
    if pose_info:
        yaw = pose_info.get("yaw", 0.0)
        pitch = pose_info.get("pitch", 0.0)
        roll = pose_info.get("roll", 0.0)
        p_stat = "PASS" if pose_valid else "FAIL"
        p_txt = f"2. Head Pose [{p_stat}] : Y:{yaw:+.1f} P:{pitch:+.1f} R:{roll:+.1f}"
        p_col = (0, 255, 0) if pose_valid else (0, 0, 255)
    else:
        p_txt = "2. Head Pose     : UNKNOWN"
        p_col = (0, 0, 255)
    cv2.putText(vis, p_txt, (25, 95), cv2.FONT_HERSHEY_SIMPLEX, 0.44, p_col, 1, cv2.LINE_AA)

    # 3. MobileNetV2 Anti-Spoofing
    if has_spoof_face:
        as_txt = f"3. MobileNetV2   : SPOOF DETECTED ({spoof_faces_count} FAKE FACE) -> REJECT"
        as_col = (0, 0, 255)
        cv2.putText(vis, as_txt, (25, 118), cv2.FONT_HERSHEY_SIMPLEX, 0.44, as_col, 1, cv2.LINE_AA)
        r_sc = spoof_info.get("real_score", 0.0) if spoof_info else 0.0
        f_sc = spoof_info.get("fake_score", 1.0) if spoof_info else 1.0
        draw_mobilenet_prob_bar(vis, 25, 126, min(card_w - 40, 480), r_sc, f_sc)
    elif spoof_info:
        as_lbl = spoof_info["label"]
        r_sc = spoof_info.get("real_score", 0.0)
        f_sc = spoof_info.get("fake_score", 0.0)
        as_col = (0, 255, 0) if spoof_info["is_real"] else (0, 0, 255)
        as_txt = f"3. MobileNetV2   : {as_lbl} (Real: {r_sc*100:.1f}% | Fake: {f_sc*100:.1f}%)"
        cv2.putText(vis, as_txt, (25, 118), cv2.FONT_HERSHEY_SIMPLEX, 0.44, as_col, 1, cv2.LINE_AA)
        draw_mobilenet_prob_bar(vis, 25, 126, min(card_w - 40, 480), r_sc, f_sc)
    else:
        as_txt = "3. MobileNetV2   : NO DATA"
        as_col = (0, 165, 255)
        cv2.putText(vis, as_txt, (25, 118), cv2.FONT_HERSHEY_SIMPLEX, 0.44, as_col, 1, cv2.LINE_AA)

    # 4. Blink Liveness
    b_txt = f"4. Blink Liveness: PASS ({blink_count} blinks)" if blink_passed else f"4. Blink Liveness: FAIL ({blink_count} blinks)"
    b_col = (0, 255, 0) if blink_passed else (0, 0, 255)
    cv2.putText(vis, b_txt, (25, 152), cv2.FONT_HERSHEY_SIMPLEX, 0.44, b_col, 1, cv2.LINE_AA)

    # 5. Head Movement Liveness
    hm_txt = f"5. Head Movement : PASS [{head_action_name.upper()}]" if head_movement_passed else f"5. Head Movement : FAIL [{head_action_name.upper()}]"
    hm_col = (0, 255, 0) if head_movement_passed else (0, 0, 255)
    cv2.putText(vis, hm_txt, (25, 175), cv2.FONT_HERSHEY_SIMPLEX, 0.44, hm_col, 1, cv2.LINE_AA)

    cv2.line(vis, (25, 192), (15 + card_w - 20, 192), (80, 80, 80), 1)

    # 6. Final Decision
    verdict_text = "eKYC: APPROVED (HOP LE)" if final_pass else "eKYC: REJECTED (TU CHOI)"
    verdict_col = (0, 255, 0) if final_pass else (0, 0, 255)
    cv2.putText(vis, verdict_text, (25, 220),
                cv2.FONT_HERSHEY_SIMPLEX, 0.68, verdict_col, 2, cv2.LINE_AA)

    # Chi tiết các lý do từ chối
    if not final_pass and clean_reasons:
        cv2.putText(vis, "Ly do tu choi:", (25, 242),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.44, (0, 200, 255), 1, cv2.LINE_AA)
        start_y = 262
        line_spacing = 20
        for idx_r, r_text in enumerate(clean_reasons[:5]):
            if len(r_text) > 65:
                r_text = r_text[:62] + "..."
            line_txt = f" * {r_text}"
            cv2.putText(vis, line_txt, (25, start_y + idx_r * line_spacing),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.40, (140, 210, 255), 1, cv2.LINE_AA)

    return vis


# =============================================================================
# 3. PIPELINE STAGES (State Machine chuẩn quy trình eKYC)
# =============================================================================
class PipelineStage(Enum):
    PREVIEW_ALIGN = 1       # Giai đoạn 1: Mở webcam, canh góc mặt & chờ chụp ảnh
    RUN_AI_STATIC = 2       # Giai đoạn 2: Chạy Face -> Landmark -> Pose -> Align/Crop -> MobileNetV2 trên ảnh chụp
    LIVE_BLINK = 3          # Giai đoạn 3: Active Liveness - Thử thách chớp mắt
    LIVE_HEAD_MOVEMENT = 4  # Giai đoạn 4: Active Liveness - Thử thách quay đầu
    FINAL_DECISION = 5      # Giai đoạn 5: Hiển thị kết quả duyệt eKYC, lưu file & chờ phiên mới


# =============================================================================
# 4. WEBCAM INTERACTIVE eKYC PIPELINE (Chế độ mặc định)
# =============================================================================
def run_webcam_pipeline(
    cam_id: int = 0,
    output_dir: str = DEFAULT_OUTPUT_DIR,
    models_bundle: tuple = None,
    real_threshold: float = 0.6,
    scale_factor: float = 1.2,
    skip_liveness: bool = False
):
    """
    Quy trình eKYC đầy đủ chuẩn pipeline_full tích hợp mô hình MobileNetV2:
      1. Preview & Canh chỉnh mặt
      2. Chụp ảnh lưu data_raw/<id>.jpg
      3. Chạy AI Model (Face Detect, Landmark, Pose 3D, Align & Crop 224x224, MobileNetV2 Anti-Spoof)
      4. Active Liveness (Blink EAR -> Head Movement Challenge)
      5. Final eKYC Decision -> Lưu báo cáo đầy đủ
    """
    detector, landmark_detector, pose_validator, aligner, anti_spoof_mobilenet, head_movement_detector = models_bundle

    print("\n" + "=" * 75)
    print("      E-KYC PIPELINE: MOBILENETV2 (CHỤP ẢNH -> AI MODEL -> ACTIVE LIVENESS)")
    print("=" * 75)
    print(f"  * Thư mục lưu ảnh gốc : {DATA_RAW_DIR}")
    print(f"  * Thư mục lưu kết quả : {output_dir}")
    print(f"  * Model Anti-Spoof    : MobileNetV2 ({anti_spoof_mobilenet.weight_name})")
    print(f"  * Ngưỡng Real Thresh  : {real_threshold}")
    print("  * Điều khiển:")
    print("      [SPACE] hoặc [c]  : Chụp ảnh và bắt đầu chu trình eKYC (AI + Active Liveness)")
    print("      [s]               : CHỤP NHANH & LƯU NGAY (Bỏ qua thử thách Liveness)")
    print("      [a]               : Bật/Tắt chế độ tự động chụp khi mặt chuẩn")
    print("      [r]               : Khởi tạo lại phiên eKYC mới (ảnh tiếp theo)")
    print("      [q] hoặc [ESC]    : Thoát")
    print("=" * 75 + "\n")

    os.makedirs(DATA_RAW_DIR, exist_ok=True)
    os.makedirs(output_dir, exist_ok=True)

    cap = cv2.VideoCapture(cam_id)
    if not cap.isOpened():
        print(f"[ERROR] Không thể mở Camera ID {cam_id}!")
        return

    # Trạng thái luồng
    stage = PipelineStage.PREVIEW_ALIGN
    auto_capture_mode = False
    quick_snapshot_mode = False
    consecutive_center_frames = 0

    current_img_idx = get_next_image_index(DATA_RAW_DIR)
    captured_frame = None
    captured_img_path = None
    captured_result_dir = None

    # Dữ liệu tĩnh từ ảnh chụp
    primary_face = None
    num_faces = 0
    faces = []
    landmarks_static = None
    pose_dict_static = None
    pose_valid_static = False
    face_crop_static = None
    aligned_img_static = None
    spoof_info_static = None
    has_spoof_face = False
    spoof_faces_count = 0

    # Dữ liệu động từ Active Liveness
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
        nonlocal primary_face, num_faces, faces, landmarks_static, pose_dict_static, pose_valid_static
        nonlocal face_crop_static, aligned_img_static, spoof_info_static, has_spoof_face, spoof_faces_count
        nonlocal blink_counter, blink_state, blink_passed, head_movement_passed, current_head_action, head_action_prompt
        nonlocal final_pass, reasons, final_display_img, final_record, consecutive_center_frames, quick_snapshot_mode

        current_img_idx = get_next_image_index(DATA_RAW_DIR)
        stage = PipelineStage.PREVIEW_ALIGN
        captured_frame = None
        captured_img_path = None
        captured_result_dir = None
        quick_snapshot_mode = False

        primary_face = None
        num_faces = 0
        faces.clear()
        landmarks_static = None
        pose_dict_static = None
        pose_valid_static = False
        face_crop_static = None
        aligned_img_static = None
        spoof_info_static = None
        has_spoof_face = False
        spoof_faces_count = 0

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

        print(f"\n[PHIÊN MỚI] Sẵn sàng chụp ảnh ID tiếp theo: {current_img_idx}.jpg")

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        frame = cv2.flip(frame, 1)
        h, w = frame.shape[:2]
        display = frame.copy()

        # =====================================================================
        # GIAI ĐOẠN 1: PREVIEW & CANH CHỈNH KHUÔN MẶT
        # =====================================================================
        if stage == PipelineStage.PREVIEW_ALIGN:
            # Chỉ phát hiện vị trí mốc mặt và góc nhìn tạm thời để canh khung ảnh (nhẹ máy)
            landmarks_live = landmark_detector.detect(frame)
            pose_valid_live = False

            if landmarks_live:
                pose_valid_live, _, _ = pose_validator.validate(landmarks_live, get_landmark_point)
                display = draw_landmarks(display, landmarks_live)

            # Khung banner hướng dẫn
            draw_ui_card(display, 20, 20, w - 40, 115, bg_color=(15, 15, 25), alpha=0.85)
            cv2.putText(display, f"E-KYC MOBILENETV2: CHUAN BI CHUP ANH (ID: {current_img_idx}.jpg)", (35, 48),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.60, (0, 230, 255), 2)

            face_size_h = 0
            is_too_far = False
            if landmarks_live and len(landmarks_live) >= 468:
                ys = [p[1] for p in landmarks_live]
                face_size_h = max(ys) - min(ys)
                if face_size_h < 165:
                    is_too_far = True

            is_aligned_good = (landmarks_live is not None and pose_valid_live and not is_too_far)
            if is_aligned_good:
                align_msg = "Goc mat CHUAN! Nhan [SPACE] hoac [c] de chup anh"
                align_col = (0, 255, 0)
                consecutive_center_frames += 1
            elif is_too_far:
                align_msg = "Vui long tien lai GAN CAMERA hon (Khuon mat qua nho)..."
                align_col = (0, 165, 255)
                consecutive_center_frames = max(0, consecutive_center_frames - 1)
            else:
                align_msg = "Vui long nhin thang, giu mat chinh giua khung hinh..."
                align_col = (0, 200, 255)
                consecutive_center_frames = max(0, consecutive_center_frames - 1)

            cv2.putText(display, align_msg, (35, 80), cv2.FONT_HERSHEY_SIMPLEX, 0.52, align_col, 2)

            mode_str = f"Auto-Capture: {'BAT (Chup sau 2s)' if auto_capture_mode else 'TAT (Nhan SPACE de chup)'} | Real Thresh: {real_threshold}"
            cv2.putText(display, mode_str, (35, 110), cv2.FONT_HERSHEY_SIMPLEX, 0.44, (180, 180, 180), 1)

            if auto_capture_mode and consecutive_center_frames >= 25:
                key_trigger = ord(' ')
            else:
                key_trigger = None

        # =====================================================================
        # GIAI ĐOẠN 2: CHẠY AI MODEL TĨNH TẬP TRUNG TÊN ẢNH VỪA CHỤP
        # =====================================================================
        elif stage == PipelineStage.RUN_AI_STATIC:
            draw_ui_card(display, 20, 20, w - 40, 90, bg_color=(15, 15, 25), alpha=0.9)
            cv2.putText(display, f"DANG CHAY AI MODEL TREN ANH ID {current_img_idx}.jpg...", (35, 55),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.65, (0, 230, 255), 2)
            cv2.putText(display, "Tien trinh: Face Detect -> Landmark -> Pose 3D -> Crop 224 -> MobileNetV2", (35, 85),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.48, (0, 255, 0), 1)
            cv2.imshow("Full E-KYC Pipeline - MobileNetV2", display)
            cv2.waitKey(1)

            # 1. Lưu ảnh gốc vào data_raw/<id>.jpg
            captured_img_path = os.path.join(DATA_RAW_DIR, f"{current_img_idx}.jpg")
            cv2.imwrite(captured_img_path, captured_frame)
            print(f"\n[1. CHỤP ẢNH GỐC] Đã lưu ảnh vào: {captured_img_path}")

            # 2. Tạo thư mục output/pipeline_mobilenet/<id>/ và thư mục con all_faces_cropped/
            captured_result_dir = os.path.join(output_dir, str(current_img_idx))
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

            # Chọn Khuôn mặt chính (Primary Face: To nhất và gần trung tâm)
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
                if num_faces > 1:
                    print(f"  [CẢNH BÁO] Phát hiện {num_faces} người trong ảnh! Đã crop lưu tất cả mặt vào all_faces_cropped/")

            # 4. Chạy Landmarks
            landmarks_static = landmark_detector.detect(captured_frame)
            print(f"[3. Landmarks] Trích xuất được {len(landmarks_static) if landmarks_static else 0} điểm.")

            # 5. Chạy Pose 3D
            pose_valid_static = False
            pose_dict_static = None
            if landmarks_static:
                pose_valid_static, _, pose_dict_static = pose_validator.validate(landmarks_static, get_landmark_point)
                if pose_dict_static:
                    print(f"[4. Head Pose 3D] Y={pose_dict_static['yaw']:+.1f}° | P={pose_dict_static['pitch']:+.1f}° | R={pose_dict_static['roll']:+.1f}° -> {'PASS' if pose_valid_static else 'FAIL'}")

            # 6. Face Alignment (Xoay thẳng trục mắt phục vụ ảnh chân dung 3_aligned_full.jpg)
            aligned_img_static = None
            if landmarks_static:
                aligned_img_static = aligner.align_face(captured_frame, landmarks_static)

            if aligned_img_static is None:
                aligned_img_static = captured_frame.copy()

            out_align_p = os.path.join(captured_result_dir, "3_aligned_full.jpg")
            cv2.imwrite(out_align_p, aligned_img_static)

            # 7. Dự đoán Anti-Spoofing với MobileNetV2 cho TẤT CẢ các khuôn mặt trong ảnh
            has_spoof_face = False
            spoof_faces_count = 0
            spoof_info_static = None
            face_crop_static = None

            print(f"[6. MobileNetV2 Anti-Spoof] Đang phân tích chống giả mạo cho {num_faces} khuôn mặt...")
            for idx_f, f_item in enumerate(faces, 1):
                f_bbox = f_item["bbox"]
                f_spoof = anti_spoof_mobilenet.predict_face(captured_frame, f_bbox, scale=scale_factor)
                r_sc = f_spoof["real_score"]
                f_spoof["is_real"] = (r_sc >= real_threshold)
                f_spoof["label"] = "REAL" if f_spoof["is_real"] else "FAKE"
                f_spoof["confidence"] = r_sc if f_spoof["is_real"] else f_spoof["fake_score"]
                f_item["spoof_info"] = f_spoof

                if not f_spoof["is_real"]:
                    has_spoof_face = True
                    spoof_faces_count += 1

                is_primary_tag = " (PRIMARY FACE)" if (primary_face and f_bbox == primary_face["bbox"]) else ""
                print(f"  * Mặt #{idx_f}{is_primary_tag}: {f_spoof['label']} (Real: {r_sc*100:.1f}%, Fake: {f_spoof['fake_score']*100:.1f}%) | BBox: {f_bbox}")

                # Cập nhật kết quả vào danh sách lưu file
                for af in all_face_crops_info:
                    if af["face_index"] == idx_f:
                        af["is_real"] = f_spoof["is_real"]
                        af["label"] = f_spoof["label"]
                        af["real_score"] = round(r_sc, 4)
                        af["fake_score"] = round(f_spoof["fake_score"], 4)

            # Chọn kết quả của Primary Face
            if primary_face and "spoof_info" in primary_face:
                spoof_info_static = primary_face["spoof_info"]
                if "face_crop" in spoof_info_static and spoof_info_static["face_crop"] is not None:
                    face_crop_static = cv2.resize(spoof_info_static["face_crop"], (224, 224))
            elif landmarks_static:
                aligned_lms = aligner.get_landmarks(aligned_img_static)
                if aligned_lms:
                    face_crop_static = aligner.crop_face(aligned_img_static, aligned_lms, padding=20, output_size=(224, 224))
                    spoof_info_static = anti_spoof_mobilenet.predict_crop(face_crop_static)

            if face_crop_static is None:
                face_crop_static = cv2.resize(captured_frame, (224, 224))

            out_crop_p = os.path.join(captured_result_dir, "2_face_crop_224.jpg")
            cv2.imwrite(out_crop_p, face_crop_static)
            print(f"[5. Face Crop 224x224] Cắt ảnh chuẩn BBox scale={scale_factor}x thành công.")

            if has_spoof_face:
                print(f"  ⚠️ [CẢNH BÁO AN NINH] Phát hiện {spoof_faces_count} khuôn mặt giả mạo (SPOOF) trong khung hình!")
            elif spoof_info_static:
                print(f"  -> Kết quả Primary Face: {spoof_info_static['label']} (Real: {spoof_info_static['real_score']*100:.1f}%) | Ngưỡng: {real_threshold}")

            # Phân nhánh tiếp theo: Nếu chụp nhanh (Snapshot) hoặc bỏ qua liveness -> Chuyển ngay đến FINAL_DECISION
            if quick_snapshot_mode or skip_liveness:
                print("\n[INFO] Chế độ Quick Save / Skip Liveness -> Chuyển ngay đến Lưu Kết quả Final...")
                blink_passed = True
                head_movement_passed = True
                stage = PipelineStage.FINAL_DECISION
            else:
                # THỰC HIỆN TIẾP BƯỚC TIẾP THEO TRONG LỘ TRÌNH: ACTIVE LIVENESS
                print("\n[TIẾP TỤC LỘ TRÌNH] Bắt đầu Active Liveness (Blink Detection & Head Movement)...")
                print("  (Mẹo: Nhấn phím 's' bất cứ lúc nào để lưu kết quả ngay lập tức)")
                stage = PipelineStage.LIVE_BLINK
                blink_counter = 0
                blink_state = False
                blink_passed = False

        # =====================================================================
        # GIAI ĐOẠN 3: ACTIVE LIVENESS - THỬ THÁCH CHỚP MẮT (LIVE WEBCAM)
        # =====================================================================
        elif stage == PipelineStage.LIVE_BLINK:
            landmarks_live = landmark_detector.detect(frame)
            ear_l, ear_r, ear_avg = compute_eye_aspect_ratio(landmarks_live) if landmarks_live else (0.0, 0.0, 0.0)

            if landmarks_live:
                display = draw_landmarks(display, landmarks_live)

            # Thuật toán đếm chớp mắt
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
                # Chuyển tiếp sang thử thách cử động đầu
                stage = PipelineStage.LIVE_HEAD_MOVEMENT
                current_head_action = head_movement_detector.start_challenge()
                head_action_prompt = head_movement_detector.get_prompt()
                print(f"[LIVENESS 2: HEAD MOVEMENT] Thử thách: {current_head_action.value} -> {head_action_prompt}")

            # Vẽ HUD Blink
            draw_ui_card(display, 20, 20, w - 40, 110, bg_color=(20, 20, 25), alpha=0.85)
            cv2.putText(display, f"E-KYC BUOC 1/2: THU THACH CHOP MAT (ID: {current_img_idx})", (35, 48),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.62, (0, 230, 255), 2)
            cv2.putText(display, f"VUI LONG CHOP MAT (EAR: {ear_avg:.2f} | Blinks: {blink_counter}/1)", (35, 80),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.60, (0, 255, 255), 2)

            b_prog = 1.0 if blink_counter >= 1 else (0.5 if blink_state else 0.0)
            bar_w = w - 110
            cv2.rectangle(display, (35, 95), (35 + bar_w, 107), (50, 50, 50), -1)
            if b_prog > 0:
                cv2.rectangle(display, (35, 95), (35 + int(bar_w * b_prog), 107), (0, 255, 0), -1)
            cv2.rectangle(display, (35, 95), (35 + bar_w, 107), (120, 120, 120), 1)

        # =====================================================================
        # GIAI ĐOẠN 4: ACTIVE LIVENESS - THỬ THÁCH CỬ ĐỘNG ĐẦU (LIVE WEBCAM)
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

            # Vẽ HUD Head Movement
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
        # GIAI ĐOẠN 5: TỔNG HỢP KẾT QUẢ & LƯU BÁO CÁO OUTPUT
        # =====================================================================
        elif stage == PipelineStage.FINAL_DECISION:
            if final_record is None:
                c_face = (primary_face is not None)
                c_single = (num_faces == 1)
                c_pose = pose_valid_static
                is_primary_real = (spoof_info_static is not None and spoof_info_static["is_real"])
                c_spoof = (is_primary_real and not has_spoof_face)
                c_blink = blink_passed
                c_head = head_movement_passed

                reasons.clear()
                if not c_face:
                    reasons.append("Không tìm thấy khuôn mặt trong ảnh")
                elif not c_single:
                    reasons.append(f"Phát hiện {num_faces} khuôn mặt trong khung hình (Yêu cầu 1 người duy nhất)")

                if not c_pose:
                    reasons.append("Góc mặt ảnh chụp bị nghiêng/lệch")

                if has_spoof_face:
                    reasons.append(f"Phát hiện {spoof_faces_count} khuôn mặt giả mạo (SPOOF) trong khung hình!")
                elif not is_primary_real:
                    reasons.append(f"Khuôn mặt chính không đạt chuẩn Real (Score < {real_threshold})")

                if not c_blink:
                    reasons.append("Chưa hoàn thành chớp mắt (Blink)")
                if not c_head:
                    reasons.append("Chưa hoàn thành cử động đầu (Head Movement)")

                final_pass = (c_face and c_single and c_pose and c_spoof and c_blink and c_head)

                # Vẽ Dashboard kết quả lên ảnh chụp gốc
                res_img = captured_frame.copy()

                # Vẽ Bounding Box và nhãn MobileNetV2 cho TẤT CẢ các khuôn mặt tìm thấy (giống test_anti_spoof_mobilenetv2)
                for idx_f, f_it in enumerate(faces, 1):
                    bx1, by1, bx2, by2 = f_it["bbox"]
                    is_p = (primary_face and f_it["bbox"] == primary_face["bbox"])
                    f_sp = f_it.get("spoof_info", None)

                    if f_sp:
                        is_real_f = f_sp["is_real"]
                        b_col = (46, 204, 113) if is_real_f else (60, 76, 231)  # Xanh (Real) / Đỏ (Fake)
                        border_col = (0, 255, 127) if is_real_f else (0, 0, 255)
                        p_prefix = "[CHÍNH] " if is_p else ""
                        conf_pct = f_sp["real_score"] * 100 if is_real_f else f_sp["fake_score"] * 100
                        lbl_txt = f"{p_prefix}Face #{idx_f}: {f_sp['label']} ({conf_pct:.0f}%)"
                    else:
                        b_col = (200, 200, 200)
                        border_col = (200, 200, 200)
                        lbl_txt = f"Face #{idx_f}"

                    cv2.rectangle(res_img, (bx1, by1), (bx2, by2), b_col, 2)
                    cv2.putText(res_img, lbl_txt, (bx1, max(22, by1 - 8)),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.52, border_col, 2, cv2.LINE_AA)

                if landmarks_static:
                    res_img = draw_landmarks(res_img, landmarks_static)

                final_display_img = draw_pipeline_mobilenet_result_hud(
                    image=res_img,
                    img_idx=current_img_idx,
                    face_info=primary_face,
                    num_faces=num_faces,
                    pose_info=pose_dict_static,
                    pose_valid=pose_valid_static,
                    spoof_info=spoof_info_static,
                    blink_passed=blink_passed,
                    blink_count=blink_counter,
                    head_movement_passed=head_movement_passed,
                    head_action_name=current_head_action.value,
                    final_pass=final_pass,
                    reasons=reasons,
                    has_spoof_face=has_spoof_face,
                    spoof_faces_count=spoof_faces_count
                )

                # 1. Lưu 1_pipeline_result.jpg
                out_res_path = os.path.join(captured_result_dir, "1_pipeline_result.jpg")
                cv2.imwrite(out_res_path, final_display_img)

                # 2. Lưu 1_pipeline_result_clean.jpg (Ảnh sạch có badge)
                clean_img = res_img.copy()
                verdict_badge = "eKYC: APPROVED" if final_pass else "eKYC: REJECTED"
                badge_col = (0, 255, 0) if final_pass else (0, 0, 255)
                cv2.rectangle(clean_img, (w - 240, 15), (w - 15, 55), (15, 15, 20), -1)
                cv2.rectangle(clean_img, (w - 240, 15), (w - 15, 55), badge_col, 2)
                cv2.putText(clean_img, verdict_badge, (w - 225, 42),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.62, badge_col, 2)
                out_clean_path = os.path.join(captured_result_dir, "1_pipeline_result_clean.jpg")
                cv2.imwrite(out_clean_path, clean_img)

                # 3. Lưu 4_report.json
                final_record = {
                    "image_id": current_img_idx,
                    "image_name": f"{current_img_idx}.jpg",
                    "raw_image_path": captured_img_path,
                    "output_folder": captured_result_dir,
                    "model_anti_spoof": "MobileNetV2",
                    "real_threshold": real_threshold,
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
                    "mobilenetv2_anti_spoof": {
                        "label": spoof_info_static["label"] if spoof_info_static else "NONE",
                        "is_real": bool(spoof_info_static["is_real"]) if spoof_info_static else False,
                        "confidence": round(spoof_info_static["confidence"], 4) if spoof_info_static else 0.0,
                        "real_score": round(spoof_info_static["real_score"], 4) if spoof_info_static else 0.0,
                        "fake_score": round(spoof_info_static["fake_score"], 4) if spoof_info_static else 0.0,
                        "has_spoof_face_in_frame": bool(has_spoof_face),
                        "spoof_faces_count": int(spoof_faces_count),
                        "weight_name": anti_spoof_mobilenet.weight_name
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

                # 4. Cập nhật Báo cáo tổng kết batch_summary_mobilenet.csv
                batch_csv_path = os.path.join(output_dir, "batch_summary_mobilenet.csv")
                file_exists = os.path.exists(batch_csv_path)
                with open(batch_csv_path, "a", newline="", encoding="utf-8-sig") as f:
                    writer = csv.writer(f)
                    if not file_exists:
                        writer.writerow([
                            "Image ID", "Verdict", "Num Faces", "MobileNet Label", "Real Score", "Fake Score",
                            "Pose Valid", "Blink", "Head Movement", "Reasons", "Output Folder"
                        ])
                    writer.writerow([
                        f"{current_img_idx}.jpg",
                        final_record["final_verdict"],
                        num_faces,
                        final_record["mobilenetv2_anti_spoof"]["label"],
                        f"{final_record['mobilenetv2_anti_spoof']['real_score']*100:.1f}%",
                        f"{final_record['mobilenetv2_anti_spoof']['fake_score']*100:.1f}%",
                        "PASS" if final_record["pose_validation"]["is_valid"] else "FAIL",
                        "PASS" if blink_passed else "FAIL",
                        f"PASS ({current_head_action.value})" if head_movement_passed else f"FAIL ({current_head_action.value})",
                        "; ".join(reasons) if reasons else "None",
                        captured_result_dir
                    ])

                print("\n" + "=" * 65)
                print(f"  [HOÀN TẤT eKYC ID: {current_img_idx}] Kết quả: {final_record['final_verdict']}")
                print(f"  * Ảnh gốc đã lưu      : {captured_img_path}")
                print(f"  * Thư mục kết quả     : {captured_result_dir}")
                print(f"  * Chi tiết 4_report   : {out_json_path}")
                print("=" * 65 + "\n")

            display = final_display_img.copy()
            draw_ui_card(display, 20, h - 70, w - 40, 50, bg_color=(15, 15, 20), alpha=0.85)
            cv2.putText(display, "[r]: Tiep tuc chup anh tiep theo | [q]: Thoat", (35, h - 38),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 230, 255), 2)

        # Vẽ FPS ở góc phải trên
        curr_time = time.time()
        fps = 1.0 / (curr_time - prev_fps_time) if curr_time > prev_fps_time else 0.0
        prev_fps_time = curr_time
        cv2.putText(display, f"FPS: {fps:.1f}", (w - 120, 45), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 200, 200), 1)

        cv2.imshow("Full E-KYC Pipeline - MobileNetV2", display)

        # Xử lý phím bấm
        key = cv2.waitKey(1) & 0xFF
        if key == 27 or key == ord('q'):
            break

        elif key == ord('r'):
            start_new_session()

        elif key == ord('a') and stage == PipelineStage.PREVIEW_ALIGN:
            auto_capture_mode = not auto_capture_mode
            print(f"[INFO] Chế độ Auto-Capture: {'BẬT' if auto_capture_mode else 'TẮT'}")

        # Phím 's': Chụp ảnh nhanh (Snapshot Mode) và lưu kết quả ngay lập tức
        elif (key == ord('s') or key == ord('S')):
            if stage == PipelineStage.PREVIEW_ALIGN:
                captured_frame = frame.copy()
                quick_snapshot_mode = True
                stage = PipelineStage.RUN_AI_STATIC
                print(f"\n[QUICK SAVE] Đã kích hoạt Chụp nhanh & Lưu ngay cho ID: {current_img_idx}!")
            elif stage in (PipelineStage.LIVE_BLINK, PipelineStage.LIVE_HEAD_MOVEMENT):
                print("\n[QUICK SAVE] Bỏ qua các bước Liveness tiếp theo và Lưu kết quả ngay lập tức!")
                blink_passed = True
                head_movement_passed = True
                stage = PipelineStage.FINAL_DECISION

        # Phím SPACE hoặc 'c': Chụp ảnh và chạy Full quy trình eKYC
        elif (key == 32 or key == ord('c') or key_trigger == ord(' ')) and stage == PipelineStage.PREVIEW_ALIGN:
            captured_frame = frame.copy()
            quick_snapshot_mode = False
            stage = PipelineStage.RUN_AI_STATIC
            print(f"\n[TRIGGER] Đã kích hoạt chụp ảnh Full Quy trình cho ID: {current_img_idx}!")

    cap.release()
    cv2.destroyAllWindows()
    print("[INFO] Đã đóng chương trình Pipeline MobileNetV2 an toàn.")


# =============================================================================
# 5. BATCH DIRECTORY & SINGLE IMAGE PROCESSING (Tùy chọn phụ khi có cờ)
# =============================================================================
def process_single_image(
    image_path: str,
    output_root_dir: str,
    models_bundle: tuple,
    real_threshold: float = 0.6,
    scale_factor: float = 1.2
) -> Optional[dict]:
    detector, landmark_detector, pose_validator, aligner, anti_spoof_mobilenet, _ = models_bundle

    filename = os.path.basename(image_path)
    stem_name = os.path.splitext(filename)[0]

    img_output_dir = os.path.join(output_root_dir, stem_name)
    os.makedirs(img_output_dir, exist_ok=True)

    print(f"\n{'='*70}")
    print(f"  [XỬ LÝ ẢNH] : {filename}")
    print(f"  -> Thư mục lưu kết quả : {img_output_dir}")
    print(f"{'='*70}")

    image = cv2.imread(image_path)
    if image is None:
        print(f"[ERROR] Không đọc được file ảnh: {image_path}")
        return None

    img_h, img_w = image.shape[:2]
    t0 = time.perf_counter()

    # Step 1: Face Detection
    faces = detector.detect(image)
    primary_face = faces[0] if len(faces) > 0 else None
    print(f"[1. Face Detection] Tìm thấy {len(faces)} khuôn mặt.")

    # Step 2: Landmarks
    landmarks = landmark_detector.detect(image)
    print(f"[2. Landmarks] Trích xuất được {len(landmarks) if landmarks else 0} điểm.")

    # Step 3: Pose
    pose_valid = False
    pose_dict = None
    if landmarks and len(landmarks) > 0:
        pose_valid, _, pose_dict = pose_validator.validate(landmarks, get_landmark_point)

    # Step 4: Face Alignment (cho lưu ảnh hồ sơ)
    aligned_img = None
    if landmarks and len(landmarks) > 0:
        aligned_img = aligner.align_face(image, landmarks)

    # Step 5: MobileNetV2 Anti-Spoofing cho TẤT CẢ các khuôn mặt trong ảnh
    has_spoof_face = False
    spoof_faces_count = 0
    spoof_info = None
    face_crop = None

    for idx_f, f_item in enumerate(faces, 1):
        f_res = anti_spoof_mobilenet.predict_face(image, f_item["bbox"], scale=scale_factor)
        r_sc = f_res["real_score"]
        f_res["is_real"] = (r_sc >= real_threshold)
        f_res["label"] = "REAL" if f_res["is_real"] else "FAKE"
        f_res["confidence"] = r_sc if f_res["is_real"] else f_res["fake_score"]
        f_item["spoof_info"] = f_res
        if not f_res["is_real"]:
            has_spoof_face = True
            spoof_faces_count += 1
        print(f"  * Mặt #{idx_f}: {f_res['label']} (Real: {r_sc*100:.1f}%, Fake: {f_res['fake_score']*100:.1f}%) | BBox: {f_item['bbox']}")

    if primary_face and "spoof_info" in primary_face:
        spoof_info = primary_face["spoof_info"]
        if "face_crop" in spoof_info and spoof_info["face_crop"] is not None:
            face_crop = cv2.resize(spoof_info["face_crop"], (224, 224))
    elif landmarks and len(landmarks) > 0:
        aligned_lms = aligner.get_landmarks(aligned_img)
        if aligned_lms:
            face_crop = aligner.crop_face(aligned_img, aligned_lms, padding=20, output_size=(224, 224))
            spoof_info = anti_spoof_mobilenet.predict_crop(face_crop)

    # Step 6: EAR
    ear_l, ear_r, ear_avg = compute_eye_aspect_ratio(landmarks) if landmarks else (0.0, 0.0, 0.0)
    eye_open = (ear_avg >= 0.20)

    # Final Decision
    c1 = (primary_face is not None)
    c2 = pose_valid
    c3 = (spoof_info is not None and spoof_info["is_real"] and not has_spoof_face)
    c4 = eye_open
    c_single = (len(faces) == 1)

    reasons = []
    if not c1: reasons.append("Không tìm thấy khuôn mặt")
    if not c_single: reasons.append(f"Phát hiện {len(faces)} người trong ảnh (Yêu cầu 1 người duy nhất)")
    if not c2: reasons.append("Góc nghiêng mặt chưa chuẩn (Pose)")
    if has_spoof_face: reasons.append(f"Phát hiện {spoof_faces_count} khuôn mặt giả mạo (SPOOF) trong ảnh!")
    elif not (spoof_info and spoof_info["is_real"]): reasons.append(f"Nghi vấn giả mạo (Real Score < {real_threshold})")
    if not c4: reasons.append("Mắt đang nhắm hoặc quá hẹp")

    final_pass = (c1 and c2 and c3 and c4 and c_single)
    elapsed_ms = (time.perf_counter() - t0) * 1000.0

    res_vis = image.copy()
    for idx_f, f_it in enumerate(faces, 1):
        bx1, by1, bx2, by2 = f_it["bbox"]
        f_sp = f_it.get("spoof_info", None)
        if f_sp:
            is_real_f = f_sp["is_real"]
            b_col = (46, 204, 113) if is_real_f else (60, 76, 231)
            border_col = (0, 255, 127) if is_real_f else (0, 0, 255)
            conf_pct = f_sp["real_score"] * 100 if is_real_f else f_sp["fake_score"] * 100
            lbl_str = f"#{idx_f} {f_sp['label']} ({conf_pct:.0f}%)"
        else:
            b_col = (200, 200, 200)
            border_col = (200, 200, 200)
            lbl_str = f"Face #{idx_f}"

        cv2.rectangle(res_vis, (bx1, by1), (bx2, by2), b_col, 2)
        cv2.putText(res_vis, lbl_str, (bx1, max(20, by1 - 8)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.52, border_col, 2, cv2.LINE_AA)

    if landmarks:
        try:
            res_vis = draw_landmarks(res_vis, landmarks)
        except Exception:
            pass

    final_vis = draw_pipeline_mobilenet_result_hud(
        image=res_vis,
        img_idx=0,
        face_info=primary_face,
        num_faces=len(faces),
        pose_info=pose_dict,
        pose_valid=pose_valid,
        spoof_info=spoof_info,
        blink_passed=True,
        blink_count=1,
        head_movement_passed=True,
        head_action_name="STATIC_IMAGE",
        final_pass=final_pass,
        reasons=reasons,
        has_spoof_face=has_spoof_face,
        spoof_faces_count=spoof_faces_count
    )

    cv2.imwrite(os.path.join(img_output_dir, "1_pipeline_result.jpg"), final_vis)
    if face_crop is not None:
        cv2.imwrite(os.path.join(img_output_dir, "2_face_crop_224.jpg"), face_crop)
    if aligned_img is not None:
        cv2.imwrite(os.path.join(img_output_dir, "3_aligned_full.jpg"), aligned_img)

    rep_data = {
        "filename": filename,
        "process_time_ms": round(elapsed_ms, 2),
        "final_decision": "APPROVED" if final_pass else "REJECTED",
        "is_approved": bool(final_pass),
        "reasons": reasons,
        "mobilenetv2": spoof_info
    }
    with open(os.path.join(img_output_dir, "4_report.json"), "w", encoding="utf-8") as f:
        json.dump(rep_data, f, ensure_ascii=False, indent=2, default=json_serialize_helper)

    print(f"[LƯU THÀNH CÔNG] Đã lưu kết quả ảnh vào: {img_output_dir}")
    return rep_data


def run_batch_directory(
    data_dir: str,
    output_dir: str,
    models_bundle: tuple,
    real_threshold: float = 0.6,
    scale_factor: float = 1.2,
    show_vis: bool = True
):
    if not os.path.exists(data_dir):
        print(f"[ERROR] Thư mục '{data_dir}' không tồn tại!")
        return

    all_files = sorted([
        f for f in os.listdir(data_dir)
        if os.path.splitext(f)[1].lower() in IMAGE_EXTENSIONS
    ])

    if not all_files:
        print(f"[WARNING] Không tìm thấy ảnh hợp lệ trong {data_dir}!")
        return

    print("\n" + "=" * 75)
    print(f" DUYỆT BATCH THƯ MỤC ẢNH VỚI MOBILENETV2 (TỔNG: {len(all_files)})")
    print("=" * 75)

    os.makedirs(output_dir, exist_ok=True)
    for fname in all_files:
        img_path = os.path.join(data_dir, fname)
        process_single_image(
            image_path=img_path,
            output_root_dir=output_dir,
            models_bundle=models_bundle,
            real_threshold=real_threshold,
            scale_factor=scale_factor
        )


# =============================================================================
# 6. MAIN CONTROLLER
# =============================================================================
def main():
    parser = argparse.ArgumentParser(description="Full E-KYC Pipeline: MobileNetV2 Anti-Spoofing & Liveness")
    parser.add_argument("--cam", type=int, default=0, help="Chỉ số thiết bị camera (mặc định 0)")
    parser.add_argument("--static", "--skip-liveness", action="store_true", help="Chế độ chụp và lưu AI nhanh, bỏ qua thử thách Liveness")
    parser.add_argument("--thresh", type=float, default=0.6, help="Ngưỡng phân loại REAL threshold (mặc định 0.6)")
    parser.add_argument("--scale", type=float, default=1.2, help="Tỷ lệ crop mở rộng khuôn mặt (mặc định 1.2)")
    parser.add_argument("--batch", action="store_true", help="Chạy chế độ duyệt toàn bộ thư mục ảnh data_raw/")
    parser.add_argument("--dir", type=str, default=None, help="Đường dẫn thư mục ảnh tùy chỉnh")
    parser.add_argument("--image", type=str, default=None, help="Đường dẫn file ảnh đơn lẻ cụ thể để kiểm thử")
    parser.add_argument("--model", type=str, default=None, help="Đường dẫn tới file hoặc thư mục model MobileNetV2")
    parser.add_argument("--output", type=str, default=DEFAULT_OUTPUT_DIR, help="Thư mục xuất báo cáo và hình ảnh")

    args = parser.parse_args()

    print("\n" + "=" * 75)
    print(" KHỞI ĐỘNG HỆ THỐNG FULL E-KYC PIPELINE (MOBILENETV2 ANTI-SPOOFING)")
    print("=" * 75)

    # 1. Tải Face Detection (YOLO)
    print("[INFO] 1/6 Đang tải Face Detection (Face_Detection.pt)...")
    detector = FaceDetector()

    # 2. Tải Landmark Detection (MediaPipe)
    print("[INFO] 2/6 Đang tải Landmark Detector (face_landmarker.task)...")
    landmark_detector = LandmarkDetector()

    # 3. Tải Pose Validator
    print("[INFO] 3/6 Đang khởi tạo 3D Pose Validator...")
    pose_validator = PoseValidator()

    # 4. Tải Face Aligner & Cropper
    print("[INFO] 4/6 Đang khởi tạo Face Aligner & Cropper (224x224)...")
    aligner = FaceAligner()

    # 5. Tải MobileNetV2 Anti-Spoofing với real_threshold = args.thresh (mặc định 0.6)
    print(f"[INFO] 5/6 Đang nạp mô hình MobileNetV2 Anti-Spoofing (Ngưỡng REAL: {args.thresh})...")
    anti_spoof_mobilenet = AntiSpoofMobileNetV2(
        model_path=args.model,
        scale_factor=args.scale,
        real_threshold=args.thresh
    )

    # 6. Khởi tạo Head Movement Detector
    print("[INFO] 6/6 Đang khởi tạo Head Movement Detector...")
    head_movement_detector = HeadMovementDetector(yaw_threshold=16.0, pitch_threshold=12.0, timeout=7.0)

    models_bundle = (detector, landmark_detector, pose_validator, aligner, anti_spoof_mobilenet, head_movement_detector)
    print("\n[OK] ĐÃ TẢI HOÀN TẤT TOÀN BỘ CÁC MÔ HÌNH!\n")

    # MẶC ĐỊNH LÀ WEBCAM INTERACTIVE PIPELINE
    if args.image:
        process_single_image(
            image_path=args.image,
            output_root_dir=args.output,
            models_bundle=models_bundle,
            real_threshold=args.thresh,
            scale_factor=args.scale
        )
    elif args.batch or args.dir:
        target_dir = args.dir if args.dir else DATA_RAW_DIR
        run_batch_directory(
            data_dir=target_dir,
            output_dir=args.output,
            models_bundle=models_bundle,
            real_threshold=args.thresh,
            scale_factor=args.scale
        )
    else:
        # MẶC ĐỊNH MỞ WEBCAM INTERACTIVE eKYC
        run_webcam_pipeline(
            cam_id=args.cam,
            output_dir=args.output,
            models_bundle=models_bundle,
            real_threshold=args.thresh,
            scale_factor=args.scale,
            skip_liveness=args.static
        )


if __name__ == "__main__":
    main()
