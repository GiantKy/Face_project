"""
=============================================================================
Full E-KYC Pipeline 4 (Pipeline v4 - Interactive Capture & Live Verification)
Quy trình thực hiện:
  1. Mở Webcam: Hiển thị giao diện xem trước & canh chỉnh khuôn mặt.
  2. Chụp ảnh (Phím SPACE / 'c' hoặc Tự động khi mặt chuẩn):
     - Lưu ảnh gốc vào data_raw/<id>.jpg (đánh số thứ tự tăng dần tiếp theo: 4.jpg, 5.jpg,...).
  3. Chạy AI Model trên ảnh vừa chụp (đến hết bước Anti-Spoof):
     - Face Detection -> Landmark Detection -> Pose 3D -> Face Align & Crop 224x224 -> Anti-Spoof Model.
  4. Bắt đầu Active Liveness trên luồng Live Webcam:
     - Blink Detection: Yêu cầu người dùng chớp mắt (đo EAR).
     - Head Movement: Đưa ra thử thách quay đầu ngẫu nhiên (Trái/Phải/Ngước/Cúi).
  5. Tổng hợp toàn bộ dữ liệu & Đưa ra quyết định cuối cùng (Final eKYC Decision).
  6. Lưu toàn bộ kết quả vào output/<id>/ gồm:
     - 1_pipeline_result.jpg
     - 2_face_crop_224.jpg
     - 3_aligned_full.jpg
     - 4_report.json
  7. Cập nhật bảng tổng kết batch_summary_v4.csv và batch_summary_v4.json.

Phím tắt:
  - SPACE hoặc 'c': Chụp ảnh và bắt đầu quy trình eKYC
  - 'a': Bật/Tắt chế độ tự động chụp khi mặt đúng vị trí
  - 'r': Bắt đầu phiên eKYC mới (ảnh mới tiếp theo)
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
from enum import Enum
from pathlib import Path
import cv2
import numpy as np
from ultralytics import YOLO

# Cấu hình đường dẫn import
CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
BACKUP_DIR = os.path.dirname(CURRENT_DIR)
BASE_DIR = os.path.dirname(BACKUP_DIR)

if BACKUP_DIR not in sys.path:
    sys.path.insert(0, BACKUP_DIR)

from src.face_detection import FaceDetector
from src.landmark_detection import LandmarkDetector
from src.landmark_detection.draw_landmarks import draw_landmarks
from src.landmark_detection.utils import get_landmark_point
from src.pose_validation import PoseValidator
from src.pose_validation.draw_pose import draw_pose_info
from src.face_alignment_crop import FaceAligner
from src.head_movement import HeadMovementDetector, HeadAction, ChallengeState

DATA_RAW_DIR = os.path.join(BASE_DIR, "data_raw")
OUTPUT_DIR = os.path.join(CURRENT_DIR, "output")


# =============================================================================
# 1. HELPER FUNCTIONS: EAR & QUẢN LÝ THỨ TỰ ẢNH
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


def get_next_image_index(data_dir):
    """
    Tìm số thứ tự tiếp theo cho ảnh mới trong thư mục data_raw.
    Ví dụ: nếu đã có 0.jpg, 1.jpg, 2.jpg, 3.jpg -> trả về 4.
    """
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
# 2. ANTI-SPOOF DETECTOR CLASS
# =============================================================================
class AntiSpoofDetector:
    def __init__(self, model_version="v2"):
        candidate_files = [
            f"Anti_Spoof_{model_version}.pt",
            "Anti_Spoof_v2.pt",
            "Anti_Spoof_v1.pt",
            "Anti_Spoof_v4.pt",
            "Anti_Spoof_v3.pt",
            "Anti_Spoof.pt"
        ]

        self.model_path = None
        for filename in candidate_files:
            path = os.path.join(BASE_DIR, "models", filename)
            if os.path.exists(path):
                self.model_path = path
                break

        if self.model_path is None:
            raise FileNotFoundError("Không tìm thấy model Anti_Spoof trong thư mục models/")

        print(f"[INFO] Loading Anti-Spoof model: {self.model_path}")
        self.model = YOLO(self.model_path)
        self.classes = self.model.names
        print(f"[OK] Anti-Spoof classes: {self.classes}")

    def predict(self, frame, conf_threshold=0.3):
        results = self.model(frame, verbose=False, conf=conf_threshold)
        detections = []

        for r in results:
            if hasattr(r, 'probs') and r.probs is not None:
                probs = r.probs.data.cpu().numpy()
                top1 = int(r.probs.top1)
                score = float(probs[top1])
                label = self.classes.get(top1, f"class_{top1}").lower()
                is_real = ("real" in label)
                detections.append({
                    "bbox": (0, 0, frame.shape[1], frame.shape[0]),
                    "is_real": is_real,
                    "label": "REAL" if is_real else "SPOOF",
                    "confidence": score,
                    "raw_class": label
                })
            elif hasattr(r, 'boxes') and len(r.boxes) > 0:
                for box in r.boxes:
                    x1, y1, x2, y2 = map(int, box.xyxy[0])
                    conf = float(box.conf[0])
                    cls = int(box.cls[0])
                    label = self.classes.get(cls, f"class_{cls}").lower()
                    is_real = ("real" in label)
                    detections.append({
                        "bbox": (x1, y1, x2, y2),
                        "is_real": is_real,
                        "label": "REAL" if is_real else "SPOOF",
                        "confidence": conf,
                        "raw_class": label
                    })

        return detections


# =============================================================================
# 3. HUD & VISUALIZATION UTILITIES
# =============================================================================
def draw_ui_card(image, x, y, w, h, bg_color=(15, 15, 20), alpha=0.85):
    """Vẽ khung card bán trong suốt làm nền HUD"""
    overlay = image.copy()
    cv2.rectangle(overlay, (x, y), (x + w, y + h), bg_color, -1)
    cv2.addWeighted(overlay, alpha, image, 1 - alpha, 0, image)
    cv2.rectangle(image, (x, y), (x + w, y + h), (100, 100, 100), 1)


def draw_pipeline4_result_hud(
    image,
    img_idx,
    face_info,
    pose_info,
    pose_valid,
    anti_spoof_info,
    blink_passed,
    blink_count,
    head_movement_passed,
    head_action_name,
    final_pass,
    reasons
):
    h, w = image.shape[:2]
    vis = image.copy()

    card_w = min(480, w - 20)
    card_h = 280
    draw_ui_card(vis, 15, 15, card_w, card_h, bg_color=(15, 15, 20), alpha=0.88)

    cv2.putText(vis, f"E-KYC PIPELINE v4 REPORT (ID: {img_idx})", (25, 42),
                cv2.FONT_HERSHEY_SIMPLEX, 0.62, (0, 230, 255), 2)
    cv2.line(vis, (25, 50), (15 + card_w - 20, 50), (80, 80, 80), 1)

    # 1. Face Detection
    f_txt = f"1. Face Detection : DETECTED ({face_info['confidence']:.2f})" if face_info else "1. Face Detection : NO FACE"
    f_col = (0, 255, 0) if face_info else (0, 0, 255)
    cv2.putText(vis, f_txt, (25, 72), cv2.FONT_HERSHEY_SIMPLEX, 0.46, f_col, 1)

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
    cv2.putText(vis, p_txt, (25, 95), cv2.FONT_HERSHEY_SIMPLEX, 0.46, p_col, 1)

    # 3. Anti-Spoof
    if anti_spoof_info:
        as_lbl = anti_spoof_info["label"]
        as_conf = anti_spoof_info["confidence"]
        as_col = (0, 255, 0) if anti_spoof_info["is_real"] else (0, 0, 255)
        as_txt = f"3. Anti-Spoof    : {as_lbl} ({as_conf*100:.1f}%)"
    else:
        as_txt = "3. Anti-Spoof    : NO DATA"
        as_col = (0, 165, 255)
    cv2.putText(vis, as_txt, (25, 118), cv2.FONT_HERSHEY_SIMPLEX, 0.46, as_col, 1)

    # 4. Blink Liveness
    b_txt = f"4. Blink Liveness: PASS ({blink_count} blinks)" if blink_passed else f"4. Blink Liveness: FAIL ({blink_count} blinks)"
    b_col = (0, 255, 0) if blink_passed else (0, 0, 255)
    cv2.putText(vis, b_txt, (25, 141), cv2.FONT_HERSHEY_SIMPLEX, 0.46, b_col, 1)

    # 5. Head Movement Liveness
    hm_txt = f"5. Head Movement : PASS [{head_action_name.upper()}]" if head_movement_passed else f"5. Head Movement : FAIL [{head_action_name.upper()}]"
    hm_col = (0, 255, 0) if head_movement_passed else (0, 0, 255)
    cv2.putText(vis, hm_txt, (25, 164), cv2.FONT_HERSHEY_SIMPLEX, 0.46, hm_col, 1)

    cv2.line(vis, (25, 180), (15 + card_w - 20, 180), (80, 80, 80), 1)

    # 6. Final Decision
    verdict_text = "eKYC: APPROVED (HOP LE)" if final_pass else "eKYC: REJECTED (TU CHOI)"
    verdict_col = (0, 255, 0) if final_pass else (0, 0, 255)
    cv2.putText(vis, verdict_text, (25, 212),
                cv2.FONT_HERSHEY_SIMPLEX, 0.68, verdict_col, 2)

    if not final_pass and reasons:
        reason_str = "Ly do: " + ", ".join(reasons[:2])
        cv2.putText(vis, reason_str, (25, 240),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.44, (0, 200, 255), 1)

    return vis


# =============================================================================
# 4. PIPELINE 4 WORKFLOW STATE MACHINE
# =============================================================================
class PipelineStage(Enum):
    PREVIEW_ALIGN = 1       # Giai đoạn 1: Mở webcam, canh góc mặt & chờ chụp ảnh
    RUN_AI_STATIC = 2       # Giai đoạn 2: Chạy Face -> Landmark -> Pose -> Align/Crop -> Anti-Spoof trên ảnh chụp
    LIVE_BLINK = 3          # Giai đoạn 3: Active Liveness - Thử thách chớp mắt
    LIVE_HEAD_MOVEMENT = 4  # Giai đoạn 4: Active Liveness - Thử thách quay đầu
    FINAL_DECISION = 5      # Giai đoạn 5: Tổng hợp toàn bộ & lưu vào output/<id>/


def main_pipeline_4(cam_id=0):
    print("\n" + "=" * 75)
    print("      FULL E-KYC PIPELINE 4 (CHỤP ẢNH -> AI MODEL -> BLINK & HEAD MOVEMENT)")
    print("=" * 75)
    print(f"  * Thư mục lưu ảnh gốc : {DATA_RAW_DIR}")
    print(f"  * Thư mục lưu kết quả : {OUTPUT_DIR}")
    print("  * Điều khiển:")
    print("      [SPACE] hoặc [c]  : Chụp ảnh ngay và bắt đầu quy trình")
    print("      [a]               : Bật/Tắt chế độ tự động chụp khi mặt chuẩn")
    print("      [r]               : Khởi tạo lại phiên eKYC mới")
    print("      [q] hoặc [ESC]    : Thoát")
    print("=" * 75 + "\n")

    os.makedirs(DATA_RAW_DIR, exist_ok=True)
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    # 1. Khởi tạo Models
    print("[INFO] Đang khởi tạo các AI Models...")
    detector = FaceDetector()
    landmark_detector = LandmarkDetector()
    pose_validator = PoseValidator()
    aligner = FaceAligner()
    anti_spoof_detector = AntiSpoofDetector()
    head_movement_detector = HeadMovementDetector(yaw_threshold=16.0, pitch_threshold=12.0, timeout=7.0)
    print("[OK] Đã khởi tạo hoàn tất toàn bộ Models!\n")

    cap = cv2.VideoCapture(cam_id)
    if not cap.isOpened():
        print(f"[ERROR] Không thể mở Camera ID {cam_id}!")
        return

    # Trạng thái luồng
    stage = PipelineStage.PREVIEW_ALIGN
    auto_capture_mode = False
    consecutive_center_frames = 0

    # Dữ liệu của phiên hiện tại
    current_img_idx = get_next_image_index(DATA_RAW_DIR)
    captured_frame = None
    captured_img_path = None
    captured_result_dir = None

    # Dữ liệu tĩnh từ ảnh chụp
    primary_face = None
    landmarks_static = None
    pose_dict_static = None
    pose_valid_static = False
    face_crop_static = None
    aligned_img_static = None
    best_spoof_static = None

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
        nonlocal primary_face, landmarks_static, pose_dict_static, pose_valid_static
        nonlocal face_crop_static, aligned_img_static, best_spoof_static
        nonlocal blink_counter, blink_state, blink_passed, head_movement_passed, current_head_action, head_action_prompt
        nonlocal final_pass, reasons, final_display_img, final_record, consecutive_center_frames

        current_img_idx = get_next_image_index(DATA_RAW_DIR)
        stage = PipelineStage.PREVIEW_ALIGN
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
        # GIAI ĐOẠN 1: PREVIEW & CHỤP ẢNH
        # =====================================================================
        if stage == PipelineStage.PREVIEW_ALIGN:
            # Phát hiện vị trí mặt và góc nhìn tạm thời trên webcam
            landmarks_live = landmark_detector.detect(frame)
            pose_valid_live = False
            pose_dict_live = None

            if landmarks_live:
                pose_valid_live, _, pose_dict_live = pose_validator.validate(landmarks_live, get_landmark_point)
                display = draw_landmarks(display, landmarks_live)

            # Khung banner hướng dẫn chụp ảnh
            draw_ui_card(display, 20, 20, w - 40, 115, bg_color=(15, 15, 25), alpha=0.85)
            cv2.putText(display, f"E-KYC PIPELINE 4: CHUAN BI CHUP ANH (ID: {current_img_idx}.jpg)", (35, 48),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.62, (0, 230, 255), 2)

            is_aligned_good = (landmarks_live is not None and pose_valid_live)
            if is_aligned_good:
                align_msg = "Goc mat CHUAN! Nhan [SPACE] hoac [c] de chup anh"
                align_col = (0, 255, 0)
                consecutive_center_frames += 1
            else:
                align_msg = "Vui long nhin thang, giu mat chinh giua khung hinh..."
                align_col = (0, 200, 255)
                consecutive_center_frames = max(0, consecutive_center_frames - 1)

            cv2.putText(display, align_msg, (35, 80),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.55, align_col, 2)

            mode_str = f"Che do Auto-Capture: {'BAT (Chup sau 2s)' if auto_capture_mode else 'TAT (Nhan SPACE de chup)'}"
            cv2.putText(display, mode_str, (35, 110),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.46, (180, 180, 180), 1)

            # Tự động chụp nếu bật auto_capture_mode và giữ mặt chuẩn 25 frames
            if auto_capture_mode and consecutive_center_frames >= 25:
                trigger_capture = True
            else:
                trigger_capture = False

            if trigger_capture:
                key_trigger = ord(' ')
            else:
                key_trigger = None

        # =====================================================================
        # GIAI ĐOẠN 2: CHẠY AI MODEL TRÊN ẢNH CHỤP ĐẾN BƯỚC ANTI-SPOOF
        # =====================================================================
        elif stage == PipelineStage.RUN_AI_STATIC:
            draw_ui_card(display, 20, 20, w - 40, 90, bg_color=(15, 15, 25), alpha=0.9)
            cv2.putText(display, f"DANG CHAY AI MODEL TREN ANH ID {current_img_idx}.jpg...", (35, 55),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.65, (0, 230, 255), 2)
            cv2.putText(display, "Tien trinh: Face Detect -> Landmark -> Pose 3D -> Crop 224 -> Anti-Spoof", (35, 85),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.48, (0, 255, 0), 1)
            cv2.imshow("Full E-KYC Pipeline 4", display)
            cv2.waitKey(1)

            # 1. Lưu ảnh gốc vào data_raw/<id>.jpg
            captured_img_path = os.path.join(DATA_RAW_DIR, f"{current_img_idx}.jpg")
            cv2.imwrite(captured_img_path, captured_frame)
            print(f"\n[1. CHỤP ẢNH GỐC] Đã lưu ảnh vào: {captured_img_path}")

            # 2. Tạo thư mục output/<id>/
            captured_result_dir = os.path.join(OUTPUT_DIR, str(current_img_idx))
            os.makedirs(captured_result_dir, exist_ok=True)

            # 3. Chạy Face Detection
            faces = detector.detect(captured_frame)
            primary_face = faces[0] if len(faces) > 0 else None
            print(f"[2. Face Detection] Tìm thấy {len(faces)} khuôn mặt.")

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

            # 6. Face Alignment & 224x224 Crop
            aligned_img_static = None
            face_crop_static = None
            if landmarks_static:
                aligned_img_static = aligner.align_face(captured_frame, landmarks_static)
                aligned_lms = aligner.get_landmarks(aligned_img_static)
                if aligned_lms:
                    face_crop_static = aligner.crop_face(aligned_img_static, aligned_lms, padding=20, output_size=(224, 224))
                    print(f"[5. Face Alignment & Crop] Cắt ảnh chuẩn 224x224 thành công.")

            # 7. Chạy Anti-Spoofing Model (YOLO Detection hoạt động chính xác nhất trên ảnh nguyên khung hình)
            input_spoof = captured_frame
            spoof_res = anti_spoof_detector.predict(input_spoof, conf_threshold=0.25)
            best_spoof_static = spoof_res[0] if spoof_res else None
            if best_spoof_static:
                print(f"[6. Anti-Spoofing Model] Kết quả: {best_spoof_static['label']} ({best_spoof_static['confidence']*100:.1f}%) | Real={best_spoof_static['is_real']}")

            # Chuyển sang giai đoạn Active Liveness trên Webcam
            print("\n[INFO] Chuyển sang giai đoạn Live Active Liveness (Blink & Head Movement)...")
            stage = PipelineStage.LIVE_BLINK
            blink_counter = 0
            blink_state = False
            blink_passed = False

        # =====================================================================
        # GIAI ĐOẠN 3: ACTIVE LIVENESS - BLINK DETECTION (LIVE WEBCAM)
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
                # Chuyển sang thử thách cử động đầu
                stage = PipelineStage.LIVE_HEAD_MOVEMENT
                current_head_action = head_movement_detector.start_challenge()
                head_action_prompt = head_movement_detector.get_prompt()
                print(f"[LIVENESS 2: HEAD MOVEMENT] Thử thách: {current_head_action.value} -> {head_action_prompt}")

            # Vẽ HUD Blink
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
        # GIAI ĐOẠN 4: ACTIVE LIVENESS - HEAD MOVEMENT CHALLENGE (LIVE WEBCAM)
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
        # GIAI ĐOẠN 5: TỔNG HỢP KẾT QUẢ & LƯU VÀO OUTPUT/<ID>/
        # =====================================================================
        elif stage == PipelineStage.FINAL_DECISION:
            if final_record is None:
                # 1. Đánh giá Final Decision
                c_face = (primary_face is not None)
                c_pose = pose_valid_static
                c_spoof = (best_spoof_static is not None and best_spoof_static["is_real"])
                c_blink = blink_passed
                c_head = head_movement_passed

                reasons.clear()
                if not c_face: reasons.append("Không tìm thấy khuôn mặt trong ảnh")
                if not c_pose: reasons.append("Góc mặt ảnh chụp bị nghiêng/lệch")
                if not c_spoof: reasons.append("Phát hiện giả mạo Anti-Spoof (Fake/Spoof)")
                if not c_blink: reasons.append("Chưa hoàn thành chớp mắt (Blink)")
                if not c_head: reasons.append("Chưa hoàn thành cử động đầu (Head Movement)")

                final_pass = (c_face and c_pose and c_spoof and c_blink and c_head)

                # 2. Vẽ Dashboard kết quả lên ảnh chụp gốc
                res_img = captured_frame.copy()
                if primary_face:
                    bx1, by1, bx2, by2 = primary_face["bbox"]
                    cv2.rectangle(res_img, (bx1, by1), (bx2, by2), (0, 255, 0), 2)
                if landmarks_static:
                    res_img = draw_landmarks(res_img, landmarks_static)
                if best_spoof_static:
                    sx1, sy1, sx2, sy2 = best_spoof_static["bbox"]
                    s_col = (0, 255, 0) if best_spoof_static["is_real"] else (0, 0, 255)
                    cv2.rectangle(res_img, (sx1, sy1), (sx2, sy2), s_col, 2)
                    cv2.putText(res_img, f"{best_spoof_static['label']} {best_spoof_static['confidence']*100:.1f}%",
                                (sx1, max(25, sy1 - 10)), cv2.FONT_HERSHEY_SIMPLEX, 0.7, s_col, 2)

                final_display_img = draw_pipeline4_result_hud(
                    res_img,
                    current_img_idx,
                    primary_face,
                    pose_dict_static,
                    pose_valid_static,
                    best_spoof_static,
                    blink_passed,
                    blink_counter,
                    head_movement_passed,
                    current_head_action.value,
                    final_pass,
                    reasons
                )

                # 3. Lưu các file vào thư mục output/<id>/
                # File 1A: 1_pipeline_result.jpg (Kèm bảng điều khiển HUD Dashboard chi tiết)
                out_res_path = os.path.join(captured_result_dir, "1_pipeline_result.jpg")
                cv2.imwrite(out_res_path, final_display_img)

                # File 1B: 1_pipeline_result_clean.jpg (Ảnh kết quả sạch, giữ BBox/Landmarks/Tag nhưng BỎ ĐI BẢNG ĐIỀU KHIỂN)
                clean_img = res_img.copy()
                # Vẽ 1 badge kết quả nhỏ gọn góc trên bên phải không che mặt
                verdict_badge = "eKYC: APPROVED" if final_pass else "eKYC: REJECTED"
                badge_col = (0, 255, 0) if final_pass else (0, 0, 255)
                cv2.rectangle(clean_img, (w - 240, 15), (w - 15, 55), (15, 15, 20), -1)
                cv2.rectangle(clean_img, (w - 240, 15), (w - 15, 55), badge_col, 2)
                cv2.putText(clean_img, verdict_badge, (w - 225, 42),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.62, badge_col, 2)
                out_clean_path = os.path.join(captured_result_dir, "1_pipeline_result_clean.jpg")
                cv2.imwrite(out_clean_path, clean_img)

                # File 2: 2_face_crop_224.jpg
                if face_crop_static is not None:
                    out_crop_path = os.path.join(captured_result_dir, "2_face_crop_224.jpg")
                    cv2.imwrite(out_crop_path, face_crop_static)

                # File 3: 3_aligned_full.jpg
                if aligned_img_static is not None:
                    out_align_path = os.path.join(captured_result_dir, "3_aligned_full.jpg")
                    cv2.imwrite(out_align_path, aligned_img_static)

                # File 4: 4_report.json
                final_record = {
                    "image_id": current_img_idx,
                    "image_name": f"{current_img_idx}.jpg",
                    "raw_image_path": captured_img_path,
                    "output_folder": captured_result_dir,
                    "face_detected": primary_face is not None,
                    "face_confidence": round(primary_face["confidence"], 4) if primary_face else 0.0,
                    "face_bbox": primary_face["bbox"] if primary_face else None,
                    "pose_validation": {
                        "is_valid": bool(pose_valid_static),
                        "yaw": round(pose_dict_static["yaw"], 2) if pose_dict_static else 0.0,
                        "pitch": round(pose_dict_static["pitch"], 2) if pose_dict_static else 0.0,
                        "roll": round(pose_dict_static["roll"], 2) if pose_dict_static else 0.0,
                    },
                    "anti_spoof": {
                        "label": best_spoof_static["label"] if best_spoof_static else "NONE",
                        "is_real": bool(best_spoof_static["is_real"]) if best_spoof_static else False,
                        "confidence": round(best_spoof_static["confidence"], 4) if best_spoof_static else 0.0,
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

                # 4. Cập nhật Báo cáo tổng kết batch_summary_v4.csv / json
                batch_csv_path = os.path.join(OUTPUT_DIR, "batch_summary_v4.csv")
                file_exists = os.path.exists(batch_csv_path)
                with open(batch_csv_path, "a", newline="", encoding="utf-8") as f:
                    writer = csv.writer(f)
                    if not file_exists:
                        writer.writerow(["Image ID", "Verdict", "AntiSpoof", "Spoof Conf", "Pose Valid", "Blink", "Head Movement", "Reasons", "Output Folder"])
                    writer.writerow([
                        f"{current_img_idx}.jpg",
                        final_record["final_verdict"],
                        final_record["anti_spoof"]["label"],
                        final_record["anti_spoof"]["confidence"],
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

        # Vẽ thanh trạng thái FPS ở góc phải trên
        curr_time = time.time()
        fps = 1.0 / (curr_time - prev_fps_time) if curr_time > prev_fps_time else 0.0
        prev_fps_time = curr_time
        cv2.putText(display, f"FPS: {fps:.1f}", (w - 120, 45), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 200, 200), 1)

        cv2.imshow("Full E-KYC Pipeline 4", display)

        # Xử lý phím bấm
        key = cv2.waitKey(1) & 0xFF
        if key == 27 or key == ord('q'):
            break

        elif key == ord('r'):
            start_new_session()

        elif key == ord('a') and stage == PipelineStage.PREVIEW_ALIGN:
            auto_capture_mode = not auto_capture_mode
            print(f"[INFO] Chế độ Auto-Capture: {'BẬT' if auto_capture_mode else 'TẮT'}")

        elif (key == 32 or key == ord('c') or key_trigger == ord(' ')) and stage == PipelineStage.PREVIEW_ALIGN:
            captured_frame = frame.copy()
            stage = PipelineStage.RUN_AI_STATIC
            print(f"\n[TRIGGER] Đã kích hoạt chụp ảnh cho ID: {current_img_idx}!")

    cap.release()
    cv2.destroyAllWindows()
    print("[INFO] Đã đóng chương trình Pipeline 4.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Full E-KYC Pipeline 4 (Interactive Capture & Live Liveness)")
    parser.add_argument("--cam", type=int, default=0, help="Camera device index (mặc định 0)")
    args = parser.parse_args()

    main_pipeline_4(cam_id=args.cam)
