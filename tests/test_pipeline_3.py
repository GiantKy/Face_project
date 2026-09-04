"""
=============================================================================
Full E-KYC Pipeline 3 (Pipeline v3 - Đầy đủ toàn diện nhất)
Quy trình eKYC chuẩn Ngân hàng:
  1. Face Detection (YOLO Face Detection)
  2. Landmark Detection (MediaPipe 478 Keypoints)
  3. Pose Estimation & Validation (3D Head Pose: Yaw, Pitch, Roll)
  4. Face Alignment & 224x224 Face Crop (Căn chỉnh xoay ngang và cắt mặt)
  5. Anti-Spoofing Model (YOLO Model: Phân loại Real vs Fake/Spoof)
  6. Blink Detection (EAR - Eye Aspect Ratio: Đếm số lần chớp mắt)
  7. Head Movement Detection (Thử thách cử động đầu: Quay trái/phải)
  8. Final eKYC Decision Engine (Đánh giá tổng hợp toàn diện các tiêu chí)

Chế độ hoạt động:
  - Batch Mode (mặc định): Tự động xử lý toàn bộ ảnh trong data_raw/,
    phân tích chi tiết từng bước, phân loại hướng đầu tĩnh, lưu vào output/
  - Realtime Webcam Mode (--webcam): Trải nghiệm phiên eKYC tương tác thực tế
    qua 4 bước: [1] Căn giữa mặt -> [2] Anti-Spoof -> [3] Chớp mắt -> [4] Cử động đầu
=============================================================================
"""

import sys
import os
import argparse
import time
import math
import json
import csv
from enum import Enum
from pathlib import Path
import cv2
import numpy as np
from ultralytics import YOLO

# Cấu hình đường dẫn import
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

IMAGE_EXTENSIONS = {'.jpg', '.jpeg', '.png', '.bmp', '.webp', '.tiff'}


# =============================================================================
# 1. HELPER FUNCTIONS: EAR & MAR
# =============================================================================
def calc_dist(p1, p2):
    """Tính khoảng cách Euclidean giữa 2 điểm (x, y)"""
    return math.hypot(p1[0] - p2[0], p1[1] - p2[1])


def compute_eye_aspect_ratio(landmarks):
    """
    Tính EAR (Eye Aspect Ratio) từ MediaPipe 478 landmarks.
    Left Eye: 33, 133, 160, 158, 144, 153
    Right Eye: 362, 263, 385, 387, 380, 373
    """
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
    """Tính MAR (Mouth Aspect Ratio) từ landmarks"""
    if not landmarks or len(landmarks) < 468:
        return 0.0

    m_height = calc_dist(landmarks[13], landmarks[14])
    m_width = calc_dist(landmarks[61], landmarks[291])
    return (m_height / m_width) if m_width > 0 else 0.0


# =============================================================================
# 2. ANTI-SPOOF DETECTOR CLASS
# =============================================================================
class AntiSpoofDetector:
    def __init__(self, model_version="YOLO"):
        candidate_files = [
            "Anti_Spoof_YOLO.pt",
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
            import glob
            pts = glob.glob(os.path.join(BASE_DIR, "models", "*Anti_Spoof*.pt"))
            if pts:
                self.model_path = pts[0]
            else:
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
def draw_ui_card(image, x, y, w, h, bg_color=(15, 15, 20), alpha=0.82):
    overlay = image.copy()
    cv2.rectangle(overlay, (x, y), (x + w, y + h), bg_color, -1)
    cv2.addWeighted(overlay, alpha, image, 1 - alpha, 0, image)
    cv2.rectangle(image, (x, y), (x + w, y + h), (100, 100, 100), 1)


def draw_pipeline3_hud(
    image,
    face_info,
    pose_info,
    pose_valid,
    anti_spoof_info,
    ear_val,
    head_movement_info,
    final_pass,
    reasons
):
    h, w = image.shape[:2]
    vis = image.copy()

    card_w = min(460, w - 20)
    card_h = 280
    draw_ui_card(vis, 15, 15, card_w, card_h, bg_color=(15, 15, 20), alpha=0.85)

    cv2.putText(vis, "FULL E-KYC PIPELINE v3 DASHBOARD", (25, 42),
                cv2.FONT_HERSHEY_SIMPLEX, 0.62, (0, 230, 255), 2)
    cv2.line(vis, (25, 52), (15 + card_w - 20, 52), (80, 80, 80), 1)

    # 1. Face Detection
    if face_info:
        f_conf = face_info["confidence"]
        f_txt = f"1. Face Detection : DETECTED ({f_conf:.2f})"
        f_col = (0, 255, 0)
    else:
        f_txt = "1. Face Detection : NO FACE"
        f_col = (0, 0, 255)
    cv2.putText(vis, f_txt, (25, 75), cv2.FONT_HERSHEY_SIMPLEX, 0.48, f_col, 1)

    # 2. Head Pose 3D
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
    cv2.putText(vis, p_txt, (25, 100), cv2.FONT_HERSHEY_SIMPLEX, 0.48, p_col, 1)

    # 3. Anti-Spoof
    if anti_spoof_info:
        as_lbl = anti_spoof_info["label"]
        as_conf = anti_spoof_info["confidence"]
        as_col = (0, 255, 0) if anti_spoof_info["is_real"] else (0, 0, 255)
        as_txt = f"3. Anti-Spoof    : {as_lbl} ({as_conf*100:.1f}%)"
    else:
        as_txt = "3. Anti-Spoof    : NO DATA"
        as_col = (0, 165, 255)
    cv2.putText(vis, as_txt, (25, 125), cv2.FONT_HERSHEY_SIMPLEX, 0.48, as_col, 1)

    # 4. Blink / Eye EAR
    eye_stat = "OPEN" if ear_val >= 0.20 else "CLOSED"
    l_txt = f"4. Blink / EAR   : {ear_val:.2f} ({eye_stat})"
    l_col = (0, 255, 0) if ear_val >= 0.20 else (0, 165, 255)
    cv2.putText(vis, l_txt, (25, 150), cv2.FONT_HERSHEY_SIMPLEX, 0.48, l_col, 1)

    # 5. Head Movement
    if head_movement_info:
        hm_action = head_movement_info.get("detected_action", "none")
        hm_passed = head_movement_info.get("passed", False)
        hm_txt = f"5. Head Movement : {hm_action.upper()}"
        hm_col = (0, 255, 0) if hm_passed else (0, 200, 255)
    else:
        hm_txt = "5. Head Movement : NONE"
        hm_col = (180, 180, 180)
    cv2.putText(vis, hm_txt, (25, 175), cv2.FONT_HERSHEY_SIMPLEX, 0.48, hm_col, 1)

    cv2.line(vis, (25, 192), (15 + card_w - 20, 192), (80, 80, 80), 1)

    # 6. Final Decision Banner
    verdict_text = "eKYC: APPROVED (HOP LE)" if final_pass else "eKYC: REJECTED (TU CHOI)"
    verdict_col = (0, 255, 0) if final_pass else (0, 0, 255)
    cv2.putText(vis, verdict_text, (25, 222),
                cv2.FONT_HERSHEY_SIMPLEX, 0.68, verdict_col, 2)

    if not final_pass and reasons:
        reason_str = "Ly do: " + ", ".join(reasons[:2])
        cv2.putText(vis, reason_str, (25, 250),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 200, 255), 1)

    return vis


# =============================================================================
# 4. PROCESS SINGLE IMAGE (BATCH PIPELINE)
# =============================================================================
def process_single_image(
    image_path: str,
    output_root_dir: str,
    models_bundle: tuple
) -> dict:
    detector, landmark_detector, pose_validator, aligner, anti_spoof_detector, head_movement_detector = models_bundle

    filename = os.path.basename(image_path)
    stem_name = os.path.splitext(filename)[0]

    img_output_dir = os.path.join(output_root_dir, stem_name)
    os.makedirs(img_output_dir, exist_ok=True)

    print(f"\n{'='*65}")
    print(f"  [PIPELINE v3] Đang xử lý file: {filename}")
    print(f"  -> Thư mục lưu: {img_output_dir}")
    print(f"{'='*65}")

    image = cv2.imread(image_path)
    if image is None:
        print(f"[ERROR] Không đọc được file ảnh: {image_path}")
        return None

    start_time = time.perf_counter()

    # 1. Face Detection
    faces = detector.detect(image)
    primary_face = faces[0] if len(faces) > 0 else None
    print(f"[1. Face Detection] Tìm thấy {len(faces)} khuôn mặt.")

    # 2. Landmarks
    landmarks = landmark_detector.detect(image)
    print(f"[2. Landmarks] Trích xuất được {len(landmarks) if landmarks else 0} điểm landmarks.")

    # 3. 3D Pose
    pose_valid = False
    pose_dict = None
    if landmarks:
        pose_valid, _, pose_dict = pose_validator.validate(landmarks, get_landmark_point)
        if pose_dict:
            print(f"[3. Head Pose] Y={pose_dict['yaw']:+.1f}° | P={pose_dict['pitch']:+.1f}° | R={pose_dict['roll']:+.1f}° -> {'PASS' if pose_valid else 'FAIL'}")

    # 4. Face Alignment & Crop 224x224
    aligned_img = None
    face_crop = None
    if landmarks:
        aligned_img = aligner.align_face(image, landmarks)
        aligned_lms = aligner.get_landmarks(aligned_img)
        if aligned_lms:
            face_crop = aligner.crop_face(aligned_img, aligned_lms, padding=20, output_size=(224, 224))
            print(f"[4. Alignment & Crop] Crop 224x224 thành công.")

    # 5. Anti-Spoof (YOLO Detection hoạt động chính xác nhất trên ảnh nguyên khung hình)
    input_for_spoof = image
    spoof_results = anti_spoof_detector.predict(input_for_spoof, conf_threshold=0.25)
    best_spoof_info = spoof_results[0] if spoof_results else None
    if best_spoof_info:
        print(f"[5. Anti-Spoof] {best_spoof_info['label']} ({best_spoof_info['confidence']*100:.1f}%) | Real={best_spoof_info['is_real']}")

    # 6. Eye EAR & Liveness
    ear_l, ear_r, ear_avg = compute_eye_aspect_ratio(landmarks) if landmarks else (0.0, 0.0, 0.0)
    mar_val = compute_mouth_aspect_ratio(landmarks) if landmarks else 0.0
    eye_open = (ear_avg >= 0.20)

    # 7. Head Movement / Pose Orientation Classification
    static_pose_info = head_movement_detector.classify_static_pose(pose_dict)
    print(f"[7. Head Movement] Hướng đầu: {static_pose_info['dominant_direction']} ({static_pose_info['detected_action']})")

    # 8. Final Decision Logic
    c1 = (primary_face is not None)
    c2 = pose_valid
    c3 = (best_spoof_info is not None and best_spoof_info["is_real"])
    c4 = eye_open
    final_pass = c1 and c2 and c3 and c4

    reasons = []
    if not c1: reasons.append("Không tìm thấy khuôn mặt")
    if not c2: reasons.append("Góc mặt lệch (Pose)")
    if not c3: reasons.append("Nghi vấn giả mạo (Spoof)")
    if not c4: reasons.append("Mắt đang nhắm")

    elapsed_ms = (time.perf_counter() - start_time) * 1000
    print(f"[8. Final Verdict] {'APPROVED (ĐẠT)' if final_pass else 'REJECTED (TỪ CHỐI)'} | Lý do: {', '.join(reasons) if reasons else 'None'}")

    # Vẽ kết quả HUD
    display = image.copy()
    if primary_face:
        bx1, by1, bx2, by2 = primary_face["bbox"]
        cv2.rectangle(display, (bx1, by1), (bx2, by2), (0, 255, 0), 2)
    if landmarks:
        display = draw_landmarks(display, landmarks)
    if best_spoof_info:
        sx1, sy1, sx2, sy2 = best_spoof_info["bbox"]
        s_col = (0, 255, 0) if best_spoof_info["is_real"] else (0, 0, 255)
        cv2.rectangle(display, (sx1, sy1), (sx2, sy2), s_col, 2)
        tag = f"{best_spoof_info['label']} {best_spoof_info['confidence']*100:.1f}%"
        cv2.putText(display, tag, (sx1, max(25, sy1 - 10)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, s_col, 2)

    # Lưu ảnh kết quả sạch (Face Bbox + Landmarks + AntiSpoof Tag nhưng KHÔNG có bảng điều khiển)
    clean_display = display.copy()
    verdict_badge = "eKYC: APPROVED" if final_pass else "eKYC: REJECTED"
    badge_col = (0, 255, 0) if final_pass else (0, 0, 255)
    img_h, img_w = clean_display.shape[:2]
    cv2.rectangle(clean_display, (img_w - 240, 15), (img_w - 15, 55), (15, 15, 20), -1)
    cv2.rectangle(clean_display, (img_w - 240, 15), (img_w - 15, 55), badge_col, 2)
    cv2.putText(clean_display, verdict_badge, (img_w - 225, 42),
                cv2.FONT_HERSHEY_SIMPLEX, 0.62, badge_col, 2)
    out_clean_img = os.path.join(img_output_dir, "1_pipeline3_result_clean.jpg")
    cv2.imwrite(out_clean_img, clean_display)

    display = draw_pipeline3_hud(
        display,
        primary_face,
        pose_dict,
        pose_valid,
        best_spoof_info,
        ear_avg,
        static_pose_info,
        final_pass,
        reasons
    )

    # Lưu các file kết quả
    # File 1A: 1_pipeline3_result.jpg (kèm bảng điều khiển HUD)
    out_result_img = os.path.join(img_output_dir, "1_pipeline3_result.jpg")
    cv2.imwrite(out_result_img, display)

    if face_crop is not None:
        out_crop_img = os.path.join(img_output_dir, "2_face_crop_224.jpg")
        cv2.imwrite(out_crop_img, face_crop)

    if aligned_img is not None:
        out_aligned_img = os.path.join(img_output_dir, "3_aligned_full.jpg")
        cv2.imwrite(out_aligned_img, aligned_img)

    record = {
        "image_name": filename,
        "output_folder": img_output_dir,
        "time_ms": round(elapsed_ms, 2),
        "face_detected": primary_face is not None,
        "face_confidence": round(primary_face["confidence"], 4) if primary_face else 0.0,
        "face_bbox": primary_face["bbox"] if primary_face else None,
        "pose_validation": {
            "is_valid": pose_valid,
            "yaw": round(pose_dict["yaw"], 2) if pose_dict else 0.0,
            "pitch": round(pose_dict["pitch"], 2) if pose_dict else 0.0,
            "roll": round(pose_dict["roll"], 2) if pose_dict else 0.0,
        },
        "anti_spoof": {
            "label": best_spoof_info["label"] if best_spoof_info else "NONE",
            "is_real": best_spoof_info["is_real"] if best_spoof_info else False,
            "confidence": round(best_spoof_info["confidence"], 4) if best_spoof_info else 0.0,
        },
        "liveness": {
            "ear": round(ear_avg, 3),
            "mar": round(mar_val, 3),
            "eye_open": eye_open,
        },
        "head_movement": {
            "detected_action": static_pose_info["detected_action"],
            "dominant_direction": static_pose_info["dominant_direction"],
            "is_straight": static_pose_info["is_straight"]
        },
        "final_verdict": "APPROVED" if final_pass else "REJECTED",
        "reasons": reasons
    }

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

    with open(os.path.join(img_output_dir, "4_report.json"), "w", encoding="utf-8") as f:
        json.dump(record, f, ensure_ascii=False, indent=2, default=json_serialize_helper)

    print(f"  [OK] Đã lưu báo cáo hoàn chỉnh tại: {img_output_dir}")
    return record


# =============================================================================
# 5. BATCH EXECUTION
# =============================================================================
def run_pipeline_batch(input_path=None, output_path=None):
    if input_path is None:
        input_path = os.path.join(BASE_DIR, "data_raw")
        if not os.path.exists(input_path):
            input_path = os.path.join(BASE_DIR, "data", "raw")

    if output_path is None:
        output_path = os.path.join(CURRENT_DIR, "output")

    os.makedirs(output_path, exist_ok=True)

    if os.path.isfile(input_path):
        image_files = [input_path]
    elif os.path.isdir(input_path):
        image_files = [
            os.path.join(input_path, f)
            for f in sorted(os.listdir(input_path))
            if os.path.splitext(f)[1].lower() in IMAGE_EXTENSIONS
        ]
    else:
        print(f"[ERROR] Không tìm thấy đường dẫn: {input_path}")
        sys.exit(1)

    if not image_files:
        print(f"[WARNING] Không tìm thấy file ảnh nào trong: {input_path}")
        return

    print("\n" + "=" * 70)
    print("           FULL E-KYC PIPELINE v3 - BATCH PROCESSING")
    print(f"  * Thư mục đầu vào : {input_path}")
    print(f"  * Thư mục output  : {output_path}")
    print(f"  * Số lượng ảnh    : {len(image_files)}")
    print("=" * 70)

    print("\n[INFO] Đang khởi tạo tất cả các models...")
    detector = FaceDetector()
    landmark_detector = LandmarkDetector()
    pose_validator = PoseValidator()
    aligner = FaceAligner()
    anti_spoof_detector = AntiSpoofDetector()
    head_movement_detector = HeadMovementDetector()
    models_bundle = (detector, landmark_detector, pose_validator, aligner, anti_spoof_detector, head_movement_detector)

    all_records = []
    total_time = 0.0

    for idx, img_file in enumerate(image_files, 1):
        print(f"\n>>> Đang xử lý [{idx}/{len(image_files)}]: {os.path.basename(img_file)}")
        rec = process_single_image(img_file, output_path, models_bundle)
        if rec:
            all_records.append(rec)
            total_time += rec["time_ms"]

    # Xuất báo cáo tổng kết Batch
    batch_json_path = os.path.join(output_path, "batch_summary_v3.json")
    with open(batch_json_path, "w", encoding="utf-8") as f:
        json.dump(all_records, f, ensure_ascii=False, indent=2, default=lambda o: bool(o) if isinstance(o, (np.bool_, bool)) else (float(o) if isinstance(o, (np.floating, float)) else (int(o) if isinstance(o, (np.integer, int)) else str(o))))

    batch_csv_path = os.path.join(output_path, "batch_summary_v3.csv")
    with open(batch_csv_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["Image", "Verdict", "AntiSpoof", "Spoof Conf", "Pose Valid", "Head Direction", "Yaw", "Pitch", "Roll", "EAR", "Time (ms)", "Output Folder"])
        for r in all_records:
            writer.writerow([
                r["image_name"],
                r["final_verdict"],
                r["anti_spoof"]["label"],
                r["anti_spoof"]["confidence"],
                "PASS" if r["pose_validation"]["is_valid"] else "FAIL",
                r["head_movement"]["dominant_direction"],
                r["pose_validation"]["yaw"],
                r["pose_validation"]["pitch"],
                r["pose_validation"]["roll"],
                r["liveness"]["ear"],
                r["time_ms"],
                r["output_folder"]
            ])

    print("\n" + "=" * 70)
    print(f"  [HOÀN TẤT BATCH PIPELINE v3] Đã xử lý {len(all_records)}/{len(image_files)} ảnh")
    print(f"  * Tổng thời gian   : {total_time:.1f} ms (TB: {total_time/max(1, len(all_records)):.1f} ms/ảnh)")
    print(f"  * Báo cáo CSV      : {batch_csv_path}")
    print(f"  * Báo cáo JSON     : {batch_json_path}")
    print("=" * 70 + "\n")


# =============================================================================
# 6. INTERACTIVE WEBCAM REAL-TIME MODE (E-KYC ACTIVE SESSION)
# =============================================================================
class EKYCStep(Enum):
    ALIGN_FACE = 1        # Bước 1: Canh giữa khuôn mặt
    CHECK_SPOOF = 2       # Bước 2: Kiểm tra chống giả mạo
    BLINK_CHALLENGE = 3   # Bước 3: Thử thách chớp mắt
    HEAD_CHALLENGE = 4    # Bước 4: Thử thách cử động đầu (quay trái/phải)
    SUCCESS = 5           # Bước 5: eKYC thành công (Approved)
    FAILED = 6            # Bước 6: eKYC thất bại (Rejected/Timeout)


def run_pipeline_webcam(cam_id=0):
    print("\n" + "=" * 70)
    print("     FULL E-KYC PIPELINE v3 - INTERACTIVE WEBCAM ACTIVE SESSION")
    print("  Quy trình xác thực 4 bước:")
    print("    [1] Canh giữa mặt -> [2] Anti-Spoof -> [3] Chớp mắt -> [4] Cử động đầu")
    print("  * Nhấn 'r': Bắt đầu phiên eKYC mới")
    print("  * Nhấn 's': Lưu ảnh chụp màn hình")
    print("  * Nhấn 'q' hoặc ESC: Thoát")
    print("=" * 70)

    detector = FaceDetector()
    landmark_detector = LandmarkDetector()
    pose_validator = PoseValidator()
    anti_spoof_detector = AntiSpoofDetector()
    head_movement_detector = HeadMovementDetector(yaw_threshold=16.0, pitch_threshold=12.0, timeout=7.0)

    cap = cv2.VideoCapture(cam_id)
    if not cap.isOpened():
        print(f"[ERROR] Không thể mở camera ID {cam_id}!")
        sys.exit(1)

    output_dir = os.path.join(CURRENT_DIR, "output")
    os.makedirs(output_dir, exist_ok=True)

    # Trạng thái phiên eKYC
    current_step = EKYCStep.ALIGN_FACE
    step_start_time = time.time()
    session_timeout = 25.0
    session_start_time = time.time()

    # Bộ đếm
    blink_counter = 0
    blink_state = False
    consecutive_real_frames = 0
    consecutive_center_frames = 0

    current_head_action = None
    prev_time = time.time()

    def reset_session():
        nonlocal current_step, step_start_time, session_start_time
        nonlocal blink_counter, blink_state, consecutive_real_frames, consecutive_center_frames, current_head_action
        current_step = EKYCStep.ALIGN_FACE
        step_start_time = time.time()
        session_start_time = time.time()
        blink_counter = 0
        blink_state = False
        consecutive_real_frames = 0
        consecutive_center_frames = 0
        head_movement_detector.reset()
        current_head_action = None
        print("\n[INFO] Đã khởi tạo lại phiên xác thực eKYC mới!")

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        frame = cv2.flip(frame, 1)
        h, w = frame.shape[:2]
        display = frame.copy()

        # 1. Detection & Extraction
        faces = detector.detect(frame)
        primary_face = faces[0] if faces else None
        landmarks = landmark_detector.detect(frame)

        pose_valid = False
        pose_dict = None
        if landmarks:
            pose_valid, _, pose_dict = pose_validator.validate(landmarks, get_landmark_point)
            display = draw_landmarks(display, landmarks)

        ear_l, ear_r, ear_avg = compute_eye_aspect_ratio(landmarks) if landmarks else (0.0, 0.0, 0.0)

        # 2. Xử lý Chớp mắt (Blink Logic)
        if ear_avg > 0.05 and ear_avg < 0.18:
            if not blink_state:
                blink_state = True
        elif ear_avg >= 0.22:
            if blink_state:
                blink_counter += 1
                blink_state = False

        # 3. Anti-Spoof
        spoof_results = anti_spoof_detector.predict(frame, conf_threshold=0.35)
        best_spoof = spoof_results[0] if spoof_results else None
        is_real = (best_spoof is not None and best_spoof["is_real"])

        if best_spoof:
            sx1, sy1, sx2, sy2 = best_spoof["bbox"]
            s_col = (0, 255, 0) if is_real else (0, 0, 255)
            cv2.rectangle(display, (sx1, sy1), (sx2, sy2), s_col, 2)

        # 4. State Machine quản lý từng bước eKYC
        step_prompt = ""
        step_color = (0, 255, 255)
        progress_val = 0.0

        if current_step == EKYCStep.ALIGN_FACE:
            step_prompt = "BUOC 1/4: VUI LONG NHIN THANG VAO CAMERA"
            if primary_face and pose_valid:
                consecutive_center_frames += 1
                progress_val = min(1.0, consecutive_center_frames / 15.0)
                if consecutive_center_frames >= 15:
                    current_step = EKYCStep.CHECK_SPOOF
                    step_start_time = time.time()
            else:
                consecutive_center_frames = max(0, consecutive_center_frames - 1)

        elif current_step == EKYCStep.CHECK_SPOOF:
            step_prompt = "BUOC 2/4: DANG KIEM TRA CHONG GIA MAO (LIVENESS)..."
            if is_real:
                consecutive_real_frames += 1
                progress_val = min(1.0, consecutive_real_frames / 12.0)
                if consecutive_real_frames >= 12:
                    current_step = EKYCStep.BLINK_CHALLENGE
                    step_start_time = time.time()
                    blink_counter = 0
            else:
                consecutive_real_frames = max(0, consecutive_real_frames - 1)

        elif current_step == EKYCStep.BLINK_CHALLENGE:
            step_prompt = "BUOC 3/4: VUI LONG CHOP MAT (BLINK EYES)"
            progress_val = 1.0 if blink_counter >= 1 else (0.5 if blink_state else 0.0)
            if blink_counter >= 1:
                current_step = EKYCStep.HEAD_CHALLENGE
                current_head_action = head_movement_detector.start_challenge()
                step_start_time = time.time()

        elif current_step == EKYCStep.HEAD_CHALLENGE:
            hm_status = head_movement_detector.update(pose_dict)
            action_prompt = hm_status.get("prompt", "")
            time_left = hm_status.get("time_left", 0.0)
            progress_val = hm_status.get("progress", 0.0)
            step_prompt = f"BUOC 4/4: {action_prompt.upper()} ({time_left:.1f}s)"

            if hm_status["passed"]:
                current_step = EKYCStep.SUCCESS
                step_start_time = time.time()
                # Lưu snapshot thành công
                success_snap = os.path.join(output_dir, f"ekyc_success_{int(time.time())}.jpg")
                cv2.imwrite(success_snap, display)
                print(f"\n[XAC THUC THANH CONG] Đã lưu ảnh minh chứng tại: {success_snap}")
            elif hm_status["state"] == "FAILED":
                current_step = EKYCStep.FAILED
                step_start_time = time.time()

        elif current_step == EKYCStep.SUCCESS:
            step_prompt = "XAC THUC eKYC THANH CONG (APPROVED)!"
            step_color = (0, 255, 0)
            progress_val = 1.0

        elif current_step == EKYCStep.FAILED:
            step_prompt = "XAC THUC THAT BAI (REJECTED)! Nhan 'r' de thu lai"
            step_color = (0, 0, 255)
            progress_val = 0.0

        # 5. Vẽ Session Dashboard Banner
        draw_ui_card(display, 20, 20, w - 40, 100, bg_color=(20, 20, 25), alpha=0.85)

        cv2.putText(display, "E-KYC ACTIVE LIVENESS SESSION (Pipeline v3)", (35, 48),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.65, (0, 230, 255), 2)

        cv2.putText(display, step_prompt, (35, 80),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.62, step_color, 2)

        # Progress bar cho session
        bar_x = 35
        bar_y = 95
        bar_w = w - 110
        bar_h = 12
        cv2.rectangle(display, (bar_x, bar_y), (bar_x + bar_w, bar_y + bar_h), (50, 50, 50), -1)
        fill_w = int(bar_w * progress_val)
        if fill_w > 0:
            cv2.rectangle(display, (bar_x, bar_y), (bar_x + fill_w, bar_y + bar_h), step_color, -1)
        cv2.rectangle(display, (bar_x, bar_y), (bar_x + bar_w, bar_y + bar_h), (120, 120, 120), 1)

        # HUD trạng thái chi tiết góc dưới
        draw_ui_card(display, 20, h - 80, w - 40, 60, bg_color=(15, 15, 20), alpha=0.8)
        info_l1 = f"Pose: Y:{pose_dict['yaw']:+.1f} P:{pose_dict['pitch']:+.1f} | EAR:{ear_avg:.2f} | Blinks:{blink_counter}" if pose_dict else "Pose: No Face"
        info_l2 = f"AntiSpoof: {best_spoof['label'] if best_spoof else 'None'} ({best_spoof['confidence']*100:.0f}%) | [r]: Reset | [s]: Snap | [q]: Exit" if best_spoof else "[r]: Reset | [s]: Snap | [q]: Exit"
        cv2.putText(display, info_l1, (35, h - 55), cv2.FONT_HERSHEY_SIMPLEX, 0.48, (200, 200, 200), 1)
        cv2.putText(display, info_l2, (35, h - 32), cv2.FONT_HERSHEY_SIMPLEX, 0.48, (0, 220, 255), 1)

        curr_time = time.time()
        fps = 1.0 / (curr_time - prev_time) if curr_time > prev_time else 0.0
        prev_time = curr_time
        cv2.putText(display, f"FPS: {fps:.1f}", (w - 120, 48), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (200, 200, 200), 1)

        cv2.imshow("Full E-KYC Pipeline v3 (Active Liveness)", display)

        key = cv2.waitKey(1) & 0xFF
        if key == 27 or key == ord('q'):
            break
        elif key == ord('r'):
            reset_session()
        elif key == ord('s'):
            snap_path = os.path.join(output_dir, f"snapshot_v3_{int(time.time())}.jpg")
            cv2.imwrite(snap_path, display)
            print(f"[SAVED SNAPSHOT] {snap_path}")

    cap.release()
    cv2.destroyAllWindows()
    print("[INFO] Đã dừng Webcam Pipeline v3.")


# =============================================================================
# MAIN ENTRY POINT
# =============================================================================
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Full E-KYC Pipeline 3 (Batch & Realtime)")
    parser.add_argument("--input", "-i", type=str, default=None, help="Thư mục ảnh đầu vào hoặc file ảnh (mặc định: data_raw)")
    parser.add_argument("--output", "-o", type=str, default=None, help="Thư mục lưu kết quả (mặc định: backup/tests/output)")
    parser.add_argument("--image", type=str, default=None, help="Đường dẫn 1 file ảnh cụ thể")
    parser.add_argument("--webcam", action="store_true", help="Chạy chế độ Webcam Real-Time tương tác 4 bước")
    parser.add_argument("--cam", type=int, default=0, help="Camera device index (mặc định 0)")
    args = parser.parse_args()

    if args.webcam:
        run_pipeline_webcam(cam_id=args.cam)
    else:
        target_input = args.image if args.image is not None else args.input
        run_pipeline_batch(input_path=target_input, output_path=args.output)
