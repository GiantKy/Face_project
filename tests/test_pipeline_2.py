"""
=============================================================================
Full E-KYC Pipeline 2 (Pipeline v2 - Toàn diện)
Quy trình hoàn chỉnh:
  1. Face Detection (YOLO Face Detection)
  2. Landmark Detection (MediaPipe 478 Keypoints)
  3. Pose Validation (3D Head Pose: Yaw, Pitch, Roll)
  4. Face Alignment & Face Crop (224x224 chuẩn hóa)
  5. Anti-Spoofing Detection (YOLO Anti-Spoof Model: Real vs Fake/Spoof)
  6. Liveness & Facial Actions (EAR - Mắt mở/nhắm/chớp, MAR - Miệng)
  7. Final eKYC Decision Engine (Đánh giá tổng hợp tiêu chí eKYC)

Tính năng nâng cấp:
  - Tự động duyệt qua TẤT CẢ các file ảnh trong thư mục data_raw/
  - Tự động tạo thư mục con riêng biệt cho từng ảnh trong output/<tên_ảnh>/
    để lưu chi tiết từng bước mà không bị nhầm lẫn kết quả.
  - Xuất bảng tổng kết kết quả toàn bộ ảnh ra file JSON và CSV.
  - Hỗ trợ chế độ Webcam Real-Time (--webcam)
=============================================================================
"""

import sys
import os
import argparse
import time
import math
import json
import csv
from pathlib import Path
import cv2
import numpy as np
from ultralytics import YOLO

# Trỏ import vào backup/src
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

IMAGE_EXTENSIONS = {'.jpg', '.jpeg', '.png', '.bmp', '.webp', '.tiff'}


# =============================================================================
# 1. HELPER FUNCTIONS: EAR (Eye Aspect Ratio) & MAR (Mouth Aspect Ratio)
# =============================================================================
def calc_dist(p1, p2):
    """Tính khoảng cách Euclidean giữa 2 điểm (x, y)"""
    return math.hypot(p1[0] - p2[0], p1[1] - p2[1])


def compute_eye_aspect_ratio(landmarks):
    """
    Tính EAR cho cả 2 mắt từ MediaPipe 478 landmarks.
    Left Eye: 33 (outer), 133 (inner), 160, 158 (top), 144, 153 (bottom)
    Right Eye: 362 (inner), 263 (outer), 385, 387 (top), 380, 373 (bottom)
    """
    if len(landmarks) < 468:
        return 0.0, 0.0, 0.0

    # Left eye
    l_top = (calc_dist(landmarks[160], landmarks[144]) + calc_dist(landmarks[158], landmarks[153])) / 2.0
    l_width = calc_dist(landmarks[33], landmarks[133])
    ear_left = (l_top / l_width) if l_width > 0 else 0.0

    # Right eye
    r_top = (calc_dist(landmarks[385], landmarks[380]) + calc_dist(landmarks[387], landmarks[373])) / 2.0
    r_width = calc_dist(landmarks[362], landmarks[263])
    ear_right = (r_top / r_width) if r_width > 0 else 0.0

    ear_avg = (ear_left + ear_right) / 2.0
    return ear_left, ear_right, ear_avg


def compute_mouth_aspect_ratio(landmarks):
    """
    Tính MAR cho miệng từ MediaPipe landmarks.
    Lips vertical: 13 (top), 14 (bottom)
    Lips horizontal: 61 (left corner), 291 (right corner)
    """
    if len(landmarks) < 468:
        return 0.0

    m_height = calc_dist(landmarks[13], landmarks[14])
    m_width = calc_dist(landmarks[61], landmarks[291])
    mar = (m_height / m_width) if m_width > 0 else 0.0
    return mar


# =============================================================================
# 2. ANTI-SPOOF DETECTOR CLASS
# =============================================================================
class AntiSpoofDetector:
    def __init__(self, model_version="v4"):
        candidate_files = [
            f"Anti_Spoof_{model_version}.pt",
            "Anti_Spoof_v4.pt",
            "Anti_Spoof_v3.pt",
            "Anti_Spoof_v2.pt",
            "Anti_Spoof_v1.pt",
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
        """
        Dự đoán trạng thái Real / Spoof trên frame hoặc face image.
        Returns: list of dict {'bbox', 'is_real', 'label', 'conf'}
        """
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
# 3. DRAWING & HUD UTILITIES
# =============================================================================
def draw_ui_card(image, x, y, w, h, bg_color=(20, 20, 20), alpha=0.75):
    """Vẽ khung card bán trong suốt làm nền cho HUD text"""
    overlay = image.copy()
    cv2.rectangle(overlay, (x, y), (x + w, y + h), bg_color, -1)
    cv2.addWeighted(overlay, alpha, image, 1 - alpha, 0, image)
    cv2.rectangle(image, (x, y), (x + w, y + h), (100, 100, 100), 1)


def draw_ekyc_hud(
    image,
    face_info,
    pose_info,
    pose_valid,
    anti_spoof_info,
    ear_val,
    mar_val,
    final_pass,
    reasons
):
    """Vẽ giao diện HUD thông tin chi tiết eKYC lên ảnh kết quả"""
    h, w = image.shape[:2]
    vis = image.copy()

    card_w = min(420, w - 20)
    card_h = 240
    draw_ui_card(vis, 15, 15, card_w, card_h, bg_color=(15, 15, 20), alpha=0.85)

    cv2.putText(vis, "E-KYC PIPELINE v2 DASHBOARD", (25, 42),
                cv2.FONT_HERSHEY_SIMPLEX, 0.65, (0, 220, 255), 2)
    cv2.line(vis, (25, 52), (15 + card_w - 20, 52), (80, 80, 80), 1)

    # 1. Face Detection status
    if face_info:
        f_conf = face_info["confidence"]
        f_txt = f"1. Face Detection: DETECTED ({f_conf:.2f})"
        f_col = (0, 255, 0)
    else:
        f_txt = "1. Face Detection: NO FACE"
        f_col = (0, 0, 255)
    cv2.putText(vis, f_txt, (25, 75), cv2.FONT_HERSHEY_SIMPLEX, 0.5, f_col, 1)

    # 2. Pose status
    if pose_info:
        yaw = pose_info.get("yaw", 0)
        pitch = pose_info.get("pitch", 0)
        roll = pose_info.get("roll", 0)
        p_stat = "PASS" if pose_valid else "FAIL"
        p_txt = f"2. Head Pose [{p_stat}]: Y:{yaw:.1f} P:{pitch:.1f} R:{roll:.1f}"
        p_col = (0, 255, 0) if pose_valid else (0, 0, 255)
    else:
        p_txt = "2. Head Pose: UNKNOWN"
        p_col = (0, 0, 255)
    cv2.putText(vis, p_txt, (25, 100), cv2.FONT_HERSHEY_SIMPLEX, 0.5, p_col, 1)

    # 3. Anti-Spoof status
    if anti_spoof_info:
        as_lbl = anti_spoof_info["label"]
        as_conf = anti_spoof_info["confidence"]
        as_col = (0, 255, 0) if anti_spoof_info["is_real"] else (0, 0, 255)
        as_txt = f"3. Anti-Spoof: {as_lbl} (Conf: {as_conf*100:.1f}%)"
    else:
        as_txt = "3. Anti-Spoof: NO DATA"
        as_col = (0, 165, 255)
    cv2.putText(vis, as_txt, (25, 125), cv2.FONT_HERSHEY_SIMPLEX, 0.5, as_col, 1)

    # 4. Liveness / EAR & MAR
    eye_stat = "OPEN" if ear_val >= 0.20 else "CLOSED"
    l_txt = f"4. Eye EAR: {ear_val:.2f} ({eye_stat}) | MAR: {mar_val:.2f}"
    l_col = (0, 255, 0) if ear_val >= 0.20 else (0, 165, 255)
    cv2.putText(vis, l_txt, (25, 150), cv2.FONT_HERSHEY_SIMPLEX, 0.5, l_col, 1)

    cv2.line(vis, (25, 165), (15 + card_w - 20, 165), (80, 80, 80), 1)

    # 5. Final Decision Banner
    verdict_text = "eKYC: APPROVED (HOP LE)" if final_pass else "eKYC: REJECTED (TU CHOI)"
    verdict_col = (0, 255, 0) if final_pass else (0, 0, 255)
    cv2.putText(vis, verdict_text, (25, 195),
                cv2.FONT_HERSHEY_SIMPLEX, 0.68, verdict_col, 2)

    if not final_pass and reasons:
        reason_str = "Ly do: " + ", ".join(reasons[:2])
        cv2.putText(vis, reason_str, (25, 220),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 200, 255), 1)

    return vis


# =============================================================================
# 4. PROCESS SINGLE IMAGE IN PIPELINE
# =============================================================================
def process_single_image(
    image_path: str,
    output_root_dir: str,
    models_bundle: tuple,
) -> dict:
    detector, landmark_detector, pose_validator, aligner, anti_spoof_detector = models_bundle

    filename = os.path.basename(image_path)
    stem_name = os.path.splitext(filename)[0]

    # Tạo thư mục con riêng biệt cho từng ảnh trong output/
    img_output_dir = os.path.join(output_root_dir, stem_name)
    os.makedirs(img_output_dir, exist_ok=True)

    print(f"\n{'='*65}")
    print(f"  [DANG XU LY] File anh: {filename}")
    print(f"  -> Thu muc luu rieng: {img_output_dir}")
    print(f"{'='*65}")

    image = cv2.imread(image_path)
    if image is None:
        print(f"[ERROR] Khong doc duoc file anh: {image_path}")
        return None

    img_h, img_w = image.shape[:2]
    start_time = time.perf_counter()

    # Step 1: Face Detection
    faces = detector.detect(image)
    primary_face = faces[0] if len(faces) > 0 else None
    print(f"[1. Face Detection] Tim thay {len(faces)} khuon mat.")
    if primary_face:
        bx1, by1, bx2, by2 = primary_face["bbox"]
        print(f"   -> BBox=({bx1}, {by1}, {bx2}, {by2}) | Conf={primary_face['confidence']:.3f}")

    # Step 2: Landmarks
    landmarks = landmark_detector.detect(image)
    print(f"[2. Landmarks] Trich xuat duoc {len(landmarks) if landmarks else 0} diem landmarks.")

    # Step 3: Pose Validation
    pose_valid = False
    pose_text = "Unknown"
    pose_dict = None
    if landmarks and len(landmarks) > 0:
        pose_valid, pose_text, pose_dict = pose_validator.validate(landmarks, get_landmark_point)
        if pose_dict:
            print(f"[3. Head Pose] Y={pose_dict['yaw']:+.1f} deg | P={pose_dict['pitch']:+.1f} deg | R={pose_dict['roll']:+.1f} deg -> {'PASS' if pose_valid else 'FAIL'}")
    else:
        print("[3. Head Pose] Bo qua vi khong co landmarks.")

    # Step 4: Face Alignment & Crop
    aligned_img = None
    face_crop = None
    if landmarks and len(landmarks) > 0:
        aligned_img = aligner.align_face(image, landmarks)
        aligned_lms = aligner.get_landmarks(aligned_img)
        if aligned_lms:
            face_crop = aligner.crop_face(aligned_img, aligned_lms, padding=20, output_size=(224, 224))
            print(f"[4. Alignment & Crop] CROP 224x224 thanh cong: shape={face_crop.shape}")

    # Step 5: Anti-Spoof
    input_for_spoof = face_crop if face_crop is not None else image
    spoof_results = anti_spoof_detector.predict(input_for_spoof, conf_threshold=0.25)
    best_spoof_info = spoof_results[0] if spoof_results else None
    if best_spoof_info:
        print(f"[5. Anti-Spoof] {best_spoof_info['label']} ({best_spoof_info['confidence']*100:.1f}%) | Real={best_spoof_info['is_real']}")
    else:
        print("[5. Anti-Spoof] Khong co ket qua Anti-Spoof.")

    # Step 6: EAR & MAR
    ear_l, ear_r, ear_avg = compute_eye_aspect_ratio(landmarks) if landmarks else (0.0, 0.0, 0.0)
    mar_val = compute_mouth_aspect_ratio(landmarks) if landmarks else 0.0
    eye_open = (ear_avg >= 0.20)
    print(f"[6. Liveness] EAR={ear_avg:.3f} ({'OPEN' if eye_open else 'CLOSED'}) | MAR={mar_val:.3f}")

    # Step 7: Final Decision
    c1 = (primary_face is not None)
    c2 = pose_valid
    c3 = (best_spoof_info is not None and best_spoof_info["is_real"])
    c4 = eye_open

    reasons = []
    if not c1: reasons.append("Khong tim thay khuon mat")
    if not c2: reasons.append("Goc mat chua chuan (Pose)")
    if not c3: reasons.append("Nghi van gia mao (Spoof)")
    if not c4: reasons.append("Mat dang nham")

    final_pass = c1 and c2 and c3 and c4
    elapsed_ms = (time.perf_counter() - start_time) * 1000

    print(f"[7. Final Verdict] {'APPROVED (DAT)' if final_pass else 'REJECTED (TU CHOI)'} | Ly do: {', '.join(reasons) if reasons else 'None'}")

    # Step 8: Drawing & Saving Individual Results in img_output_dir
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

    display = draw_ekyc_hud(
        display,
        primary_face,
        pose_dict,
        pose_valid,
        best_spoof_info,
        ear_avg,
        mar_val,
        final_pass,
        reasons
    )

    # Lưu từng file vào thư mục riêng của ảnh
    out_result_img = os.path.join(img_output_dir, "1_pipeline_result.jpg")
    cv2.imwrite(out_result_img, display)

    if face_crop is not None:
        out_crop_img = os.path.join(img_output_dir, "2_face_crop_224.jpg")
        cv2.imwrite(out_crop_img, face_crop)

    if aligned_img is not None:
        out_aligned_img = os.path.join(img_output_dir, "3_aligned_full.jpg")
        cv2.imwrite(out_aligned_img, aligned_img)

    # Xuất file JSON chi tiết cho ảnh
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
        "final_verdict": "APPROVED" if final_pass else "REJECTED",
        "reasons": reasons
    }

    with open(os.path.join(img_output_dir, "4_report.json"), "w", encoding="utf-8") as f:
        json.dump(record, f, ensure_ascii=False, indent=2)

    print(f"  [OK] Da luu toan bo file ket qua vao: {img_output_dir}")
    return record


# =============================================================================
# 5. BATCH EXECUTION ACROSS ALL IMAGES
# =============================================================================
def run_pipeline_batch(input_path=None, output_path=None):
    if input_path is None:
        input_path = os.path.join(BASE_DIR, "data_raw")
        if not os.path.exists(input_path):
            input_path = os.path.join(BASE_DIR, "data", "raw")

    if output_path is None:
        output_path = os.path.join(CURRENT_DIR, "output")

    os.makedirs(output_path, exist_ok=True)

    # Kiểm tra nếu input_path là 1 file đơn lẻ
    if os.path.isfile(input_path):
        image_files = [input_path]
    elif os.path.isdir(input_path):
        image_files = [
            os.path.join(input_path, f)
            for f in sorted(os.listdir(input_path))
            if os.path.splitext(f)[1].lower() in IMAGE_EXTENSIONS
        ]
    else:
        print(f"[ERROR] Thu muc hoac file khong ton tai: {input_path}")
        sys.exit(1)

    if not image_files:
        print(f"[WARNING] Khong tim thay file anh nao trong: {input_path}")
        return

    print("\n" + "=" * 70)
    print("           FULL E-KYC PIPELINE v2 - BATCH PROCESSING")
    print(f"  * Thu muc dau vao : {input_path}")
    print(f"  * Thu muc output  : {output_path}")
    print(f"  * So luong anh    : {len(image_files)}")
    print("=" * 70)

    # Khởi tạo models 1 lần duy nhất cho toàn bộ batch
    print("\n[INFO] Dang khoi tao cac models cho toan bo Batch...")
    detector = FaceDetector()
    landmark_detector = LandmarkDetector()
    pose_validator = PoseValidator()
    aligner = FaceAligner()
    anti_spoof_detector = AntiSpoofDetector()
    models_bundle = (detector, landmark_detector, pose_validator, aligner, anti_spoof_detector)

    all_records = []
    total_time = 0.0

    for idx, img_file in enumerate(image_files, 1):
        print(f"\n>>> Processing [{idx}/{len(image_files)}]: {os.path.basename(img_file)}")
        rec = process_single_image(img_file, output_path, models_bundle)
        if rec:
            all_records.append(rec)
            total_time += rec["time_ms"]

    # Xuất file tổng kết batch JSON & CSV
    batch_json_path = os.path.join(output_path, "batch_summary.json")
    with open(batch_json_path, "w", encoding="utf-8") as f:
        json.dump(all_records, f, ensure_ascii=False, indent=2)

    batch_csv_path = os.path.join(output_path, "batch_summary.csv")
    with open(batch_csv_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["Image", "Verdict", "Liveness", "Spoof Conf", "Pose Valid", "Yaw", "Pitch", "Roll", "EAR", "Time (ms)", "Output Folder"])
        for r in all_records:
            writer.writerow([
                r["image_name"],
                r["final_verdict"],
                r["anti_spoof"]["label"],
                r["anti_spoof"]["confidence"],
                "PASS" if r["pose_validation"]["is_valid"] else "FAIL",
                r["pose_validation"]["yaw"],
                r["pose_validation"]["pitch"],
                r["pose_validation"]["roll"],
                r["liveness"]["ear"],
                r["time_ms"],
                r["output_folder"]
            ])

    print("\n" + "=" * 70)
    print(f"  [HOAN TAT BATCH] Da xu ly xong {len(all_records)}/{len(image_files)} anh")
    print(f"  * Tong thoi gian   : {total_time:.1f} ms (TB: {total_time/max(1, len(all_records)):.1f} ms/anh)")
    print(f"  * Thu muc ket qua  : {output_path}")
    print(f"  * File tong ket CSV: {batch_csv_path}")
    print(f"  * File tong ket JSON: {batch_json_path}")
    print("=" * 70 + "\n")


# =============================================================================
# 6. WEBCAM REAL-TIME MODE
# =============================================================================
def run_pipeline_webcam(cam_id=0):
    print("\n" + "=" * 65)
    print("       CHAY FULL E-KYC PIPELINE v2 TREN WEBCAM REALTIME")
    print("  * Nhan ESC hoac 'q' de thoat.")
    print("  * Nhan 's' de luu anh snapshot vao thu muc output.")
    print("=" * 65)

    detector = FaceDetector()
    landmark_detector = LandmarkDetector()
    pose_validator = PoseValidator()
    aligner = FaceAligner()
    anti_spoof_detector = AntiSpoofDetector()

    cap = cv2.VideoCapture(cam_id)
    if not cap.isOpened():
        print(f"[ERROR] Khong the mo camera thiet bi ID {cam_id}!")
        sys.exit(1)

    output_dir = os.path.join(CURRENT_DIR, "output")
    os.makedirs(output_dir, exist_ok=True)

    blink_counter = 0
    blink_state = False
    prev_time = time.time()

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        frame = cv2.flip(frame, 1)
        display = frame.copy()

        faces = detector.detect(frame)
        primary_face = faces[0] if faces else None

        landmarks = landmark_detector.detect(frame)

        pose_valid = False
        pose_dict = None
        if landmarks:
            pose_valid, _, pose_dict = pose_validator.validate(landmarks, get_landmark_point)

        ear_l, ear_r, ear_avg = compute_eye_aspect_ratio(landmarks)
        mar_val = compute_mouth_aspect_ratio(landmarks)
        eye_open = (ear_avg >= 0.20)

        if ear_avg > 0.05 and ear_avg < 0.18:
            if not blink_state:
                blink_state = True
        elif ear_avg >= 0.22:
            if blink_state:
                blink_counter += 1
                blink_state = False

        spoof_results = anti_spoof_detector.predict(frame, conf_threshold=0.35)
        best_spoof = spoof_results[0] if spoof_results else None

        c1 = (primary_face is not None)
        c2 = pose_valid
        c3 = (best_spoof is not None and best_spoof["is_real"])
        c4 = eye_open or (blink_counter > 0)
        final_pass = c1 and c2 and c3 and c4

        reasons = []
        if not c1: reasons.append("No Face")
        if not c2: reasons.append("Pose Invalid")
        if not c3: reasons.append("Spoof Detected")
        if not c4: reasons.append("Liveness Check")

        if landmarks:
            display = draw_landmarks(display, landmarks)

        if best_spoof:
            sx1, sy1, sx2, sy2 = best_spoof["bbox"]
            s_col = (0, 255, 0) if best_spoof["is_real"] else (0, 0, 255)
            cv2.rectangle(display, (sx1, sy1), (sx2, sy2), s_col, 2)
            cv2.putText(display, f"{best_spoof['label']} {best_spoof['confidence']*100:.0f}%",
                        (sx1, max(20, sy1 - 8)), cv2.FONT_HERSHEY_SIMPLEX, 0.6, s_col, 2)

        display = draw_ekyc_hud(
            display,
            primary_face,
            pose_dict,
            pose_valid,
            best_spoof,
            ear_avg,
            mar_val,
            final_pass,
            reasons
        )

        cv2.putText(display, f"Blinks: {blink_counter}", (display.shape[1] - 170, 40),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)

        curr_time = time.time()
        fps = 1.0 / (curr_time - prev_time) if curr_time > prev_time else 0.0
        prev_time = curr_time
        cv2.putText(display, f"FPS: {fps:.1f}", (display.shape[1] - 170, 75),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (200, 200, 200), 1)

        cv2.imshow("Full E-KYC Pipeline v2 (Real-Time)", display)

        key = cv2.waitKey(1) & 0xFF
        if key == 27 or key == ord('q'):
            break
        elif key == ord('s'):
            snap_path = os.path.join(output_dir, f"snapshot_{int(time.time())}.jpg")
            cv2.imwrite(snap_path, display)
            print(f"[SAVED SNAPSHOT] {snap_path}")

    cap.release()
    cv2.destroyAllWindows()
    print("[INFO] Webcam pipeline da dung.")


# =============================================================================
# MAIN ENTRY POINT
# =============================================================================
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Full E-KYC Pipeline 2 (Batch & Realtime)")
    parser.add_argument("--input", "-i", type=str, default=None, help="Thu muc chua anh hoac file anh dau vao (mac dinh: data_raw)")
    parser.add_argument("--output", "-o", type=str, default=None, help="Thu muc luu ket qua (mac dinh: backup/tests/output)")
    parser.add_argument("--image", type=str, default=None, help="Duong dan 1 file anh cu the (tuong thich cu)")
    parser.add_argument("--webcam", action="store_true", help="Chay che do Webcam Real-Time")
    parser.add_argument("--cam", type=int, default=0, help="Camera device index (mac dinh 0)")
    args = parser.parse_args()

    if args.webcam:
        run_pipeline_webcam(cam_id=args.cam)
    else:
        target_input = args.image if args.image is not None else args.input
        run_pipeline_batch(input_path=target_input, output_path=args.output)
