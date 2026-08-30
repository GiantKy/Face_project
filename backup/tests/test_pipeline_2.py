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

Hỗ trợ:
  - Chế độ Ảnh tĩnh (mặc định): test trên ảnh (mặc định: data_raw/0.jpg)
  - Chế độ Webcam (--webcam): test thời gian thực với Real-time HUD & Blink Counter
=============================================================================
"""

import sys
import os
import argparse
import time
import math
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
        # Ưu tiên các model theo thứ tự
        candidate_files = [
            f"Anti_Spoof_{model_version}.pt",
            "Anti_Spoof_v4.pt",
            "Anti_Spoof_v3.pt",
            "Anti_Spoof_v2.pt",
            "Anti_Spoof_v1.pt"
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
            for box in r.boxes:
                x1, y1, x2, y2 = map(int, box.xyxy[0])
                conf = float(box.conf[0])
                cls = int(box.cls[0])
                label = self.classes.get(cls, f"class_{cls}").lower()

                # Kiểm tra nhãn là real hay spoof/fake
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

    # Panel tổng hợp bên góc trái trên
    card_w = min(420, w - 20)
    card_h = 240
    draw_ui_card(vis, 15, 15, card_w, card_h, bg_color=(15, 15, 20), alpha=0.85)

    # Tiêu đề
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
# 4. FULL PIPELINE EXECUTION FOR SINGLE IMAGE
# =============================================================================
def run_pipeline_on_image(image_path=None):
    if image_path is None:
        image_path = os.path.join(BASE_DIR, "data_raw", "0.jpg")

    print("\n" + "=" * 65)
    print("           CHẠY FULL E-KYC PIPELINE v2 TRÊN ẢNH")
    print("=" * 65)

    if not os.path.exists(image_path):
        print(f"[ERROR] Không tìm thấy file ảnh: {image_path}")
        sys.exit(1)

    print(f"[INFO] File ảnh đầu vào: {image_path}")
    image = cv2.imread(image_path)
    if image is None:
        print(f"[ERROR] Không đọc được file ảnh: {image_path}")
        sys.exit(1)

    img_h, img_w = image.shape[:2]
    print(f"[INFO] Kích thước ảnh: {img_w}x{img_h}")

    # -------------------------------------------------------------
    # KHỞI TẠO TẤT CẢ CÁC MODULE
    # -------------------------------------------------------------
    print("\n[INFO] Đang khởi tạo các model trong Pipeline...")
    detector = FaceDetector()
    print("  [✓] 1. FaceDetector (YOLO) sẵn sàng")

    landmark_detector = LandmarkDetector()
    print("  [✓] 2. LandmarkDetector (MediaPipe 478 Tasks) sẵn sàng")

    pose_validator = PoseValidator()
    print("  [✓] 3. PoseValidator (Head Pose 3D) sẵn sàng")

    aligner = FaceAligner()
    print("  [✓] 4. FaceAligner & Cropper sẵn sàng")

    anti_spoof_detector = AntiSpoofDetector()
    print("  [✓] 5. AntiSpoofDetector (YOLO Anti-Spoof) sẵn sàng")

    # -------------------------------------------------------------
    # BƯỚC 1: FACE DETECTION
    # -------------------------------------------------------------
    print("\n" + "-" * 40)
    print("BƯỚC 1: Phát hiện khuôn mặt (Face Detection)")
    print("-" * 40)
    faces = detector.detect(image)
    print(f"[KẾT QUẢ] Tìm thấy {len(faces)} khuôn mặt.")

    primary_face = None
    if len(faces) > 0:
        primary_face = faces[0]
        x1, y1, x2, y2 = primary_face["bbox"]
        conf = primary_face["confidence"]
        print(f"  -> Khuôn mặt chính: BBox=({x1}, {y1}, {x2}, {y2}), Confidence={conf:.3f}")
    else:
        print("  [CẢNH BÁO] Không phát hiện được khuôn mặt nào!")

    # -------------------------------------------------------------
    # BƯỚC 2: LANDMARK DETECTION
    # -------------------------------------------------------------
    print("\n" + "-" * 40)
    print("BƯỚC 2: Trích xuất Landmark 3D (Landmark Detection)")
    print("-" * 40)
    landmarks = landmark_detector.detect(image)
    print(f"[KẾT QUẢ] Trích xuất thành công {len(landmarks)} điểm mốc khuôn mặt.")

    # -------------------------------------------------------------
    # BƯỚC 3: POSE VALIDATION (Góc quay đầu)
    # -------------------------------------------------------------
    print("\n" + "-" * 40)
    print("BƯỚC 3: Kiểm tra tư thế đầu (Pose Validation)")
    print("-" * 40)
    pose_valid = False
    pose_text = "Unknown"
    pose_dict = None

    if len(landmarks) > 0:
        pose_valid, pose_text, pose_dict = pose_validator.validate(
            landmarks,
            get_landmark_point
        )
        if pose_dict:
            print(f"  -> Yaw (Trái/Phải):   {pose_dict['yaw']:+.2f}°")
            print(f"  -> Pitch (Lên/Xuống): {pose_dict['pitch']:+.2f}°")
            print(f"  -> Roll (Nghiêng):    {pose_dict['roll']:+.2f}°")
        print(f"[KẾT QUẢ] Đánh giá Pose: {pose_text} -> {'ĐẠT (PASS)' if pose_valid else 'KHÔNG ĐẠT (FAIL)'}")
    else:
        print("  [BỎ QUA] Không có landmark để tính Pose.")

    # -------------------------------------------------------------
    # BƯỚC 4: FACE ALIGNMENT & FACE CROP
    # -------------------------------------------------------------
    print("\n" + "-" * 40)
    print("BƯỚC 4: Căn chỉnh & Cắt khuôn mặt (Face Alignment & Crop)")
    print("-" * 40)
    aligned_img = None
    face_crop = None

    if len(landmarks) > 0:
        aligned_img = aligner.align_face(image, landmarks)
        print("  [✓] Căn chỉnh mắt nằm ngang (Face Alignment) thành công.")

        aligned_lms = aligner.get_landmarks(aligned_img)
        if aligned_lms:
            face_crop = aligner.crop_face(aligned_img, aligned_lms, padding=20, output_size=(224, 224))
            print(f"  [✓] Cắt khuôn mặt (Face Crop 224x224) thành công: shape={face_crop.shape}")
        else:
            print("  [CẢNH BÁO] Không lấy được landmark trên ảnh đã align để crop.")
    else:
        print("  [BỎ QUA] Không có landmark để thực hiện Align & Crop.")

    # -------------------------------------------------------------
    # BƯỚC 5: ANTI-SPOOFING DETECTION
    # -------------------------------------------------------------
    print("\n" + "-" * 40)
    print("BƯỚC 5: Kiểm tra Giả mạo (Anti-Spoofing Detection)")
    print("-" * 40)
    spoof_results = anti_spoof_detector.predict(image, conf_threshold=0.25)
    best_spoof_info = None

    if spoof_results:
        best_spoof_info = spoof_results[0]
        status_str = "REAL (Người thật)" if best_spoof_info["is_real"] else "SPOOF / FAKE (Giả mạo)"
        print(f"[KẾT QUẢ] Phân loại Anti-Spoof: {status_str}")
        print(f"  -> Độ tin cậy (Confidence): {best_spoof_info['confidence']*100:.2f}%")
        print(f"  -> Bounding Box: {best_spoof_info['bbox']}")
    else:
        print("  [CẢNH BÁO] Model Anti-Spoof không tìm thấy box phù hợp trên ảnh.")

    # -------------------------------------------------------------
    # BƯỚC 6: LIVENESS / BLINK & EXPRESSION ANALYSIS
    # -------------------------------------------------------------
    print("\n" + "-" * 40)
    print("BƯỚC 6: Phân tích biểu cảm & Mắt (Liveness / EAR & MAR)")
    print("-" * 40)
    ear_l, ear_r, ear_avg = compute_eye_aspect_ratio(landmarks)
    mar_val = compute_mouth_aspect_ratio(landmarks)
    eye_open = (ear_avg >= 0.20)

    print(f"  -> Eye Aspect Ratio (EAR): Trái={ear_l:.3f}, Phải={ear_r:.3f}, Trung bình={ear_avg:.3f}")
    print(f"  -> Trạng thái mắt: {'MỞ MẮT (OPEN)' if eye_open else 'NHẮM MẮT / CHỚP (CLOSED)'}")
    print(f"  -> Mouth Aspect Ratio (MAR): {mar_val:.3f} ({'MỞ MIỆNG / NÓI' if mar_val > 0.4 else 'NGẬM MIỆNG (CLOSED)'})")

    # -------------------------------------------------------------
    # BƯỚC 7: TỔNG HỢP ĐÁNH GIÁ CUỐI CÙNG (FINAL eKYC DECISION)
    # -------------------------------------------------------------
    print("\n" + "=" * 65)
    print("              BẢNG TỔNG KẾT TIÊU CHÍ eKYC")
    print("=" * 65)

    reasons = []
    
    # Tiêu chí 1: Có khuôn mặt
    c1 = (primary_face is not None)
    if not c1:
        reasons.append("Không tìm thấy khuôn mặt")

    # Tiêu chí 2: Góc nghiêng hợp lệ
    c2 = pose_valid
    if not c2:
        reasons.append("Góc nghiêng đầu không hợp lệ (Pose FAIL)")

    # Tiêu chí 3: Người thật (Anti-Spoof = REAL)
    c3 = (best_spoof_info is not None and best_spoof_info["is_real"])
    if not c3:
        reasons.append("Phát hiện giả mạo hoặc chưa xác thực Anti-Spoof")

    # Tiêu chí 4: Mắt mở bình thường
    c4 = eye_open
    if not c4:
        reasons.append("Mắt đang nhắm")

    final_pass = c1 and c2 and c3 and c4

    print(f"1. Phát hiện khuôn mặt (Face Detection) : {'[ PASS ]' if c1 else '[ FAIL ]'}")
    print(f"2. Tư thế chuẩn góc mặt (Head Pose)     : {'[ PASS ]' if c2 else '[ FAIL ]'}")
    print(f"3. Chống giả mạo (Anti-Spoofing Real)  : {'[ PASS ]' if c3 else '[ FAIL ]'}")
    print(f"4. Trạng thái mắt (Eye Open / Liveness): {'[ PASS ]' if c4 else '[ FAIL ]'}")
    print("-" * 65)
    if final_pass:
        print(">>> KẾT QUẢ eKYC CUỐI CÙNG: [ ĐẠT CHUẨN - APPROVED ] <<<")
    else:
        print(f">>> KẾT QUẢ eKYC CUỐI CÙNG: [ BỊ TỪ CHỐI - REJECTED ] <<<")
        print(f"    Lý do: {', '.join(reasons)}")
    print("=" * 65)

    # -------------------------------------------------------------
    # BƯỚC 8: VẼ HUD VÀ LƯU ẢNH KẾT QUẢ
    # -------------------------------------------------------------
    display = image.copy()

    # 1. Vẽ Face BBox
    if primary_face:
        bx1, by1, bx2, by2 = primary_face["bbox"]
        cv2.rectangle(display, (bx1, by1), (bx2, by2), (0, 255, 0), 2)

    # 2. Vẽ Landmarks
    if landmarks:
        display = draw_landmarks(display, landmarks)

    # 3. Vẽ Anti-Spoof BBox nếu có
    if best_spoof_info:
        sx1, sy1, sx2, sy2 = best_spoof_info["bbox"]
        s_col = (0, 255, 0) if best_spoof_info["is_real"] else (0, 0, 255)
        cv2.rectangle(display, (sx1, sy1), (sx2, sy2), s_col, 2)
        tag = f"{best_spoof_info['label']} {best_spoof_info['confidence']*100:.1f}%"
        cv2.putText(display, tag, (sx1, max(25, sy1 - 10)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, s_col, 2)

    # 4. Vẽ HUD Dashboard
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

    # Lưu kết quả vào backup/tests/output/
    output_dir = os.path.join(CURRENT_DIR, "output")
    os.makedirs(output_dir, exist_ok=True)

    out_full = os.path.join(output_dir, "pipeline2_result.jpg")
    cv2.imwrite(out_full, display)
    print(f"\n[LƯU FILE] Ảnh kết quả Pipeline 2 : {out_full}")

    if face_crop is not None:
        out_crop = os.path.join(output_dir, "pipeline2_face_crop.jpg")
        cv2.imwrite(out_crop, face_crop)
        print(f"[LƯU FILE] Ảnh khuôn mặt crop     : {out_crop}")

    if aligned_img is not None:
        out_aligned = os.path.join(output_dir, "pipeline2_aligned_full.jpg")
        cv2.imwrite(out_aligned, aligned_img)
        print(f"[LƯU FILE] Ảnh sau Alignment      : {out_aligned}")

    print("\n[HOÀN TẤT] Pipeline 2 đã chạy xong toàn bộ các bước thành công!\n")


# =============================================================================
# 5. WEBCAM REAL-TIME MODE
# =============================================================================
def run_pipeline_webcam(cam_id=0):
    print("\n" + "=" * 65)
    print("       CHẠY FULL E-KYC PIPELINE v2 TRÊN WEBCAM REALTIME")
    print("  * Nhấn ESC hoặc 'q' để thoát.")
    print("  * Nhấn 's' để lưu ảnh snapshot vào thư mục output.")
    print("=" * 65)

    detector = FaceDetector()
    landmark_detector = LandmarkDetector()
    pose_validator = PoseValidator()
    aligner = FaceAligner()
    anti_spoof_detector = AntiSpoofDetector()

    cap = cv2.VideoCapture(cam_id)
    if not cap.isOpened():
        print(f"[ERROR] Không thể mở camera thiết bị ID {cam_id}!")
        sys.exit(1)

    output_dir = os.path.join(CURRENT_DIR, "output")
    os.makedirs(output_dir, exist_ok=True)

    # State tracking cho Liveness (Blink Counter)
    blink_counter = 0
    blink_state = False  # True nếu mắt đang nhắm
    prev_time = time.time()

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        frame = cv2.flip(frame, 1)
        display = frame.copy()

        # Step 1: Face Detection
        faces = detector.detect(frame)
        primary_face = faces[0] if faces else None

        # Step 2: Landmarks
        landmarks = landmark_detector.detect(frame)

        # Step 3: Pose
        pose_valid = False
        pose_dict = None
        if landmarks:
            pose_valid, _, pose_dict = pose_validator.validate(landmarks, get_landmark_point)

        # Step 4: Liveness EAR / MAR & Blink count
        ear_l, ear_r, ear_avg = compute_eye_aspect_ratio(landmarks)
        mar_val = compute_mouth_aspect_ratio(landmarks)
        eye_open = (ear_avg >= 0.20)

        # Blink state machine
        if ear_avg > 0.05 and ear_avg < 0.18:
            if not blink_state:
                blink_state = True
        elif ear_avg >= 0.22:
            if blink_state:
                blink_counter += 1
                blink_state = False

        # Step 5: Anti-Spoof
        spoof_results = anti_spoof_detector.predict(frame, conf_threshold=0.35)
        best_spoof = spoof_results[0] if spoof_results else None

        # Step 6: Final Decision
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

        # Step 7: Drawing
        if landmarks:
            display = draw_landmarks(display, landmarks)

        if best_spoof:
            sx1, sy1, sx2, sy2 = best_spoof["bbox"]
            s_col = (0, 255, 0) if best_spoof["is_real"] else (0, 0, 255)
            cv2.rectangle(display, (sx1, sy1), (sx2, sy2), s_col, 2)
            cv2.putText(display, f"{best_spoof['label']} {best_spoof['confidence']*100:.0f}%",
                        (sx1, max(20, sy1 - 8)), cv2.FONT_HERSHEY_SIMPLEX, 0.6, s_col, 2)

        # Draw HUD
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

        # Blink counter badge
        cv2.putText(display, f"Blinks: {blink_counter}", (display.shape[1] - 170, 40),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)

        # FPS
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
    print("[INFO] Webcam pipeline dừng.")


# =============================================================================
# MAIN ENTRY POINT
# =============================================================================
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Full E-KYC Pipeline 2")
    parser.add_argument("--image", type=str, default=None, help="Đường dẫn đến file ảnh cần test")
    parser.add_argument("--webcam", action="store_true", help="Chạy chế độ Webcam Real-Time")
    parser.add_argument("--cam", type=int, default=0, help="Camera device index (mặc định 0)")
    args = parser.parse_args()

    if args.webcam:
        run_pipeline_webcam(cam_id=args.cam)
    else:
        run_pipeline_on_image(image_path=args.image)
