"""
ESP32 Multi-Stage Challenge Manager.
Xử lý quy trình kiểm tra thử thách Liveness từng giai đoạn bằng ảnh tĩnh cho ESP32-CAM:
- Bước 1: Face Detect & Ghi nhận Baseline nhìn thẳng (EAR & 3D Pose).
- Bước 2: Eye Blink (Đo độ sụt giảm EAR so với baseline mở mắt).
- Bước 3: Head Movement (Xoay đầu theo chỉ thị ngẫu nhiên: TURN_LEFT, TURN_RIGHT, LOOK_DOWN).
- Bước 4: Hoàn tất & Chạy Ensemble Anti-Spoofing (YOLO_4 + RF-DETR).
"""

import os
import sys
import time
import math
import random
import uuid
import json
from typing import Dict, Any, Optional, Tuple, List
import numpy as np
import cv2

CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(CURRENT_DIR)
OUTPUT_DIR = os.path.join(PROJECT_ROOT, "output")

try:
    from server_module.utils import (
        load_image,
        image_to_base64,
        calculate_iou,
        get_default_oval_params,
        is_face_in_oval,
        compute_eye_aspect_ratio,
        create_pipeline_result_dashboard,
        create_side_by_side_result
    )
    from server_module.components.pose_validation.utils import get_landmark_point
except ImportError:
    from utils import (
        load_image,
        image_to_base64,
        calculate_iou,
        get_default_oval_params,
        is_face_in_oval,
        compute_eye_aspect_ratio,
        create_pipeline_result_dashboard,
        create_side_by_side_result
    )
    from components.pose_validation.utils import get_landmark_point


def preprocess_esp32_image(frame: np.ndarray, apply_clahe: bool = True, sharpen: bool = True) -> np.ndarray:
    """
    Tiền xử lý ảnh tĩnh từ cảm biến OV2640 (ESP32-CAM):
    1. Phát hiện tình huống ngược sáng (Backlight / High Dynamic Range):
       - Đo độ sáng vùng trung tâm (vùng oval khuôn mặt) so với toàn khung hình.
       - Khi vùng mặt bị tối (center_L < 78) hoặc ngược sáng (mean_L - center_L > 18),
         tự động áp dụng Gamma LUT nâng sáng bóng râm (Shadow Lifting) và CLAHE thích nghi.
    2. Cân bằng sáng cục bộ CLAHE trên kênh Luminance (không gian màu LAB).
    3. Làm sắc nét viền khuôn mặt nhẹ nhàng (Unsharp Masking).
    """
    if frame is None or frame.size == 0:
        return frame

    enhanced = frame.copy()

    # 1. CLAHE trên kênh L & Tự động cứu sáng Gamma nếu ngược sáng hoặc thiếu sáng
    if apply_clahe:
        try:
            lab = cv2.cvtColor(enhanced, cv2.COLOR_BGR2LAB)
            l, a, b = cv2.split(lab)
            h, w = l.shape[:2]

            # Vùng ROI trung tâm nơi mặt người dùng xuất hiện trong khung Oval
            cy1, cy2 = int(h * 0.15), int(h * 0.85)
            cx1, cx2 = int(w * 0.20), int(w * 0.80)
            center_roi = l[cy1:cy2, cx1:cx2]

            mean_l = float(np.mean(l))
            center_l = float(np.mean(center_roi)) if center_roi.size > 0 else mean_l

            # Nhận diện ngược sáng: Nền sáng nhưng mặt tối, hoặc vùng mặt quá tối
            is_backlit = (center_l < 78.0) or ((mean_l - center_l) > 18.0 and center_l < 95.0)

            if is_backlit:
                # Nâng sáng vùng tối có chọn lọc (Shadow lifting gamma curve)
                gamma = max(0.50, min(0.75, center_l / 115.0))
                inv_gamma = 1.0 / gamma
                table = np.array([((i / 255.0) ** inv_gamma) * 255 for i in np.arange(0, 256)]).astype("uint8")
                l = cv2.LUT(l, table)
                clip = 3.2
            elif mean_l < 85.0:
                gamma = max(0.60, mean_l / 110.0)
                inv_gamma = 1.0 / gamma
                table = np.array([((i / 255.0) ** inv_gamma) * 255 for i in np.arange(0, 256)]).astype("uint8")
                l = cv2.LUT(l, table)
                clip = 2.8
            else:
                clip = 2.0

            clahe = cv2.createCLAHE(clipLimit=clip, tileGridSize=(8, 8))
            cl = clahe.apply(l)
            limg = cv2.merge((cl, a, b))
            enhanced = cv2.cvtColor(limg, cv2.COLOR_LAB2BGR)
        except Exception:
            pass

    # 2. Unsharp Masking nhẹ
    if sharpen:
        try:
            gaussian = cv2.GaussianBlur(enhanced, (0, 0), sigmaX=1.5)
            enhanced = cv2.addWeighted(enhanced, 1.25, gaussian, -0.25, 0)
        except Exception:
            pass

    return enhanced


class ChallengeSession:
    """Đại diện cho một phiên thử thách eKYC của thiết bị ESP32."""

    def __init__(self, session_id: str, device_id: str, ttl_seconds: int = 90):
        self.session_id = session_id
        self.device_id = device_id
        self.created_at = time.time()
        self.last_activity = time.time()
        self.ttl_seconds = ttl_seconds

        # Trạng thái luồng
        self.current_step = "face_detect"  # 'face_detect' -> 'eye_blink' -> 'head_movement' -> 'completed'
        self.completed_steps: List[str] = []

        # Dữ liệu cơ sở từ Bước 1
        self.baseline_ear: float = 0.28
        self.baseline_pose: Dict[str, float] = {"yaw": 0.0, "pitch": 0.0, "roll": 0.0}
        self.primary_face_bbox: Optional[List[int]] = None
        self.num_faces: int = 1
        self.pose_dict_static: Optional[Dict[str, Any]] = None
        self.pose_valid_static: bool = True
        self.best_spoof_static: Optional[Dict[str, Any]] = None
        self.primary_spoof_iou: float = 0.0
        self.reasons: List[str] = []
        self.face_crop_static: Optional[np.ndarray] = None
        self.aligned_img_static: Optional[np.ndarray] = None
        self.frontal_frame: Optional[np.ndarray] = None
        self.face_crop_224: Optional[np.ndarray] = None
        self.base_descriptor: Optional[Dict[str, Any]] = None

        # Thử thách quay đầu ngẫu nhiên
        self.target_head_action: str = "TURN_LEFT"
        self.head_prompt: str = "Hãy quay mặt sang bên TRÁI (nhích nhẹ ~10°)"
        self.target_angle_threshold: float = 8.0

        # Kết quả các bài test
        self.face_detect_passed: bool = False
        self.is_real: bool = False
        self.confidence: float = 0.0
        self.crop_face_base64: Optional[str] = None
        self.eye_blink_passed: bool = False
        self.head_movement_passed: bool = False
        self.ensemble_anti_spoof_res: Optional[Dict[str, Any]] = None
        self.final_result: Optional[Dict[str, Any]] = None

        # Quản lý trạng thái Real-time Video Stream Liveness
        self.stream_phase: str = "blink"  # 'blink' -> 'head_movement' -> 'completed'
        self.blink_state: bool = False
        self.blink_counter: int = 0
        self.consecutive_turn_frames: int = 0
        self.current_yaw: float = 0.0
        self.current_pitch: float = 0.0
        self.delta_yaw: float = 0.0
        self.delta_pitch: float = 0.0
        self.current_ear: float = 0.0

    def is_expired(self) -> bool:
        return (time.time() - self.last_activity) > self.ttl_seconds

    def touch(self):
        self.last_activity = time.time()


class ESP32ChallengeManager:
    """Quản lý các phiên thử thách đa giai đoạn của ESP32."""

    def __init__(self):
        self.sessions: Dict[str, ChallengeSession] = {}

    def cleanup_expired(self):
        """Dọn dẹp các session đã quá thời gian chờ."""
        expired_ids = [sid for sid, sess in self.sessions.items() if sess.is_expired()]
        for sid in expired_ids:
            del self.sessions[sid]

    def reset_session(self, session_id: str) -> bool:
        """Xóa thủ công một session."""
        if session_id in self.sessions:
            del self.sessions[session_id]
            return True
        return False

    def _generate_dual_window_report(
        self,
        session: ChallengeSession,
        approved: bool,
        reasons: Optional[List[str]] = None
    ) -> Tuple[np.ndarray, str, Dict[str, Any]]:
        """
        Tạo ảnh Dual Window (Side-by-Side) và báo cáo 4_report.json chuẩn test_pipeline_ensemble_full.py:
        - Nửa trái: Ảnh khuôn mặt có Bounding Box, nhãn REAL/SPOOF, điểm tin cậy %, sub-tag YOLO & RF-DETR,
          và huy hiệu eKYC: APPROVED / eKYC: REJECTED.
        - Dải ngăn cách (divider).
        - Nửa phải: Bảng Dashboard phân tích AI chi tiết 5 phần (Face Detect, 3D Pose, Anti-Spoofing, Liveness, Verdict).
        """
        reasons = reasons or []
        frame = session.frontal_frame if session.frontal_frame is not None else np.zeros((480, 640, 3), dtype=np.uint8)
        clean_img = frame.copy()
        h, w = clean_img.shape[:2]

        # Vẽ Bounding Box & nhãn trên ảnh gốc
        if session.primary_face_bbox:
            bx1, by1, bx2, by2 = session.primary_face_bbox
            cv2.rectangle(clean_img, (bx1, by1), (bx2, by2), (0, 255, 0), 2)

        if session.best_spoof_static:
            sd = session.best_spoof_static
            sx1, sy1, sx2, sy2 = sd.get("bbox", session.primary_face_bbox or [0, 0, w, h])
            if not sd.get("both_detected", True):
                scol = (0, 165, 255)  # Cam: Thiếu đồng thuận 1 model
                tag = f"DISCARD: 1 Model Only ({sd.get('confidence', 0.0)*100:.1f}%)"
            elif approved:
                scol = (0, 255, 0)    # Xanh: Real
                tag = f"REAL {sd.get('confidence', 0.0)*100:.1f}% (Ensemble Matched)"
            else:
                scol = (0, 0, 255)    # Đỏ: Spoof
                tag = f"SPOOF {sd.get('confidence', 0.0)*100:.1f}% (Ensemble Matched)"

            cv2.rectangle(clean_img, (sx1, sy1), (sx2, sy2), scol, 2)
            cv2.putText(clean_img, tag, (sx1, max(20, sy1 - 8)),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.45, scol, 1, cv2.LINE_AA)
            sub_tag = f"YOLO: {sd.get('yolo_res', 'N/A')} | RF: {sd.get('rfdetr_res', 'N/A')}"
            cv2.putText(clean_img, sub_tag, (sx1, min(h - 8, sy2 + 16)),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.38, (220, 220, 220), 1, cv2.LINE_AA)

        # Huy hiệu góc phải
        verdict_badge = "eKYC: APPROVED" if approved else "eKYC: REJECTED"
        badge_col = (0, 255, 0) if approved else (0, 0, 255)
        cv2.rectangle(clean_img, (w - 240, 15), (w - 15, 55), (15, 15, 20), -1)
        cv2.rectangle(clean_img, (w - 240, 15), (w - 15, 55), badge_col, 2)
        cv2.putText(clean_img, verdict_badge, (w - 225, 42), cv2.FONT_HERSHEY_SIMPLEX, 0.62, badge_col, 2)

        # Tạo Dashboard Panel 5 mục
        face_info = {"bbox": session.primary_face_bbox, "confidence": 0.95} if session.primary_face_bbox else None
        dashboard_img = create_pipeline_result_dashboard(
            img_idx=session.session_id,
            face_info=face_info,
            num_faces=session.num_faces,
            pose_info=session.baseline_pose,
            pose_valid=session.pose_valid_static,
            anti_spoof_info=session.best_spoof_static,
            spoof_iou=session.primary_spoof_iou,
            blink_passed=session.eye_blink_passed,
            blink_count=session.blink_counter,
            head_movement_passed=session.head_movement_passed,
            head_action_name=session.target_head_action,
            final_pass=approved,
            reasons=reasons,
            face_crop=session.face_crop_static,
            target_height=h
        )

        # Ghép ảnh Side-by-Side Dual Window
        side_by_side = create_side_by_side_result(clean_img, dashboard_img)

        # Lưu kết quả vào output/<session_id>/
        output_dir = os.path.join(OUTPUT_DIR, str(session.session_id))
        report_data = {
            "session_id": session.session_id,
            "device_id": session.device_id,
            "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
            "approved": approved,
            "verdict": "APPROVED" if approved else "REJECTED",
            "is_real": session.is_real,
            "confidence": round(session.confidence, 4),
            "reasons": reasons,
            "models_used": {
                "model_1": "Anti_Spoof_YOLO_4.pt",
                "model_2": "RF-DETR Small Transformer"
            },
            "face_detection": {
                "num_faces": session.num_faces,
                "bbox": session.primary_face_bbox
            },
            "pose_validation": session.baseline_pose,
            "ensemble_anti_spoof": session.best_spoof_static,
            "active_liveness": {
                "blink_passed": session.eye_blink_passed,
                "blink_count": session.blink_counter,
                "head_movement_passed": session.head_movement_passed,
                "head_action": session.target_head_action
            }
        }
        try:
            os.makedirs(output_dir, exist_ok=True)
            cv2.imwrite(os.path.join(output_dir, "1_pipeline_side_by_side.jpg"), side_by_side)
            cv2.imwrite(os.path.join(output_dir, "1_dashboard_panel.jpg"), dashboard_img)
            cv2.imwrite(os.path.join(output_dir, "1_pipeline_result_clean.jpg"), clean_img)
            if session.face_crop_static is not None and session.face_crop_static.size > 0:
                cv2.imwrite(os.path.join(output_dir, "2_face_crop_224.jpg"), session.face_crop_static)

            with open(os.path.join(output_dir, "4_report.json"), "w", encoding="utf-8") as f:
                json.dump(report_data, f, ensure_ascii=False, indent=2)
        except Exception as e:
            print(f"[ERROR] Không thể ghi file kết quả Dual Window: {e}")

        dual_window_b64 = image_to_base64(side_by_side, quality=85)
        return side_by_side, dual_window_b64, report_data

    def start_challenge(
        self,
        image_input: Any,
        pipeline: Any,
        device_id: str = "ESP32_DEFAULT"
    ) -> Dict[str, Any]:
        """
        GIAI ĐOẠN 1: Bắt đầu phiên bằng 1 ẢNH TĨNH (Snapshot).
        1. Face Detection & Pose Validation (nhìn thẳng).
        2. ENSEMBLE ANTI-SPOOFING (YOLO_4 + RF-DETR).
        3. Nguyên tắc Fail-Fast: Nếu là SPOOF -> Từ chối ngay lập tức, không tốn tài nguyên chạy bước tiếp.
        4. Nếu là REAL -> Khởi tạo phiên, lưu baseline EAR & Pose, sinh ngẫu nhiên hướng quay đầu, sẵn sàng cho Live Stream.
        """
        self.cleanup_expired()
        t0 = time.time()

        raw_frame = load_image(image_input)
        h, w = raw_frame.shape[:2]

        # 0. Kiểm tra số lượng người nghiêm ngặt (Single Person Strict Enforcement)
        if hasattr(pipeline, "identity_verifier") and pipeline.identity_verifier is not None:
            num_faces, is_single = pipeline.identity_verifier.count_faces(raw_frame, pipeline.detector)
            if num_faces > 1:
                captured_b64 = image_to_base64(raw_frame, quality=75)
                return {
                    "success": False,
                    "step": "face_detect",
                    "passed": False,
                    "approved": False,
                    "verdict": "MULTI_FACES",
                    "is_real": False,
                    "num_faces": num_faces,
                    "message": f"Phát hiện {num_faces} người trong khung hình (Yêu cầu 1 người duy nhất).",
                    "reasons": ["MULTI_FACES_DETECTED"],
                    "hint": "Vui lòng chỉ 1 người đứng trước camera.",
                    "captured_image_base64": captured_b64
                }

        # Áp dụng tiền xử lý tối ưu cho ESP32
        proc_frame = preprocess_esp32_image(raw_frame)

        # 1. Face Detection đa tầng độ nhạy cao cho ESP32-CAM
        # Tầng 1: Dò trên ảnh gốc với conf=0.28, min_size=20
        faces = pipeline.detector.detect(raw_frame, conf=0.28, min_size=20)
        if not faces:
            # Tầng 2: Dò trên ảnh tiền xử lý (CLAHE + Sharpness)
            faces = pipeline.detector.detect(proc_frame, conf=0.22, min_size=20)
        if not faces:
            # Tầng 3: Dò với conf nhạy hơn 0.18
            faces = pipeline.detector.detect(raw_frame, conf=0.18, min_size=18)

        # Tầng 4 (MediaPipe BlazeFace Fallback):
        # Nếu YOLO bỏ sót do góc nghiêng hoặc ánh sáng, tận dụng MediaPipe Landmark Detector
        landmarks = None
        if not faces:
            landmarks = pipeline.landmark_detector.detect(proc_frame)
            if not landmarks or len(landmarks) < 468:
                landmarks = pipeline.landmark_detector.detect(raw_frame)
            if landmarks and len(landmarks) >= 468:
                xs = [p[0] for p in landmarks]
                ys = [p[1] for p in landmarks]
                min_x, max_x = min(xs), max(xs)
                min_y, max_y = min(ys), max(ys)
                bw = max_x - min_x
                bh = max_y - min_y
                pad_x = int(bw * 0.18)
                pad_y_top = int(bh * 0.28)
                pad_y_bot = int(bh * 0.12)
                fx1 = max(0, min_x - pad_x)
                fy1 = max(0, min_y - pad_y_top)
                fx2 = min(w, max_x + pad_x)
                fy2 = min(h, max_y + pad_y_bot)
                faces.append({
                    "bbox": [fx1, fy1, fx2, fy2],
                    "confidence": 0.85,
                    "class_id": 0,
                    "face_crop": raw_frame[fy1:fy2, fx1:fx2]
                })

        # Tầng 5: Kiểm tra góc xoay 180 độ nếu camera lắp ngược
        if not faces:
            rot_frame = cv2.rotate(raw_frame, cv2.ROTATE_180)
            rot_faces = pipeline.detector.detect(rot_frame, conf=0.25, min_size=20)
            if rot_faces:
                captured_b64 = image_to_base64(raw_frame, quality=75)
                return {
                    "success": False,
                    "step": "face_detect",
                    "passed": False,
                    "approved": False,
                    "verdict": "CAMERA_UPSIDE_DOWN",
                    "is_real": False,
                    "message": "Camera đang bị lắp ngược 180 độ! Vui lòng xoay lại camera hoặc bật vflip=1.",
                    "reasons": ["CAMERA_INVERTED"],
                    "hint": "Chỉnh lại chiều lắp của ESP32-CAM.",
                    "captured_image_base64": captured_b64
                }

        if not faces:
            captured_b64 = image_to_base64(raw_frame, quality=75)
            return {
                "success": False,
                "step": "face_detect",
                "passed": False,
                "approved": False,
                "verdict": "NO_FACE",
                "is_real": False,
                "message": "Không tìm thấy khuôn mặt trong ảnh. Hãy nhìn thẳng và đứng gần camera hơn.",
                "reasons": ["NO_FACE_DETECTED"],
                "hint": "Chỉnh lại góc camera hoặc tiến lại gần 35-50cm.",
                "captured_image_base64": captured_b64
            }

        # Chọn khuôn mặt chính diện lớn nhất
        def face_size(f):
            b = f["bbox"]
            return (b[2] - b[0]) * (b[3] - b[1])

        primary_face = max(faces, key=face_size)
        bx1, by1, bx2, by2 = primary_face["bbox"]
        face_w = bx2 - bx1
        face_h = by2 - by1

        # Kiểm tra kích thước mặt tối thiểu linh hoạt theo độ phân giải (hỗ trợ từ 240x240 đến VGA)
        min_face_h = min(60, max(32, int(h * 0.15)))
        min_face_w = min(50, max(28, int(w * 0.13)))
        if face_h < min_face_h or face_w < min_face_w:
            annotated = raw_frame.copy()
            cv2.rectangle(annotated, (bx1, by1), (bx2, by2), (0, 165, 255), 2)
            cv2.putText(annotated, f"MAT NHO ({face_h}px)", (bx1, max(18, by1 - 6)),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 165, 255), 1, cv2.LINE_AA)
            captured_b64 = image_to_base64(annotated, quality=75)
            return {
                "success": False,
                "step": "face_detect",
                "passed": False,
                "approved": False,
                "verdict": "TOO_FAR",
                "is_real": False,
                "message": f"Khuôn mặt quá nhỏ ({face_h}px). Hãy tiến lại gần camera hơn.",
                "reasons": ["FACE_TOO_FAR"],
                "hint": "Đứng cách camera khoảng 30 - 45cm.",
                "captured_image_base64": captured_b64
            }

        # 2. MediaPipe Landmarks & Pose Baseline
        if landmarks is None or len(landmarks) < 468:
            landmarks = pipeline.landmark_detector.detect(proc_frame)
            if not landmarks or len(landmarks) < 468:
                landmarks = pipeline.landmark_detector.detect(raw_frame)

        if not landmarks or len(landmarks) < 468:
            captured_b64 = image_to_base64(raw_frame, quality=75)
            return {
                "success": False,
                "step": "face_detect",
                "passed": False,
                "approved": False,
                "verdict": "LANDMARKS_FAILED",
                "is_real": False,
                "message": "Ảnh bị mờ hoặc thiếu chi tiết mắt mũi. Hãy giữ yên và đủ sáng.",
                "reasons": ["LANDMARKS_FAILED"],
                "hint": "Bật đèn chiếu sáng hoặc lau sạch ống kính camera.",
                "captured_image_base64": captured_b64
            }

        # Kiểm tra che mặt (Face Occlusion Check) trước khi cho phép vượt qua Bước 1
        if hasattr(pipeline, "occlusion_detector") and pipeline.occlusion_detector is not None:
            is_occ, occ_code, occ_msg = pipeline.occlusion_detector.check_occlusion(
                frame=raw_frame,
                landmarks=landmarks,
                num_faces=num_faces
            )
            if is_occ:
                captured_b64 = image_to_base64(raw_frame, quality=75)
                return {
                    "success": False,
                    "step": "face_detect",
                    "passed": False,
                    "approved": False,
                    "verdict": "FACE_OCCLUDED",
                    "is_real": False,
                    "message": occ_msg or "CẢNH BÁO: Phát hiện che mặt! Vui lòng không che mặt.",
                    "reasons": ["FACE_OCCLUSION_DETECTED"],
                    "hint": "Bỏ khẩu trang, tay hoặc vật cản ra khỏi khuôn mặt.",
                    "captured_image_base64": captured_b64
                }

        # Tính toán Baseline EAR
        ear_l, ear_r, ear_avg = compute_eye_aspect_ratio(landmarks)
        baseline_ear = max(0.18, float(ear_avg)) if ear_avg > 0.10 else 0.25

        # Tính toán Baseline Pose (Góc nhìn thẳng ban đầu)
        pose_valid, text_status, pose_dict = pipeline.pose_validator.validate(
            landmarks, get_landmark_point, img_w=w, img_h=h
        )
        base_yaw = float(pose_dict.get("yaw", 0.0)) if pose_dict else 0.0
        base_pitch = float(pose_dict.get("pitch", 0.0)) if pose_dict else 0.0
        base_roll = float(pose_dict.get("roll", 0.0)) if pose_dict else 0.0

        # Kiểm tra người dùng có đang nhìn thẳng không (dung sai linh hoạt theo góc đặt camera)
        if abs(base_yaw) > 25.0 or abs(base_pitch) > 22.0:
            annotated = raw_frame.copy()
            cv2.putText(annotated, f"GOC: Y:{base_yaw:+.0f} P:{base_pitch:+.0f}", (bx1, max(18, by1 - 6)),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 200, 255), 1, cv2.LINE_AA)
            captured_b64 = image_to_base64(annotated, quality=75)
            return {
                "success": False,
                "step": "face_detect",
                "passed": False,
                "approved": False,
                "verdict": "NOT_FRONTAL",
                "is_real": False,
                "message": f"Vui lòng nhìn thẳng vào camera để bắt đầu ({text_status}).",
                "reasons": ["POSE_NOT_FRONTAL"],
                "pose": {"yaw": round(base_yaw, 1), "pitch": round(base_pitch, 1)},
                "captured_image_base64": captured_b64
            }

        # Face Crop 224x224 & Alignment
        face_crop_static = None
        aligned_img_static = None
        if hasattr(pipeline, "aligner") and pipeline.aligner is not None:
            try:
                face_crop_static = pipeline.aligner.crop_face(
                    raw_frame, bbox=primary_face["bbox"], padding=25, output_size=(224, 224), mode="bbox"
                )
                if landmarks:
                    aligned_img_static = pipeline.aligner.align_face(raw_frame, landmarks)
            except Exception as e:
                print(f"[WARN] Aligner error: {e}")
        if face_crop_static is None:
            bx = primary_face["bbox"]
            x1, y1, x2, y2 = max(0, bx[0]), max(0, bx[1]), min(w, bx[2]), min(h, bx[3])
            if y2 > y1 and x2 > x1:
                face_crop_static = cv2.resize(raw_frame[y1:y2, x1:x2], (224, 224))

        # 3. DUYỆT ENSEMBLE ANTI-SPOOFING NGAY TẠI BƯỚC 1 (FAIL-FAST)
        t_spoof_start = time.time()
        all_dets, y_dets, rf_dets = pipeline.ensemble_anti_spoof.predict_ensemble(
            proc_frame,
            conf_threshold=0.28,
            iou_thresh=0.38,
            strict_spoof_veto=True,
            spoof_veto_threshold=0.65
        )
        t_spoof = (time.time() - t_spoof_start) * 1000

        # Khớp primary_face với all_dets dùng IoU
        best_spoof_static = None
        primary_spoof_iou = 0.0
        if primary_face and all_dets:
            matching_spoofs = [sd for sd in all_dets if calculate_iou(primary_face["bbox"], sd["bbox"]) > 0.15]
            if matching_spoofs:
                best_spoof_static = max(matching_spoofs, key=lambda x: x["confidence"])
                primary_spoof_iou = calculate_iou(primary_face["bbox"], best_spoof_static["bbox"])
            else:
                best_spoof_static = max(all_dets, key=lambda x: x["confidence"])
                primary_spoof_iou = calculate_iou(primary_face["bbox"], best_spoof_static["bbox"])
        elif all_dets:
            best_spoof_static = all_dets[0]

        is_real = bool(best_spoof_static["is_real"]) if best_spoof_static else False
        confidence = float(best_spoof_static["confidence"]) if best_spoof_static else 0.0
        both_detected = bool(best_spoof_static.get("both_detected", False)) if best_spoof_static else False

        # Kiểm tra điều kiện thất bại / thiếu đồng thuận theo test_pipeline_ensemble_full.py
        reasons = []
        if best_spoof_static is None:
            reasons.append("Khong phat hien duoc dac trung chong gia mao (Anti-Spoof None)")
        elif not both_detected:
            reasons.append(f"Chi co 1 model nhan dien ({best_spoof_static.get('source')}) -> Luoc bo anh (Thieu su dong thuan ca 2 model)")
        elif not is_real:
            reasons.append(f"Phat hien gia mao (SPOOF) voi do tin cay {confidence*100:.1f}%")

        # FAIL-FAST: Nếu là SPOOF hoặc THIẾU ĐỒNG THUẬN -> Tạo báo cáo Dual Window & từ chối ngay lập tức
        if not is_real or not both_detected or best_spoof_static is None:
            t_ms = (time.time() - t0) * 1000
            print(f"[FAIL-FAST BƯỚC 1] Từ chối xác thực: {reasons} | Conf: {confidence*100:.1f}%")

            temp_session = ChallengeSession(
                session_id=f"esp32_fail_{int(time.time())}",
                device_id=device_id
            )
            temp_session.primary_face_bbox = primary_face["bbox"]
            temp_session.num_faces = len(faces)
            temp_session.frontal_frame = raw_frame.copy()
            temp_session.baseline_pose = pose_dict or {"yaw": base_yaw, "pitch": base_pitch, "roll": base_roll}
            temp_session.pose_valid_static = pose_valid
            temp_session.best_spoof_static = best_spoof_static
            temp_session.primary_spoof_iou = primary_spoof_iou
            temp_session.face_crop_static = face_crop_static
            temp_session.aligned_img_static = aligned_img_static
            temp_session.face_detect_passed = True
            temp_session.is_real = False
            temp_session.confidence = confidence

            _, dual_b64, report_data = self._generate_dual_window_report(
                temp_session, approved=False, reasons=reasons
            )
            verdict = "DISCARD_LACK_CONSENSUS" if (not both_detected and best_spoof_static) else "SPOOF"

            crop_b64 = image_to_base64(face_crop_static) if face_crop_static is not None else None
            captured_b64 = image_to_base64(raw_frame, quality=75)
            return {
                "success": False,
                "step": "anti_spoof",
                "passed": False,
                "approved": False,
                "verdict": verdict,
                "is_real": False,
                "confidence": round(confidence, 4),
                "message": f"CẢNH BÁO: {reasons[0]}. Từ chối xác thực.",
                "reasons": reasons,
                "captured_image_base64": captured_b64,
                "dual_window_image_base64": dual_b64,
                "crop_face_base64": crop_b64,
                "processing_time_ms": round(t_ms, 1)
            }

        # 4. TẠO PHIÊN MỚI & SINH NGẪU NHIÊN THỬ THÁCH QUAY ĐẦU
        session_id = f"esp32_{int(time.time())}_{uuid.uuid4().hex[:6]}"
        session = ChallengeSession(session_id=session_id, device_id=device_id)
        session.baseline_ear = baseline_ear
        session.baseline_pose = {"yaw": base_yaw, "pitch": base_pitch, "roll": base_roll}
        session.primary_face_bbox = primary_face["bbox"]
        session.num_faces = len(faces)
        session.pose_valid_static = pose_valid
        session.frontal_frame = raw_frame.copy()
        session.face_crop_static = face_crop_static
        session.aligned_img_static = aligned_img_static
        session.best_spoof_static = best_spoof_static
        session.primary_spoof_iou = primary_spoof_iou
        session.face_detect_passed = True
        session.is_real = True
        session.confidence = confidence
        crop_b64 = image_to_base64(face_crop_static) if face_crop_static is not None else None
        session.crop_face_base64 = crop_b64
        session.completed_steps.append("face_detect")
        session.completed_steps.append("anti_spoof")
        session.current_step = "eye_blink"

        # Sinh ngẫu nhiên hành động quay đầu: QUAY TRÁI hoặc QUAY PHẢI (theo góc nhìn người dùng)
        possible_actions = [
            ("TURN_LEFT", "Hãy quay mặt nhẹ sang bên TRÁI của bạn (~5°-10°)", -5.0),
            ("TURN_RIGHT", "Hãy quay mặt nhẹ sang bên PHẢI của bạn (~5°-10°)", 5.0),
        ]
        chosen_action, prompt_text, target_thresh = random.choice(possible_actions)
        session.target_head_action = chosen_action
        session.head_prompt = prompt_text
        session.target_angle_threshold = target_thresh

        if hasattr(pipeline, "identity_verifier") and pipeline.identity_verifier is not None:
            session.base_descriptor = pipeline.identity_verifier.extract_descriptor(raw_frame)

        self.sessions[session_id] = session

        t_ms = (time.time() - t0) * 1000
        captured_b64 = image_to_base64(raw_frame, quality=75)
        print(f"[BƯỚC 1 HOÀN TẤT] Mặt thật REAL (Conf: {confidence*100:.1f}%) | Session: {session_id} | Thử thách: {chosen_action}")
        return {
            "success": True,
            "session_id": session_id,
            "step": "face_detect_and_antispoof",
            "passed": True,
            "approved": False,
            "verdict": "REAL",
            "is_real": True,
            "confidence": round(confidence, 4),
            "next_step": "eye_blink",
            "challenge_action": chosen_action,
            "action_prompt": prompt_text,
            "message": "Xác thực khuôn mặt thật thành công (REAL)! Tiếp theo: Chớp mắt tự nhiên.",
            "instructions": f"Hãy nhìn vào camera và chớp mắt tự nhiên 1-2 lần.",
            "baseline": {
                "ear": round(baseline_ear, 4),
                "pose": {"yaw": round(base_yaw, 1), "pitch": round(base_pitch, 1)}
            },
            "crop_face_base64": crop_b64,
            "captured_image_base64": captured_b64,
            "timeout_seconds": session.ttl_seconds,
            "processing_time_ms": round(t_ms, 1)
        }

    def process_step(
        self,
        session_id: str,
        image_input: Any,
        pipeline: Any,
        step_name: Optional[str] = None
    ) -> Dict[str, Any]:
        """
        GIAI ĐOẠN 2 & 3 & 4:
        - Xử lý ảnh tĩnh gửi lên cho từng bước ('eye_blink' hoặc 'head_movement').
        - Nếu hoàn thành tất cả: Tự động chạy Ensemble Anti-Spoofing và trả kết luận cuối cùng.
        """
        self.cleanup_expired()
        t0 = time.time()

        if session_id not in self.sessions:
            return {
                "success": False,
                "passed": False,
                "error": "SESSION_NOT_FOUND",
                "message": "Phiên xác thực không tồn tại hoặc đã hết thời gian (Timeout). Vui lòng bắt đầu lại từ Bước 1."
            }

        session = self.sessions[session_id]
        session.touch()
        target_step = step_name or session.current_step

        raw_frame = load_image(image_input)
        h, w = raw_frame.shape[:2]
        proc_frame = preprocess_esp32_image(raw_frame)

        # Trích xuất Landmarks trên frame
        landmarks = pipeline.landmark_detector.detect(proc_frame)
        if not landmarks or len(landmarks) < 468:
            landmarks = pipeline.landmark_detector.detect(raw_frame)

        if not landmarks or len(landmarks) < 468:
            return {
                "success": False,
                "session_id": session_id,
                "step": target_step,
                "passed": False,
                "message": "Không nhận diện được các nét khuôn mặt ở bước này. Vui lòng chụp lại rõ nét hơn.",
                "reasons": ["LANDMARKS_FAILED"]
            }

        # Kiểm tra nhiều người trong khung hình & kiểm tra tráo đổi người (Face Continuity Defense)
        if hasattr(pipeline, "identity_verifier") and pipeline.identity_verifier is not None:
            num_faces, is_single = pipeline.identity_verifier.count_faces(raw_frame, pipeline.detector)
            if num_faces > 1:
                return {
                    "success": False,
                    "session_id": session_id,
                    "step": target_step,
                    "passed": False,
                    "error": "MULTI_FACES",
                    "message": f"Phát hiện {num_faces} người trong khung hình! Vui lòng chỉ 1 người thực hiện.",
                    "reasons": ["MULTI_FACES_DETECTED"]
                }
            if getattr(session, "base_descriptor", None) is not None:
                cand_desc = pipeline.identity_verifier.extract_descriptor(raw_frame)
                if cand_desc is not None:
                    is_same, score, details = pipeline.identity_verifier.verify_identity(session.base_descriptor, cand_desc)
                    if not is_same:
                        return {
                            "success": False,
                            "session_id": session_id,
                            "step": target_step,
                            "passed": False,
                            "error": "FACE_MISMATCH",
                            "message": "CẢNH BÁO: Phát hiện đổi người! Yêu cầu đúng người chụp ảnh ban đầu thực hiện thử thách.",
                            "reasons": ["FACE_IDENTITY_MISMATCH"]
                        }

        # Kiểm tra che mặt (Face Occlusion Defense) trong lúc đang thực hiện thử thách
        # Nếu phát hiện che mặt: Dừng ngay lập tức, TUYỆT ĐỐI KHÔNG THAY ĐỔI EAR HOẶC HEAD YAW!
        if hasattr(pipeline, "occlusion_detector") and pipeline.occlusion_detector is not None:
            is_occ, occ_code, occ_msg = pipeline.occlusion_detector.check_occlusion(
                frame=raw_frame,
                landmarks=landmarks,
                num_faces=num_faces
            )
            if is_occ:
                return {
                    "success": False,
                    "session_id": session_id,
                    "step": target_step,
                    "passed": False,
                    "is_occluded": True,
                    "error": "FACE_OCCLUDED",
                    "message": occ_msg or "CẢNH BÁO: Phát hiện che mặt! Vui lòng không che mặt khi thực hiện thử thách.",
                    "reasons": ["FACE_OCCLUSION_DETECTED"]
                }

        # ---------------------------------------------------------------------
        # XỬ LÝ BƯỚC 2: EYE BLINK (CHỚP MẮT) - State Machine: MỞ→NHẮM→MỞ = 1 blink
        # Ngưỡng tham chiếu từ test_pipeline_ensemble_full.py:
        #   ear < 0.18 → nhắm, ear >= 0.22 → mở lại, cần ≥ 1 blink
        # ---------------------------------------------------------------------
        if target_step == "eye_blink":
            ear_l, ear_r, ear_avg = compute_eye_aspect_ratio(landmarks)
            base_ear = session.baseline_ear
            session.current_ear = float(ear_avg)

            # State machine: MẮT MỞ → MẮT NHẮM → MẮT MỞ LẠI = 1 blink
            # Độ nhạy thích ứng: nhắm mắt (EAR giảm > 20% hoặc < 0.18)
            is_closed = (ear_avg < 0.18) or (ear_avg <= base_ear * 0.80)
            # Mở lại: phục hồi về >= 0.20 hoặc >= 88% baseline ban đầu
            is_opened = (ear_avg >= 0.20) or (ear_avg >= base_ear * 0.88)

            if is_closed:
                # Mắt đang nhắm
                session.blink_state = True
            elif is_opened and session.blink_state:
                # Mắt vừa mở lại sau khi nhắm → hoàn thành 1 blink
                session.blink_counter += 1
                session.blink_state = False
                print(f"[BLINK PUSH] Phát hiện chớp mắt thành công! Tổng: {session.blink_counter}/1")

            if session.blink_counter >= 1:
                session.eye_blink_passed = True
                if "eye_blink" not in session.completed_steps:
                    session.completed_steps.append("eye_blink")
                session.current_step = "head_movement"

                t_ms = (time.time() - t0) * 1000
                return {
                    "success": True,
                    "session_id": session_id,
                    "step": "eye_blink",
                    "passed": True,
                    "next_step": "head_movement",
                    "message": "Đã xác nhận chớp mắt thành công! Bước tiếp theo: Quay đầu.",
                    "challenge_action": session.target_head_action,
                    "action_prompt": session.head_prompt,
                    "target_head_action": session.target_head_action,
                    "head_prompt": session.head_prompt,
                    "blink_counter": session.blink_counter,
                    "progress": 1.0,
                    "ear": {
                        "current": round(float(ear_avg), 4),
                        "baseline": round(float(base_ear), 4)
                    },
                    "processing_time_ms": round(t_ms, 1)
                }
            else:
                # Chưa hoàn thành blink: trả về trạng thái hiện tại
                progress = 0.5 if session.blink_state else 0.0
                t_ms = (time.time() - t0) * 1000
                return {
                    "success": True,
                    "session_id": session_id,
                    "step": "eye_blink",
                    "passed": False,
                    "next_step": "eye_blink",
                    "blink_state": session.blink_state,
                    "blink_counter": session.blink_counter,
                    "progress": round(progress, 2),
                    "message": f"{'Đang nhắm mắt... giữ rồi mở ra!' if session.blink_state else 'Chưa phát hiện chớp mắt. Hãy nhắm mắt rồi mở ra tự nhiên.'}",
                    "ear": {
                        "current": round(float(ear_avg), 4),
                        "baseline": round(float(base_ear), 4)
                    },
                    "processing_time_ms": round(t_ms, 1)
                }

        # ---------------------------------------------------------------------
        # XỬ LÝ BƯỚC 3: HEAD MOVEMENT (QUAY ĐẦU) - Tích lũy consecutive frames
        # Quy ước góc PnP chuẩn: TURN_LEFT là Yaw ÂM, TURN_RIGHT là Yaw DƯƠNG
        # ---------------------------------------------------------------------
        if target_step == "head_movement":
            pose_valid, text_status, pose_dict = pipeline.pose_validator.validate(
                landmarks, get_landmark_point, img_w=w, img_h=h
            )
            curr_yaw = float(pose_dict.get("yaw", 0.0)) if pose_dict else 0.0
            curr_pitch = float(pose_dict.get("pitch", 0.0)) if pose_dict else 0.0

            base_pose = session.baseline_pose
            base_yaw = base_pose.get("yaw", 0.0)
            delta_yaw = curr_yaw - base_yaw

            session.current_yaw = curr_yaw
            session.current_pitch = curr_pitch
            session.delta_yaw = delta_yaw

            # Đánh giá theo thử thách ngẫu nhiên (Đúng chuẩn hệ tọa độ PnP: TRÁI là ÂM, PHẢI là DƯƠNG)
            head_matched = False
            action = session.target_head_action

            if action == "TURN_LEFT":
                # Quay TRÁI của người dùng: delta_yaw ÂM
                head_matched = (delta_yaw <= -3.5) or (curr_yaw <= -5.0)
            elif action == "TURN_RIGHT":
                # Quay PHẢI của người dùng: delta_yaw DƯƠNG
                head_matched = (delta_yaw >= 3.5) or (curr_yaw >= 5.0)

            if head_matched:
                # Nếu quay góc rõ rệt (|delta_yaw| >= 5.0 hoặc |curr_yaw| >= 7.5): cho pass ngay sau 1 frame rõ
                if (action == "TURN_LEFT" and (delta_yaw <= -5.0 or curr_yaw <= -7.5)) or \
                   (action == "TURN_RIGHT" and (delta_yaw >= 5.0 or curr_yaw >= 7.5)):
                    session.consecutive_turn_frames += 2
                else:
                    session.consecutive_turn_frames += 1
            else:
                # Giữ điểm nhẹ nhàng, không trừ sạch điểm khi người dùng vừa quay đầu lại nhìn camera
                session.consecutive_turn_frames = max(0, session.consecutive_turn_frames - 1)

            # Cần >= 2 điểm tích lũy để pass
            progress = min(1.0, session.consecutive_turn_frames / 2.0)

            if session.consecutive_turn_frames >= 2:
                session.head_movement_passed = True
                if "head_movement" not in session.completed_steps:
                    session.completed_steps.append("head_movement")
                session.current_step = "completed"
                progress = 1.0
                print(f"[HEAD PUSH] Đã hoàn thành quay đầu [{action}]!")

                # Đánh giá điều kiện approved chuẩn test_pipeline_ensemble_full.py
                reasons = []
                if session.primary_face_bbox is None:
                    reasons.append("Khong phat hien khuon mat trong khung oval")
                elif session.num_faces > 1:
                    reasons.append(f"Phat hien nhieu khuon mat ({session.num_faces} mat)")
                if not session.pose_valid_static:
                    reasons.append("Goc mat nghieng/khong thang ve phia camera")
                if session.best_spoof_static is None:
                    reasons.append("Khong phat hien duoc dac trung chong gia mao (Anti-Spoof None)")
                elif not session.best_spoof_static.get("both_detected", False):
                    reasons.append(f"Chi co 1 model nhan dien ({session.best_spoof_static.get('source')}) -> Luoc bo anh (Thieu su dong thuan)")
                elif not session.best_spoof_static.get("is_real", False):
                    reasons.append(f"Phat hien gia mao (SPOOF) voi do tin cay {session.best_spoof_static.get('confidence', 0.0)*100:.1f}%")
                if not session.eye_blink_passed:
                    reasons.append("Chua vuot qua thu thach chop mat (Blink Liveness)")
                if not session.head_movement_passed:
                    reasons.append("Chua vuot qua thu thach quay dau (Head Movement)")

                approved = (len(reasons) == 0)
                session.reasons = reasons

                # Tạo ảnh Dual Window (Side-by-Side) và báo cáo 4_report.json
                side_by_side, dual_window_b64, report_data = self._generate_dual_window_report(
                    session=session,
                    approved=approved,
                    reasons=reasons
                )

                final_verdict = "REAL" if approved else "REJECTED"
                crop_b64 = session.crop_face_base64
                if crop_b64 is None and session.face_crop_static is not None:
                    crop_b64 = image_to_base64(session.face_crop_static)

                # Vẽ annotated frame
                annotated_b64 = None
                if session.frontal_frame is not None:
                    annotated = session.frontal_frame.copy()
                    if session.primary_face_bbox:
                        bx = session.primary_face_bbox
                        border_c = (0, 255, 0) if approved else (0, 0, 255)
                        lbl = "REAL (APPROVED)" if approved else "CHALLENGE FAILED"
                        cv2.rectangle(annotated, (bx[0], bx[1]), (bx[2], bx[3]), border_c, 2)
                        cv2.putText(annotated, lbl, (bx[0], max(18, bx[1] - 8)),
                                    cv2.FONT_HERSHEY_SIMPLEX, 0.45, border_c, 1, cv2.LINE_AA)
                    annotated_b64 = image_to_base64(annotated, quality=75)

                t_total_ms = (time.time() - t0) * 1000
                final_report = {
                    "success": True,
                    "session_id": session_id,
                    "device_id": session.device_id,
                    "step": "completed",
                    "passed": True,
                    "all_challenges_passed": True,
                    "approved": approved,
                    "verdict": final_verdict,
                    "is_real": session.is_real,
                    "confidence": round(session.confidence, 4),
                    "reasons": reasons,
                    "message": "XÁC THỰC TOÀN DIỆN THÀNH CÔNG (Mở cửa!)" if approved else f"TỪ CHỐI XÁC THỰC ({final_verdict})",
                    "progress": 1.0,
                    "steps_summary": {
                        "face_detect": session.face_detect_passed,
                        "anti_spoof_real": session.is_real,
                        "eye_blink": session.eye_blink_passed,
                        "head_movement": session.head_movement_passed
                    },
                    "head_movement_details": {
                        "action": action,
                        "delta_yaw": round(delta_yaw, 1)
                    },
                    "crop_face_base64": crop_b64,
                    "captured_image_base64": annotated_b64,
                    "dual_window_image_base64": dual_window_b64,
                    "processing_time_ms": round(t_total_ms, 1)
                }

                # Xóa session đã hoàn thành
                del self.sessions[session_id]
                return final_report

            # Chưa đạt: trả về trạng thái tiến trình
            t_ms = (time.time() - t0) * 1000
            return {
                "success": True,
                "session_id": session_id,
                "step": "head_movement",
                "passed": False,
                "next_step": "head_movement",
                "progress": round(progress, 2),
                "consecutive_frames": session.consecutive_turn_frames,
                "challenge_action": action,
                "action_prompt": session.head_prompt,
                "target_head_action": session.target_head_action,
                "head_prompt": session.head_prompt,
                "message": f"Góc quay đầu chưa đạt ({action}). delta_yaw={delta_yaw:.1f}°. Tiến trình: {int(progress*100)}%.",
                "current_pose": {"yaw": round(curr_yaw, 1), "pitch": round(curr_pitch, 1)},
                "delta": {"yaw": round(delta_yaw, 1)},
                "processing_time_ms": round(t_ms, 1)
            }

        return {
            "success": False,
            "session_id": session_id,
            "error": "INVALID_STEP",
            "message": f"Bước '{target_step}' không hợp lệ. Các bước hỗ trợ: 'eye_blink', 'head_movement'."
        }


# Singleton challenge manager
esp32_challenge_manager = ESP32ChallengeManager()


# NOTE: process_stream_frame() và process_stream_challenge() đã bị xóa.
# Kiến trúc mới: ESP32 push ảnh JPEG liên tục qua /challenge-step thay vì AI Server kéo stream MJPEG.
# Xem PIPELINE_ARCHITECTURE.md để biết chi tiết.



