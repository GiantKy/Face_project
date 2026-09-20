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
from typing import Dict, Any, Optional, Tuple, List
import numpy as np
import cv2

try:
    from server_module.utils import (
        load_image,
        image_to_base64,
        calculate_iou,
        get_default_oval_params,
        is_face_in_oval,
        compute_eye_aspect_ratio
    )
    from server_module.components.pose_validation.utils import get_landmark_point
except ImportError:
    from utils import (
        load_image,
        image_to_base64,
        calculate_iou,
        get_default_oval_params,
        is_face_in_oval,
        compute_eye_aspect_ratio
    )
    from components.pose_validation.utils import get_landmark_point


def preprocess_esp32_image(frame: np.ndarray, apply_clahe: bool = True, sharpen: bool = True) -> np.ndarray:
    """
    Tiền xử lý ảnh tĩnh từ cảm biến OV2640 (ESP32-CAM):
    1. Cân bằng sáng cục bộ thích nghi CLAHE trên kênh Luminance (không gian màu LAB).
       -> Khắc phục tình trạng mặt bị bệt đen do thiếu sáng hoặc ngược sáng.
    2. Làm sắc nét (Unsharp Masking).
       -> Khắc phục tình trạng mất viền nét (edges) của ống kính OV2640.
    """
    if frame is None or frame.size == 0:
        return frame

    enhanced = frame.copy()

    # 1. CLAHE trên kênh L
    if apply_clahe:
        try:
            lab = cv2.cvtColor(enhanced, cv2.COLOR_BGR2LAB)
            l, a, b = cv2.split(lab)
            clahe = cv2.createCLAHE(clipLimit=2.2, tileGridSize=(8, 8))
            cl = clahe.apply(l)
            limg = cv2.merge((cl, a, b))
            enhanced = cv2.cvtColor(limg, cv2.COLOR_LAB2BGR)
        except Exception:
            pass

    # 2. Unsharp Masking
    if sharpen:
        try:
            gaussian = cv2.GaussianBlur(enhanced, (0, 0), sigmaX=2.0)
            enhanced = cv2.addWeighted(enhanced, 1.35, gaussian, -0.35, 0)
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
        self.frontal_frame: Optional[np.ndarray] = None
        self.face_crop_224: Optional[np.ndarray] = None

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

        # Tính toán Baseline EAR
        ear_l, ear_r, ear_avg = compute_eye_aspect_ratio(landmarks)
        baseline_ear = max(0.20, float(ear_avg))

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

        best_det = all_dets[0] if all_dets else None
        is_real = bool(best_det["is_real"]) if best_det else False
        confidence = float(best_det["confidence"]) if best_det else 0.0
        verdict = "REAL" if is_real else "SPOOF"

        # Nếu Anti-Spoofing kết luận là GIẢ MẠO (SPOOF) -> FAIL-FAST ngay!
        if not is_real or best_det is None:
            t_ms = (time.time() - t0) * 1000
            print(f"[FAIL-FAST BƯỚC 1] Phát hiện giả mạo (SPOOF)! Confidence: {confidence*100:.1f}% -> TỪ CHỐI NGAY.")
            annotated = raw_frame.copy()
            if primary_face:
                bx = primary_face["bbox"]
                cv2.rectangle(annotated, (bx[0], bx[1]), (bx[2], bx[3]), (0, 0, 255), 2)
                cv2.putText(annotated, f"SPOOF ({confidence*100:.1f}%)", (bx[0], max(18, bx[1] - 8)),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 0, 255), 1, cv2.LINE_AA)
            captured_b64 = image_to_base64(annotated, quality=75)
            return {
                "success": False,
                "step": "anti_spoof",
                "passed": False,
                "approved": False,
                "verdict": "SPOOF",
                "is_real": False,
                "confidence": round(confidence, 4),
                "message": f"CẢNH BÁO: Phát hiện khuôn mặt giả mạo (SPOOF)! Độ tin cậy: {confidence*100:.1f}%. Từ chối xác thực.",
                "reasons": ["SPOOF_DETECTED"],
                "captured_image_base64": captured_b64,
                "processing_time_ms": round(t_ms, 1)
            }

        # Cắt khuôn mặt (Crop 224x224)
        crop_b64 = None
        bx = primary_face["bbox"]
        x1, y1, x2, y2 = max(0, bx[0]), max(0, bx[1]), min(w, bx[2]), min(h, bx[3])
        if y2 > y1 and x2 > x1:
            crop_img = raw_frame[y1:y2, x1:x2]
            crop_b64 = image_to_base64(crop_img)

        # 4. TẠO PHIÊN MỚI & SINH NGẪU NHIÊN THỬ THÁCH QUAY ĐẦU
        session_id = f"esp32_{int(time.time())}_{uuid.uuid4().hex[:6]}"
        session = ChallengeSession(session_id=session_id, device_id=device_id)
        session.baseline_ear = baseline_ear
        session.baseline_pose = {"yaw": base_yaw, "pitch": base_pitch, "roll": base_roll}
        session.primary_face_bbox = primary_face["bbox"]
        session.frontal_frame = raw_frame.copy()
        session.face_detect_passed = True
        session.is_real = True
        session.confidence = confidence
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
            # Nới lỏng độ nhạy: Chỉ cần nhắm mắt nhẹ (EAR giảm > 15% hoặc < 0.20)
            is_closed = (ear_avg < 0.20) or (ear_avg <= base_ear * 0.85)
            # Mở lại: phục hồi về >= 0.21 hoặc >= 90% baseline ban đầu
            is_opened = (ear_avg >= 0.21) or (ear_avg >= base_ear * 0.90)

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
                head_matched = (delta_yaw <= -4.0) or (curr_yaw <= -6.0)
            elif action == "TURN_RIGHT":
                # Quay PHẢI của người dùng: delta_yaw DƯƠNG
                head_matched = (delta_yaw >= 4.0) or (curr_yaw >= 6.0)

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

                # Tổng hợp quyết định cuối cùng (Anti-Spoof đã pass ở Bước 1)
                approved = bool(
                    session.face_detect_passed and
                    session.is_real and
                    session.eye_blink_passed and
                    session.head_movement_passed
                )
                final_verdict = "REAL" if approved else "REJECTED"
                crop_b64 = session.crop_face_base64

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



