"""
Core E-KYC Pipeline Server Engine — Ensemble Edition.
Sử dụng Ensemble Anti-Spoofing (YOLO_4 + RF-DETR Small) để duyệt ảnh.
Hoàn toàn headless (không webcam, không GUI), tương thích Server.
"""

import os
import sys
import math
import json
import csv
import time
import uuid
from typing import Union, Dict, Any, List, Optional, Tuple
import numpy as np
import cv2

# Import hoàn toàn từ các components nội bộ bên trong server_module
try:
    from .components import (
        FaceDetector,
        LandmarkDetector,
        draw_landmarks,
        get_landmark_point,
        PoseValidator,
        FaceAligner,
        HeadMovementDetector,
        HeadAction,
        ChallengeState,
        FaceIdentityVerifier,
        FaceOcclusionDetector
    )
    from .config import (
        FACE_DETECTION_MODEL_PATH,
        FACE_LANDMARKER_MODEL_PATH,
        ANTI_SPOOF_YOLO4_MODEL_PATH,
        RFDETR_MODEL_ID,
        RFDETR_API_KEY,
        CONF_THRESHOLD_FACE,
        ENSEMBLE_CONF_THRESHOLD,
        ENSEMBLE_IOU_THRESHOLD,
        ENSEMBLE_W_YOLO,
        ENSEMBLE_W_RFDETR,
        ENSEMBLE_SPOOF_VETO_THRESHOLD,
        POSE_MAX_YAW,
        POSE_MAX_PITCH,
        POSE_MAX_ROLL,
        MIN_FACE_HEIGHT,
        OVAL_FIT_MIN_RATIO,
        OVAL_FIT_MAX_RATIO,
        EAR_EYE_CLOSED_THRESHOLD,
        EAR_EYE_OPEN_THRESHOLD,
        MIN_BLINKS_REQUIRED,
        HEAD_YAW_THRESHOLD,
        HEAD_PITCH_THRESHOLD,
        HEAD_DELTA_YAW_THRESHOLD,
        HEAD_DELTA_PITCH_THRESHOLD,
        CHALLENGE_TIMEOUT_SECONDS
    )
    from .ensemble_anti_spoof import EnsembleAntiSpoofDetector
    from .utils import (
        load_image,
        image_to_base64,
        calculate_iou,
        compute_eye_aspect_ratio,
        json_serialize_helper,
        draw_pipeline_result_hud,
        create_pipeline_result_dashboard,
        create_side_by_side_result,
        remove_vietnamese_accents,
        get_default_oval_params,
        is_point_in_oval,
        is_face_in_oval,
        check_face_oval_fit,
        filter_faces_in_oval,
        get_oval_masked_frame,
        draw_oval_face_guide
    )
except (ImportError, ValueError):
    from components import (
        FaceDetector,
        LandmarkDetector,
        draw_landmarks,
        get_landmark_point,
        PoseValidator,
        FaceAligner,
        HeadMovementDetector,
        HeadAction,
        ChallengeState,
        FaceIdentityVerifier,
        FaceOcclusionDetector
    )
    from config import (
        FACE_DETECTION_MODEL_PATH,
        FACE_LANDMARKER_MODEL_PATH,
        ANTI_SPOOF_YOLO4_MODEL_PATH,
        RFDETR_MODEL_ID,
        RFDETR_API_KEY,
        CONF_THRESHOLD_FACE,
        ENSEMBLE_CONF_THRESHOLD,
        ENSEMBLE_IOU_THRESHOLD,
        ENSEMBLE_W_YOLO,
        ENSEMBLE_W_RFDETR,
        ENSEMBLE_SPOOF_VETO_THRESHOLD,
        POSE_MAX_YAW,
        POSE_MAX_PITCH,
        POSE_MAX_ROLL,
        MIN_FACE_HEIGHT,
        OVAL_FIT_MIN_RATIO,
        OVAL_FIT_MAX_RATIO,
        EAR_EYE_CLOSED_THRESHOLD,
        EAR_EYE_OPEN_THRESHOLD,
        MIN_BLINKS_REQUIRED,
        HEAD_YAW_THRESHOLD,
        HEAD_PITCH_THRESHOLD,
        HEAD_DELTA_YAW_THRESHOLD,
        HEAD_DELTA_PITCH_THRESHOLD,
        CHALLENGE_TIMEOUT_SECONDS
    )
    from ensemble_anti_spoof import EnsembleAntiSpoofDetector
    from utils import (
        load_image,
        image_to_base64,
        calculate_iou,
        compute_eye_aspect_ratio,
        json_serialize_helper,
        draw_pipeline_result_hud,
        create_pipeline_result_dashboard,
        create_side_by_side_result,
        remove_vietnamese_accents,
        get_default_oval_params,
        is_point_in_oval,
        is_face_in_oval,
        check_face_oval_fit,
        filter_faces_in_oval,
        get_oval_masked_frame,
        draw_oval_face_guide
    )
try:
    from src.illumination import check_illumination_quality, enhance_low_light
except (ImportError, ValueError):
    try:
        from illumination import check_illumination_quality, enhance_low_light
    except (ImportError, ValueError):
        check_illumination_quality = None
        enhance_low_light = None


class EKYCPipelineServer:
    """
    E-KYC Server Pipeline Engine — Ensemble Edition.
    
    Sử dụng Ensemble Anti-Spoofing (YOLO_4 + RF-DETR Small) kết hợp
    IoU Matching + Weighted Soft-Voting + Spoof Veto.
    
    CHẾ ĐỘ HOẠT ĐỘNG:
    - Hoàn toàn xử lý trên dữ liệu ảnh (Image file path, Base64, Bytes buffer hoặc NumPy array).
    - KHÔNG SỬ DỤNG WEBCAM: Tuyệt đối KHÔNG gọi cv2.VideoCapture(), KHÔNG mở webcam, KHÔNG dùng GUI.
    - TƯƠNG THÍCH SERVER: Chạy an toàn 100% trên máy chủ headless.

    Quy trình xử lý trên ảnh:
    1. Face Detection: YOLO (Face_Detection.pt)
    2. Face Landmarks: MediaPipe 478 points
    3. 3D Pose Validation: Euler Angles (Yaw, Pitch, Roll)
    4. Face Alignment & Normalization: Affine Transform (224x224 crop)
    5. Ensemble Anti-Spoofing: YOLO_4 + RF-DETR Small (IoU Matching + Soft Voting + Spoof Veto)
    6. Active Liveness: EAR chớp mắt & quay đầu (từ frame gửi lên)
    7. Decision Engine: Đánh giá hợp chuẩn eKYC và xuất JSON / File ảnh
    """

    def __init__(
        self,
        face_model_path: Optional[str] = None,
        yolo_antispoof_path: Optional[str] = None,
        rfdetr_model_id: Optional[str] = None,
        rfdetr_api_key: Optional[str] = None,
        lazy_load: bool = False
    ):
        self.face_model_path = face_model_path or FACE_DETECTION_MODEL_PATH
        self.yolo_antispoof_path = yolo_antispoof_path or ANTI_SPOOF_YOLO4_MODEL_PATH
        self.face_landmarker_path = FACE_LANDMARKER_MODEL_PATH
        self.rfdetr_model_id = rfdetr_model_id or RFDETR_MODEL_ID
        self.rfdetr_api_key = rfdetr_api_key or RFDETR_API_KEY

        self.detector: Optional[FaceDetector] = None
        self.landmark_detector: Optional[LandmarkDetector] = None
        self.pose_validator: Optional[PoseValidator] = None
        self.aligner: Optional[FaceAligner] = None
        self.ensemble_anti_spoof: Optional[EnsembleAntiSpoofDetector] = None
        self.head_movement_detector: Optional[HeadMovementDetector] = None
        self.identity_verifier: Optional[FaceIdentityVerifier] = None
        self.occlusion_detector: Optional[FaceOcclusionDetector] = None
        self.liveness_sessions: Dict[str, Dict[str, Any]] = {}

        if not lazy_load:
            self.load_models()

    def load_models(self):
        """Khởi tạo và tải trước toàn bộ mô hình AI vào bộ nhớ."""
        self.detector = FaceDetector(model_path=self.face_model_path, conf_thresh=CONF_THRESHOLD_FACE, iou_thresh=0.40)
        self.landmark_detector = LandmarkDetector()
        self.pose_validator = PoseValidator(
            max_yaw=POSE_MAX_YAW,
            max_pitch=POSE_MAX_PITCH,
            max_roll=POSE_MAX_ROLL
        )
        self.aligner = FaceAligner()
        self.ensemble_anti_spoof = EnsembleAntiSpoofDetector(
            yolo_model_path=self.yolo_antispoof_path,
            rfdetr_model_id=self.rfdetr_model_id,
            rfdetr_api_key=self.rfdetr_api_key,
        )
        self.head_movement_detector = HeadMovementDetector(
            yaw_threshold=HEAD_YAW_THRESHOLD,
            pitch_threshold=HEAD_PITCH_THRESHOLD,
            timeout=CHALLENGE_TIMEOUT_SECONDS,
            min_consecutive_frames=2,
            delta_yaw_threshold=HEAD_DELTA_YAW_THRESHOLD,
            delta_pitch_threshold=HEAD_DELTA_PITCH_THRESHOLD
        )
        self.identity_verifier = FaceIdentityVerifier(model_path=self.face_landmarker_path)
        self.occlusion_detector = FaceOcclusionDetector()
        print("[EKYCPipelineServer] Tải toàn bộ AI Models thành công! (Ensemble, Identity & Anti-Occlusion Ready)\n")

    def _ensure_models_loaded(self):
        if self.detector is None or self.identity_verifier is None or self.occlusion_detector is None:
            self.load_models()

    def cleanup_expired_sessions(self, ttl_seconds: int = 180):
        """Dọn dẹp các session liveness đã quá thời gian chờ (TTL 3 phút)."""
        now = time.time()
        expired = [sid for sid, s in self.liveness_sessions.items() if (now - s.get("created_at", now)) > ttl_seconds]
        for sid in expired:
            del self.liveness_sessions[sid]

    def init_liveness_session(
        self,
        base_frame_input: Union[str, bytes, np.ndarray],
        session_id: Optional[str] = None
    ) -> Dict[str, Any]:
        """
        Khởi tạo phiên thử thách Liveness từ ảnh chuẩn Bước 1 (Base Frame):
        1. Kiểm tra nghiêm ngặt 1 người duy nhất trong khung hình (num_faces == 1).
        2. Trích xuất Biometric Descriptor 3D của người chụp Bước 1.
        3. Caching descriptor vào memory để đối chiếu liên tục trong Bước 2, Bước 3, Bước 4.
        """
        self._ensure_models_loaded()
        self.cleanup_expired_sessions()

        frame = load_image(base_frame_input)

        # Tự động tăng sáng nếu ảnh bị tối (giữ nguyên hướng và tọa độ ảnh chuẩn)
        try:
            from src.illumination import enhance_low_light
            frame_proc = enhance_low_light(frame)
        except Exception:
            frame_proc = frame

        h_f, w_f = frame_proc.shape[:2]
        oval_center, oval_axes = get_default_oval_params(w_f, h_f)

        masked_frame = get_oval_masked_frame(frame_proc, oval_center, oval_axes, blur_ksize=45, dim_factor=0.35)
        num_faces, is_single = self.identity_verifier.count_faces(
            masked_frame, self.detector, oval_center=oval_center, oval_axes=oval_axes, filter_oval=True, oval_tolerance=1.0
        )
        if num_faces == 0:
            # Thử lại với conf=0.35 trên masked_frame
            raw_faces = self.detector.detect(masked_frame, conf=0.35)
            faces_in_oval = filter_faces_in_oval(raw_faces, oval_center, oval_axes, img_w=w_f, img_h=h_f, tolerance=1.0)
            num_faces = len(faces_in_oval)

        if num_faces == 0:
            # Fallback dò trên frame_proc gốc phòng khi masked_frame làm mờ viền mặt
            raw_faces = self.detector.detect(frame_proc, conf=0.35)
            faces_in_oval = filter_faces_in_oval(raw_faces, oval_center, oval_axes, img_w=w_f, img_h=h_f, tolerance=1.05)
            num_faces = len(faces_in_oval)
        if num_faces > 1:
            return {
                "success": False,
                "num_faces": num_faces,
                "error": "MULTI_FACES",
                "message": f"Phát hiện {num_faces} người trong khung hình! Vui lòng chỉ 1 người duy nhất thực hiện eKYC."
            }

        # Kiểm tra che mặt nghiêm ngặt ngay tại ảnh chụp Bước 1 (Anti-Occlusion Defense)
        landmarks = self.landmark_detector.detect(
            masked_frame, oval_center=oval_center, oval_axes=oval_axes, filter_oval=True, oval_tolerance=1.0
        )
        if not landmarks:
            landmarks = self.landmark_detector.detect(
                frame_proc, oval_center=oval_center, oval_axes=oval_axes, filter_oval=True, oval_tolerance=1.0
            )

        if not landmarks:
            return {
                "success": False,
                "error": "NO_FACE",
                "message": "Không tìm thấy các mốc điểm khuôn mặt trong khung oval. Vui lòng căn chỉnh mặt vào khung oval."
            }

        # Nếu landmark tìm thấy nhưng detector hụt do viền tối oval mask -> đảm bảo num_faces = 1
        if num_faces == 0 and landmarks:
            num_faces = 1

        # KIỂM TRA MẶT KHỚP CHUẨN KHUNG OVAL (OVAL FIT STANDARD) - BƯỚC 1
        fit_info = check_face_oval_fit(
            landmarks, w_f, h_f, oval_center, oval_axes,
            min_ratio=OVAL_FIT_MIN_RATIO, max_ratio=OVAL_FIT_MAX_RATIO, min_face_height=MIN_FACE_HEIGHT
        )
        if not fit_info["face_in_oval"]:
            return {
                "success": False,
                "error": "FACE_NOT_IN_OVAL",
                "message": "Khuôn mặt không nằm trong khung oval! Vui lòng căn chỉnh mặt vào khung oval."
            }
        if fit_info["is_too_far"]:
            pct_oval = int(round(fit_info["ratio_to_oval"] * 100))
            return {
                "success": False,
                "error": "FACE_TOO_FAR",
                "face_size_h": fit_info["face_size_h"],
                "ratio_to_oval": fit_info["ratio_to_oval"],
                "message": f"Khuôn mặt quá xa (chiếm {pct_oval}% oval, yêu cầu tối thiểu {int(round(OVAL_FIT_MIN_RATIO * 100))}%)! Vui lòng tiến lại gần cho khớp khung oval."
            }
        if fit_info["is_too_close"]:
            pct_oval = int(round(fit_info["ratio_to_oval"] * 100))
            return {
                "success": False,
                "error": "FACE_TOO_CLOSE",
                "face_size_h": fit_info["face_size_h"],
                "ratio_to_oval": fit_info["ratio_to_oval"],
                "message": f"Khuôn mặt quá gần camera (chiếm {pct_oval}% oval, giới hạn tối đa {int(round(OVAL_FIT_MAX_RATIO * 100))}%)! Vui lòng lùi lại một chút cho vừa khung oval."
            }
        if fit_info["is_off_center"]:
            return {
                "success": False,
                "error": "FACE_OFF_CENTER",
                "message": fit_info["off_center_hint"] or "Khuôn mặt bị lệch tâm! Vui lòng căn chỉnh vào giữa khung oval."
            }

        is_occ, occ_code, occ_msg = self.occlusion_detector.check_occlusion(
            frame=frame,
            landmarks=landmarks,
            num_faces=num_faces,
            oval_center=oval_center,
            oval_axes=oval_axes,
            filter_oval=True
        )
        if is_occ:
            return {
                "success": False,
                "error": "FACE_OCCLUDED",
                "occlusion_reason": occ_code,
                "message": occ_msg or "CẢNH BÁO: Phát hiện khuôn mặt bị che khuất! Vui lòng không che mặt khi thực hiện eKYC."
            }

        desc = self.identity_verifier.extract_descriptor(
            frame_proc, oval_center=oval_center, oval_axes=oval_axes, filter_oval=True
        )
        if desc is None:
            desc = self.identity_verifier.extract_descriptor(
                frame, oval_center=oval_center, oval_axes=oval_axes, filter_oval=True
            )

        if desc is None:
            return {
                "success": False,
                "error": "DESCRIPTOR_FAILED",
                "message": "Không thể trích xuất đặc trưng sinh trắc học từ khuôn mặt."
            }

        sid = session_id or f"sess_{int(time.time()*1000)}_{uuid.uuid4().hex[:6]}"
        self.liveness_sessions[sid] = {
            "session_id": sid,
            "created_at": time.time(),
            "last_activity": time.time(),
            "base_desc": desc,
            "base_frame": frame.copy(),
            "num_faces": 1,
            "fit_oval": True,
            "base_face_size_h": fit_info["face_size_h"],
            "base_ratio_to_oval": fit_info["ratio_to_oval"],
            "blink_passed": False,
            "head_passed": False,
            "blink_frame": None,
            "head_frame": None,
            "blink_start_time": time.time()
        }

        return {
            "success": True,
            "session_id": sid,
            "num_faces": 1,
            "fit_oval": True,
            "face_size_h": fit_info["face_size_h"],
            "ratio_to_oval": fit_info["ratio_to_oval"],
            "message": "Đã khởi tạo phiên xác thực thành công. Bắt đầu thử thách sinh trắc học liên tục."
        }

    def reset_liveness_session(self, session_id: str) -> bool:
        """Hủy phiên thử thách và xóa descriptor để không tái sử dụng sang phiên mới."""
        if session_id and session_id in self.liveness_sessions:
            del self.liveness_sessions[session_id]
            return True
        return False

    # =========================================================================
    # 1. KIỂM TRA TƯ THẾ & CĂN CHỈNH KHUÔN MẶT (PRE-CAPTURE CHECK)
    # =========================================================================
    def validate_pose(self, image_input: Union[str, bytes, np.ndarray]) -> Dict[str, Any]:
        """
        Kiểm tra tư thế khuôn mặt (trước khi chụp):
        - Phát hiện số lượng khuôn mặt (nghiêm ngặt 1 người duy nhất)
        - Phát hiện khuôn mặt và landmarks
        - Đánh giá khoảng cách camera (kích thước mặt)
        - Đánh giá 3 góc Euler (Yaw, Pitch, Roll)
        
        Returns:
            Dict chứa trạng thái pose, góc quay và thông báo hướng dẫn.
        """
        self._ensure_models_loaded()
        frame = load_image(image_input)
        h, w = frame.shape[:2]

        # Tọa độ khung Oval trung tâm
        oval_center, oval_axes = get_default_oval_params(w, h)
        oval_cx, oval_cy = oval_center
        oval_ax, oval_ay = oval_axes

        # 0. Kiểm tra khuôn mặt & đếm số người bằng Single-Pass MediaPipe (~15ms)
        # Quét trên masked_frame để triệt tiêu người thứ 2 ngoài oval
        masked_frame = get_oval_masked_frame(frame, oval_center, oval_axes, blur_ksize=45, dim_factor=0.35)
        landmarks, num_faces = self.landmark_detector.detect_with_count(
            masked_frame, oval_center=oval_center, oval_axes=oval_axes, filter_oval=True, oval_tolerance=1.0
        )
        if not landmarks and num_faces == 0:
            landmarks, num_faces = self.landmark_detector.detect_with_count(
                frame, oval_center=oval_center, oval_axes=oval_axes, filter_oval=True, oval_tolerance=1.0
            )
        if num_faces > 1:
            return {
                "has_face": True,
                "num_faces": num_faces,
                "is_valid": False,
                "face_in_oval": False,
                "is_aligned_good": False,
                "face_size_h": 0,
                "is_too_far": False,
                "is_too_close": False,
                "is_off_center": False,
                "off_center_hint": "",
                "oval_guide": {
                    "center": [oval_cx, oval_cy],
                    "axes": [oval_ax, oval_ay]
                },
                "pose": {"yaw": 0.0, "pitch": 0.0, "roll": 0.0, "status_text": "MULTI_FACES"},
                "message": f"Phát hiện {num_faces} người trong khung hình (Yêu cầu 1 người duy nhất)",
                "guide": "Vui lòng chỉ 1 người đứng trước camera"
            }

        if not landmarks or num_faces == 0:
            return {
                "has_face": False,
                "num_faces": 0,
                "is_valid": False,
                "face_size_h": 0,
                "is_too_far": True,
                "pose": {"yaw": 0.0, "pitch": 0.0, "roll": 0.0},
                "oval_guide": {
                    "center": [oval_cx, oval_cy],
                    "axes": [oval_ax, oval_ay]
                },
                "message": "Không tìm thấy khuôn mặt trong khung oval",
                "guide": "Vui lòng đưa khuôn mặt vào trong khung oval"
            }

        # Đánh giá kích thước và độ khớp Oval bằng check_face_oval_fit
        fit_info = check_face_oval_fit(
            landmarks, w, h, oval_center, oval_axes,
            min_ratio=OVAL_FIT_MIN_RATIO, max_ratio=OVAL_FIT_MAX_RATIO, min_face_height=MIN_FACE_HEIGHT
        )
        face_in_oval = fit_info["face_in_oval"]
        face_size_h = fit_info["face_size_h"]
        is_too_far = fit_info["is_too_far"]
        is_too_close = fit_info["is_too_close"]
        is_off_center = fit_info["is_off_center"]
        off_center_hint = fit_info["off_center_hint"]
        fit_oval = fit_info["fit_oval"]
        ratio_to_oval = fit_info["ratio_to_oval"]

        # Đánh giá góc nghiêng 3D Pose
        pose_valid, text_status, pose_dict = self.pose_validator.validate(landmarks, get_landmark_point, img_w=w, img_h=h)
        pose_data = pose_dict if pose_dict else {"yaw": 0.0, "pitch": 0.0, "roll": 0.0}

        # Kiểm tra che mặt (Face Occlusion Defense) trước khi cho phép chụp ảnh
        is_occluded, occ_reason, occ_msg = self.occlusion_detector.check_occlusion(
            frame=frame,
            landmarks=landmarks,
            num_faces=num_faces,
            pose_dict=pose_dict,
            oval_center=(oval_cx, oval_cy),
            oval_axes=(oval_ax, oval_ay),
            filter_oval=True
        )
        if is_occluded:
            return {
                "has_face": True,
                "num_faces": num_faces,
                "is_valid": False,
                "is_aligned_good": False,
                "fit_oval": bool(fit_oval),
                "ratio_to_oval": float(ratio_to_oval),
                "is_occluded": True,
                "occlusion_reason": occ_reason,
                "face_in_oval": bool(face_in_oval),
                "face_size_h": int(face_size_h),
                "is_too_far": bool(is_too_far),
                "is_too_close": bool(is_too_close),
                "is_off_center": bool(is_off_center),
                "off_center_hint": off_center_hint,
                "oval_guide": {
                    "center": [oval_cx, oval_cy],
                    "axes": [oval_ax, oval_ay]
                },
                "pose": {
                    "yaw": round(float(pose_data.get("yaw", 0.0)), 2),
                    "pitch": round(float(pose_data.get("pitch", 0.0)), 2),
                    "roll": round(float(pose_data.get("roll", 0.0)), 2),
                    "status_text": "OCCLUDED"
                },
                "message": ("Vui lòng đưa khuôn mặt vào trong khung oval" if not face_in_oval else (occ_msg or "CẢNH BÁO: PHÁT HIỆN CHE MẶT HOẶC ĐEO KÍNH!")),
                "guide": ("Vui lòng đưa khuôn mặt vào trong khung oval" if not face_in_oval else "VUI LÒNG THÁO KÍNH / KHẨU TRANG RA ĐỂ CHỤP")
            }

        is_valid_overall = (
            fit_oval and
            pose_valid and
            not is_occluded
        )

        if not face_in_oval:
            guide_msg = "Vui lòng đưa khuôn mặt vào trong khung oval"
        elif is_off_center:
            guide_msg = off_center_hint or "Vui lòng căn chỉnh khuôn mặt vào giữa khung oval"
        elif is_too_far:
            guide_msg = "Vui lòng tiến lại gần camera hơn để vừa vặn khung oval"
        elif is_too_close:
            guide_msg = "Vui lòng lùi xa camera một chút"
        elif not pose_valid:
            if text_status == "Turn Left":
                guide_msg = "Đầu đang lệch trái, vui lòng nhìn thẳng"
            elif text_status == "Turn Right":
                guide_msg = "Đầu đang lệch phải, vui lòng nhìn thẳng"
            elif text_status in ("Head Up", "Head Down"):
                guide_msg = "Vui lòng nhìn thẳng ngang tầm mắt"
            elif text_status == "Head Tilt":
                guide_msg = "Vui lòng giữ thẳng đầu, không nghiêng"
            else:
                guide_msg = f"Vui lòng nhìn thẳng vào camera ({text_status})"
        else:
            guide_msg = "Khuôn mặt chuẩn trong khung Oval!"

        return {
            "has_face": True,
            "num_faces": num_faces,
            "is_valid": bool(is_valid_overall),
            "face_in_oval": bool(face_in_oval),
            "fit_oval": bool(fit_oval),
            "ratio_to_oval": float(ratio_to_oval),
            "is_aligned_good": bool(is_valid_overall),
            "is_occluded": False,
            "occlusion_reason": "",
            "face_size_h": int(face_size_h),
            "is_too_far": bool(is_too_far),
            "is_too_close": bool(is_too_close),
            "is_off_center": bool(is_off_center),
            "off_center_hint": off_center_hint,
            "oval_guide": {
                "center": [oval_cx, oval_cy],
                "axes": [oval_ax, oval_ay]
            },
            "pose": {
                "yaw": round(float(pose_data.get("yaw", 0.0)), 2),
                "pitch": round(float(pose_data.get("pitch", 0.0)), 2),
                "roll": round(float(pose_data.get("roll", 0.0)), 2),
                "status_text": text_status
            },
            "message": guide_msg if not is_valid_overall else "OK",
            "guide": guide_msg
        }

    # =========================================================================
    # 2. KIỂM TRA CHỐNG GIẢ MẠO — ENSEMBLE (YOLO_4 + RF-DETR)
    # =========================================================================
    def check_antispoof(
        self,
        image_input: Union[str, bytes, np.ndarray],
        conf_threshold: float = ENSEMBLE_CONF_THRESHOLD
    ) -> Dict[str, Any]:
        """
        Thực hiện kiểm tra chống giả mạo bằng Ensemble (YOLO_4 + RF-DETR Small):
        - Quét toàn bộ ảnh bằng cả 2 model
        - IoU Matching + Soft Voting + Spoof Veto
        - Khớp IoU với khuôn mặt chính
        - Căn chỉnh Affine và Crop 224x224
        
        Returns:
            Dict chứa kết quả ensemble anti-spoof chi tiết.
        """
        self._ensure_models_loaded()
        frame = load_image(image_input)
        h_f, w_f = frame.shape[:2]
        oval_center, oval_axes = get_default_oval_params(w_f, h_f)

        # 1. Phát hiện khuôn mặt (làm mờ ngoại vi để triệt tiêu người thứ 2 ngoài oval, dung sai 1.0)
        masked_frame = get_oval_masked_frame(frame, oval_center, oval_axes, blur_ksize=45, dim_factor=0.35)
        raw_faces = self.detector.detect(masked_frame, conf=CONF_THRESHOLD_FACE)
        faces_in_oval = filter_faces_in_oval(raw_faces, oval_center, oval_axes, img_w=w_f, img_h=h_f, tolerance=1.0)
        faces = faces_in_oval
        num_faces = len(faces_in_oval)

        if not faces:
            return {
                "has_face": False,
                "is_real": False,
                "label": "NO_FACE",
                "confidence": 0.0,
                "num_faces": 0,
                "primary_face": None,
                "crop_face_base64": None,
                "ensemble_detail": None,
                "all_ensemble_detections": []
            }

        # Tìm Primary Face: Lớn nhất và gần tâm oval nhất
        def get_face_priority(f):
            bx1, by1, bx2, by2 = f["bbox"]
            area = (bx2 - bx1) * (by2 - by1)
            cx, cy = (bx1 + bx2) / 2.0, (by1 + by2) / 2.0
            dist_center = math.hypot(cx - oval_center[0], cy - oval_center[1])
            return area - (dist_center * 10)

        primary_face = max(faces, key=get_face_priority)

        # 2. Căn chỉnh và crop 224x224
        landmarks = self.landmark_detector.detect(
            masked_frame, oval_center=oval_center, oval_axes=oval_axes, filter_oval=True, oval_tolerance=1.0
        )
        if not landmarks:
            landmarks = self.landmark_detector.detect(
                frame, oval_center=oval_center, oval_axes=oval_axes, filter_oval=True, oval_tolerance=1.0
            )
        aligned_img = None
        face_crop_224 = None

        if landmarks:
            aligned_img = self.aligner.align_face(frame, landmarks)
            aligned_lms = self.aligner.get_landmarks(aligned_img)
            if aligned_lms:
                face_crop_224 = self.aligner.crop_face(
                    aligned_img, aligned_lms, padding=20, output_size=(224, 224)
                )

        # Fallback crop từ BBox
        if face_crop_224 is None and primary_face is not None:
            px1, py1, px2, py2 = primary_face["bbox"]
            px1_c, py1_c = max(0, min(w_f - 1, px1)), max(0, min(h_f - 1, py1))
            px2_c, py2_c = max(0, min(w_f, px2)), max(0, min(h_f, py2))
            raw_crop_p = frame[py1_c:py2_c, px1_c:px2_c]
            if raw_crop_p.size > 0:
                face_crop_224 = cv2.resize(raw_crop_p, (224, 224))

        # 3. Ensemble Anti-Spoofing (YOLO_4 + RF-DETR Small)
        all_ensemble_dets, yolo_dets, rfdetr_dets = self.ensemble_anti_spoof.predict_ensemble(
            frame, conf_threshold=conf_threshold
        )

        # Lọc chỉ lấy detections trong oval
        spoofs_in_oval = [sd for sd in all_ensemble_dets if is_face_in_oval(sd["bbox"], oval_center, oval_axes, tolerance=1.0)]
        target_spoofs = spoofs_in_oval

        # Tìm detection khớp nhất với Primary Face
        best_spoof = None
        primary_spoof_iou = 0.0
        if primary_face and target_spoofs:
            matching_spoofs = [
                sd for sd in target_spoofs
                if calculate_iou(primary_face["bbox"], sd["bbox"]) > 0.15
            ]
            if matching_spoofs:
                best_spoof = max(matching_spoofs, key=lambda x: x["confidence"])
                primary_spoof_iou = calculate_iou(primary_face["bbox"], best_spoof["bbox"])
            else:
                best_spoof = max(target_spoofs, key=lambda x: x["confidence"])
                primary_spoof_iou = calculate_iou(primary_face["bbox"], best_spoof["bbox"])

        if best_spoof is None and target_spoofs:
            best_spoof = target_spoofs[0]

        is_real = bool(best_spoof["is_real"]) if best_spoof else False
        label = best_spoof["label"] if best_spoof else "UNKNOWN"
        confidence = float(best_spoof["confidence"]) if best_spoof else 0.0

        crop_b64 = image_to_base64(face_crop_224) if face_crop_224 is not None else None

        return {
            "has_face": True,
            "num_faces": num_faces,
            "is_real": is_real,
            "label": label,
            "confidence": round(confidence, 4),
            "primary_spoof_iou": round(primary_spoof_iou, 4),
            "primary_face": {
                "bbox": primary_face["bbox"],
                "confidence": round(float(primary_face["confidence"]), 4)
            },
            "crop_face_base64": crop_b64,
            "ensemble_detail": {
                "source": best_spoof.get("source") if best_spoof else "NONE",
                "yolo_detail": best_spoof.get("yolo_res") if best_spoof else "N/A",
                "rfdetr_detail": best_spoof.get("rfdetr_res") if best_spoof else "N/A",
                "agreement": best_spoof.get("agreement") if best_spoof else False,
                "both_detected": best_spoof.get("both_detected") if best_spoof else False,
            },
            "all_ensemble_detections": [
                {
                    "bbox": sd["bbox"],
                    "is_real": sd["is_real"],
                    "label": sd["label"],
                    "confidence": round(float(sd["confidence"]), 4),
                    "source": sd.get("source", ""),
                    "yolo_res": sd.get("yolo_res", "N/A"),
                    "rfdetr_res": sd.get("rfdetr_res", "N/A"),
                    "both_detected": sd.get("both_detected", False),
                }
                for sd in all_ensemble_dets
            ]
        }

    # =========================================================================
    # 3. ĐÁNH GIÁ ACTIVE LIVENESS (CHỚP MẮT & CỬ ĐỘNG ĐẦU)
    # =========================================================================
    def evaluate_blink_frame(
        self,
        frame_input: Union[str, bytes, np.ndarray],
        current_blink_counter: int = 0,
        current_blink_state: bool = False,
        baseline_ear: float = 0.0,
        session_id: Optional[str] = None,
        base_frame_input: Optional[Union[str, bytes, np.ndarray]] = None
    ) -> Dict[str, Any]:
        """
        Đo lường chỉ số EAR và kiểm tra chớp mắt tốc độ cao (High-Speed Real-time Blink Engine).
        Tối ưu hóa:
        - Xử lý Single-Pass Landmark Detector: tốc độ 15-25ms/frame (không nghẽn CPU, không delay).
        - Ngưỡng chớp mắt tương đối thích ứng (Adaptive Relative Drop) xử lý hoàn hảo mọi dáng mắt / đeo kính / ngược sáng.
        - Miễn nhiễm với báo động giả kính râm khi nhắm mắt / bóng mí mắt trong thử thách chớp mắt.
        - Thẩm định danh tính (Face Identity Continuity) trên frame hoàn tất chớp mắt mắt mở lại rõ nét.
        """
        self._ensure_models_loaded()
        frame = load_image(frame_input)
        h, w = frame.shape[:2]

        # 0. Trích xuất landmarks và đếm số khuôn mặt trong 1 lần suy luận duy nhất (Single-Pass MediaPipe ~20ms)
        # Chỉ nhận mặt trong khung oval và bỏ đi các mặt ngoài oval
        oval_center, oval_axes = get_default_oval_params(w, h)
        landmarks, num_faces = self.landmark_detector.detect_with_count(
            frame, oval_center=oval_center, oval_axes=oval_axes, filter_oval=True
        )
        if num_faces > 1:
            return {
                "has_face": True,
                "num_faces": num_faces,
                "same_person": False,
                "passed": False,
                "error": f"Phát hiện {num_faces} người trong khung hình (Yêu cầu 1 người duy nhất)",
                "label": "⚠️ PHÁT HIỆN NHIỀU NGƯỜI",
                "ear_left": 0.0,
                "ear_right": 0.0,
                "ear_avg": 0.0,
                "baseline_ear": baseline_ear,
                "closed_thresh": 0.18,
                "open_thresh": 0.21,
                "blink_counter": current_blink_counter,
                "blink_state": current_blink_state,
                "progress": 0.0
            }

        has_face = bool(landmarks is not None and len(landmarks) >= 468 and num_faces > 0)
        ear_l, ear_r, ear_avg = compute_eye_aspect_ratio(landmarks) if landmarks else (0.0, 0.0, 0.0)

        # 1. Kiểm tra giới hạn thời gian thử thách chớp mắt (CHALLENGE_TIMEOUT_SECONDS = 10s)
        blink_time_left = CHALLENGE_TIMEOUT_SECONDS
        blink_timed_out = False
        if session_id and session_id in self.liveness_sessions:
            sess = self.liveness_sessions[session_id]
            if "blink_start_time" not in sess:
                sess["blink_start_time"] = time.time()
            elapsed_blink = time.time() - sess["blink_start_time"]
            blink_time_left = max(0.0, CHALLENGE_TIMEOUT_SECONDS - elapsed_blink)
            if elapsed_blink > CHALLENGE_TIMEOUT_SECONDS:
                blink_timed_out = True

        if blink_timed_out:
            return {
                "has_face": bool(has_face),
                "num_faces": num_faces,
                "same_person": True,
                "passed": False,
                "timed_out": True,
                "time_left": 0.0,
                "timeout_seconds": CHALLENGE_TIMEOUT_SECONDS,
                "error": "HẾT THỜI GIAN: Chưa hoàn thành chớp mắt đúng hạn (10s).",
                "label": "⚠️ HẾT THỜI GIAN (10s)",
                "ear_left": round(ear_l, 4),
                "ear_right": round(ear_r, 4),
                "ear_avg": round(ear_avg, 4),
                "baseline_ear": round(baseline_ear, 4),
                "closed_thresh": 0.18,
                "open_thresh": 0.21,
                "blink_counter": current_blink_counter,
                "blink_state": False,
                "progress": 0.0
            }

        # 2. Kiểm tra che mặt & mắt kính thực tế (Face Occlusion Defense)
        is_occluded = False
        occ_reason = ""
        occ_msg = ""
        if has_face:
            is_occ_raw, raw_code, raw_msg = self.occlusion_detector.check_occlusion(
                frame=frame,
                landmarks=landmarks,
                num_faces=num_faces
            )
            if is_occ_raw:
                is_occluded = True
                occ_reason = raw_code
                occ_msg = raw_msg

        if is_occluded:
            return {
                "has_face": bool(has_face),
                "num_faces": num_faces,
                "is_occluded": True,
                "occlusion_reason": occ_reason,
                "same_person": True,
                "passed": False,
                "timed_out": False,
                "time_left": round(blink_time_left, 1),
                "timeout_seconds": CHALLENGE_TIMEOUT_SECONDS,
                "error": occ_msg or "CẢNH BÁO: Phát hiện che mặt hoặc đeo kính! Vui lòng tháo ra để chớp mắt.",
                "label": "⚠️ PHÁT HIỆN CHE MẶT",
                "ear_left": round(ear_l, 4),
                "ear_right": round(ear_r, 4),
                "ear_avg": round(ear_avg, 4),
                "baseline_ear": round(baseline_ear, 4),
                "closed_thresh": 0.18,
                "open_thresh": 0.21,
                "blink_counter": current_blink_counter,
                "blink_state": current_blink_state,
                "progress": 0.0
            }

        # 2.3 KIỂM TRA ĐỘ KHỚP KHUNG OVAL (OVAL FIT CHECK TRÊN FRAME HIỆN TẠI)
        fit_info = check_face_oval_fit(
            landmarks, w, h, oval_center, oval_axes,
            min_ratio=OVAL_FIT_MIN_RATIO * 0.90,  # 0.36: Dung sai nhẹ nhàng khi người dùng chớp mắt
            max_ratio=OVAL_FIT_MAX_RATIO * 1.10,  # 0.66
            min_face_height=int(MIN_FACE_HEIGHT * 0.90)  # ~130px
        )
        if not fit_info["fit_oval"]:
            # Nếu mặt bị lùi ra quá xa, lệch tâm hoặc ngoài oval:
            # QUAN TRỌNG: Trả về same_person = True, fit_oval = False để KHÔNG báo nhầm là đổi người!
            if fit_info["is_too_far"]:
                label_msg = "⚠️ TIẾN LẠI GẦN CHO KHỚP OVAL"
                err_msg = "Khuôn mặt ở quá xa! Vui lòng tiến lại gần khung oval để tiếp tục."
            elif fit_info["is_off_center"]:
                label_msg = "⚠️ ĐƯA MẶT VÀO GIỮA OVAL"
                err_msg = fit_info["off_center_hint"] or "Vui lòng giữ khuôn mặt ở giữa khung oval."
            elif fit_info["is_too_close"]:
                label_msg = "⚠️ LÙI RA XA MỘT CHÚT"
                err_msg = "Khuôn mặt quá gần! Vui lòng lùi lại một chút cho vừa khung oval."
            else:
                label_msg = "⚠️ ĐƯA MẶT VÀO KHUNG OVAL"
                err_msg = "Vui lòng giữ khuôn mặt trong khung oval."

            return {
                "has_face": bool(has_face),
                "num_faces": num_faces,
                "fit_oval": False,
                "is_too_far": bool(fit_info["is_too_far"]),
                "is_too_close": bool(fit_info["is_too_close"]),
                "is_off_center": bool(fit_info["is_off_center"]),
                "same_person": True,
                "passed": False,
                "timed_out": False,
                "time_left": round(blink_time_left, 1),
                "timeout_seconds": CHALLENGE_TIMEOUT_SECONDS,
                "error": err_msg,
                "label": label_msg,
                "ear_left": round(ear_l, 4),
                "ear_right": round(ear_r, 4),
                "ear_avg": round(ear_avg, 4),
                "baseline_ear": round(baseline_ear, 4),
                "closed_thresh": 0.18,
                "open_thresh": 0.21,
                "blink_counter": current_blink_counter,
                "blink_state": current_blink_state,
                "progress": 0.0
            }

        # 2.5 Kiểm tra nhận dạng khuôn mặt liên tục (Continuous Face Identity Consistency Check)
        # Bất kỳ frame nào trong Giai đoạn 2 có mặt người khác -> Báo động đổi người ngay lập tức!
        base_desc = None
        if session_id:
            if session_id in self.liveness_sessions:
                sess = self.liveness_sessions[session_id]
                base_desc = sess.get("base_desc")
                sess["last_activity"] = time.time()
            elif base_frame_input is not None:
                try:
                    base_img = load_image(base_frame_input)
                    base_desc = self.identity_verifier.extract_descriptor(
                        base_img, oval_center=oval_center, oval_axes=oval_axes, filter_oval=True
                    )
                except Exception:
                    base_desc = None
            else:
                return {
                    "has_face": True,
                    "num_faces": 1,
                    "same_person": False,
                    "passed": False,
                    "timed_out": False,
                    "time_left": round(blink_time_left, 1),
                    "timeout_seconds": CHALLENGE_TIMEOUT_SECONDS,
                    "error": "Phiên xác thực không tồn tại hoặc đã hết hạn! Vui lòng làm lại từ Bước 1.",
                    "label": "⚠️ PHIÊN HẾT HẠN (EXPIRED SESSION)",
                    "ear_left": round(ear_l, 4),
                    "ear_right": round(ear_r, 4),
                    "ear_avg": round(ear_avg, 4),
                    "baseline_ear": round(baseline_ear, 4),
                    "closed_thresh": 0.18,
                    "open_thresh": 0.21,
                    "blink_counter": current_blink_counter,
                    "blink_state": False,
                    "progress": 0.0
                }
        elif base_frame_input is not None:
            try:
                base_img = load_image(base_frame_input)
                base_desc = self.identity_verifier.extract_descriptor(
                    base_img, oval_center=oval_center, oval_axes=oval_axes, filter_oval=True
                )
            except Exception:
                base_desc = None

        same_person = True
        identity_details = None
        if base_desc is not None and has_face:
            cand_desc = self.identity_verifier.extract_descriptor(
                frame, oval_center=oval_center, oval_axes=oval_axes, filter_oval=True
            )
            if cand_desc is not None:
                same_person, match_score, identity_details = self.identity_verifier.verify_identity(
                    base_desc, cand_desc, max_disparity_thresh=0.025, min_cosine_thresh=0.920
                )
                if not same_person:
                    return {
                        "has_face": True,
                        "num_faces": 1,
                        "fit_oval": True,
                        "same_person": False,
                        "passed": False,
                        "timed_out": False,
                        "time_left": round(blink_time_left, 1),
                        "timeout_seconds": CHALLENGE_TIMEOUT_SECONDS,
                        "error": "CẢNH BÁO: Phát hiện đổi người! Yêu cầu đúng người chụp ảnh ban đầu thực hiện thử thách.",
                        "label": "⚠️ PHÁT HIỆN ĐỔI NGƯỜI (MISMATCH)",
                        "identity_details": identity_details,
                        "ear_left": round(ear_l, 4),
                        "ear_right": round(ear_r, 4),
                        "ear_avg": round(ear_avg, 4),
                        "baseline_ear": round(baseline_ear, 4),
                        "closed_thresh": 0.18,
                        "open_thresh": 0.21,
                        "blink_counter": current_blink_counter,
                        "blink_state": False,
                        "progress": 0.0
                    }

        new_counter = current_blink_counter
        new_state = current_blink_state
        updated_baseline = baseline_ear

        # 3. Cập nhật Baseline EAR thích ứng khi mắt mở
        if not new_state and ear_avg >= 0.18:
            if updated_baseline <= 0.05:
                updated_baseline = ear_avg
            else:
                if ear_avg >= updated_baseline * 0.85:
                    updated_baseline = updated_baseline * 0.90 + ear_avg * 0.10
        elif updated_baseline <= 0.05 and ear_avg > 0.14:
            updated_baseline = ear_avg

        # 4. Ngưỡng chớp mắt chuẩn xác theo test_pipeline_ensemble_full.py:
        #    - Nhắm mắt: 0.04 < ear_avg < 0.18 (hoặc sụt giảm > 18% so với baseline)
        #    - Mở lại: ear_avg >= 0.21 (hoặc phục hồi >= 88% baseline)
        closed_thresh = 0.18
        open_thresh = 0.21

        is_closed = (0.04 < ear_avg < closed_thresh) or (updated_baseline > 0.18 and ear_avg <= updated_baseline * 0.82)
        is_open = (ear_avg >= open_thresh) or (updated_baseline > 0.18 and ear_avg >= updated_baseline * 0.88 and ear_avg >= 0.18)

        # 5. State Machine: MẮT MỞ -> MẮT NHẮM (new_state = True) -> MẮT MỞ LẠI (new_counter += 1)
        if is_closed:
            if not new_state:
                new_state = True
        elif is_open and new_state:
            new_counter += 1
            new_state = False

        passed = (new_counter >= MIN_BLINKS_REQUIRED)

        # 6. Ghi nhận hoàn tất chớp mắt vào session
        if passed and session_id and session_id in self.liveness_sessions:
            self.liveness_sessions[session_id]["blink_passed"] = True
            self.liveness_sessions[session_id]["blink_frame"] = frame.copy()

        progress = 1.0 if passed else (0.5 if new_state else 0.0)
        label = (
            "Đã xác nhận chớp mắt (100%)"
            if passed
            else ("Đang chớp mắt... (50%)" if new_state else "Đang đợi chớp mắt... (0%)")
        )

        return {
            "has_face": has_face,
            "num_faces": num_faces,
            "fit_oval": True,
            "ratio_to_oval": fit_info["ratio_to_oval"],
            "same_person": True,
            "timed_out": False,
            "time_left": round(blink_time_left, 1),
            "timeout_seconds": CHALLENGE_TIMEOUT_SECONDS,
            "ear_left": round(ear_l, 4),
            "ear_right": round(ear_r, 4),
            "ear_avg": round(ear_avg, 4),
            "baseline_ear": round(updated_baseline, 4),
            "closed_thresh": round(closed_thresh, 4),
            "open_thresh": round(open_thresh, 4),
            "blink_counter": new_counter,
            "blink_state": bool(new_state),
            "passed": bool(passed),
            "progress": progress,
            "label": label
        }

    def start_head_challenge(self) -> Dict[str, Any]:
        """Khởi tạo một thử thách quay đầu ngẫu nhiên mới."""
        self._ensure_models_loaded()
        action = self.head_movement_detector.start_challenge()
        prompt = self.head_movement_detector.get_prompt()
        return {
            "action": action.value,
            "prompt": prompt,
            "timeout_seconds": CHALLENGE_TIMEOUT_SECONDS
        }

    def update_head_challenge(
        self,
        frame_input: Union[str, bytes, np.ndarray],
        session_id: Optional[str] = None,
        base_frame_input: Optional[Union[str, bytes, np.ndarray]] = None
    ) -> Dict[str, Any]:
        """Cập nhật frame cho thử thách quay đầu, kiểm tra 1 người duy nhất và kiểm tra nhận dạng cùng người chụp Bước 1."""
        self._ensure_models_loaded()
        frame = load_image(frame_input)
        h, w = frame.shape[:2]

        oval_center, oval_axes = get_default_oval_params(w, h)
        head_oval_axes = (int(oval_axes[0] * 1.25), oval_axes[1])

        # 0. Trích xuất landmarks và đếm số khuôn mặt trong 1 lần suy luận duy nhất (Single-Pass MediaPipe ~20ms)
        # Chỉ nhận mặt trong khung oval và bỏ đi các mặt ngoài oval
        landmarks, num_faces = self.landmark_detector.detect_with_count(
            frame, oval_center=oval_center, oval_axes=head_oval_axes, filter_oval=True
        )
        if num_faces > 1:
            return {
                "state": "FAILED",
                "action": "",
                "passed": False,
                "num_faces": num_faces,
                "same_person": False,
                "prompt": "VUI LÒNG CHỈ 1 NGƯỜI ĐỨNG TRƯỚC CAMERA",
                "error": f"Phát hiện {num_faces} người trong khung hình (Yêu cầu 1 người duy nhất)",
                "time_left": 0.0,
                "progress": 0.0,
                "current_angle": 0.0,
                "target_threshold": 0.0,
                "is_matched": False,
                "pose": {"yaw": 0.0, "pitch": 0.0, "roll": 0.0}
            }

        # 0.5 Kiểm tra độ khớp khung Oval (Oval Fit Check cho Head Challenge)
        fit_info = check_face_oval_fit(
            landmarks, w, h, oval_center, head_oval_axes,
            min_ratio=OVAL_FIT_MIN_RATIO * 0.85,  # 0.34: dung sai góc quay đầu
            max_ratio=OVAL_FIT_MAX_RATIO * 1.15,  # 0.69
            min_face_height=int(MIN_FACE_HEIGHT * 0.85)
        )
        if not fit_info["fit_oval"] and fit_info["is_too_far"]:
            action_name = self.head_movement_detector.current_action.value if self.head_movement_detector.current_action else "TURN_HEAD"
            return {
                "state": "WARNING",
                "action": action_name,
                "passed": False,
                "num_faces": num_faces,
                "fit_oval": False,
                "is_too_far": True,
                "same_person": True,
                "prompt": "⚠️ TIẾN LẠI GẦN CHO KHỚP OVAL",
                "error": "Khuôn mặt ở quá xa! Vui lòng tiến lại gần khung oval để tiếp tục.",
                "time_left": 10.0,
                "progress": 0.0,
                "current_angle": 0.0,
                "target_threshold": 3.5,
                "is_matched": False,
                "pose": {"yaw": 0.0, "pitch": 0.0, "roll": 0.0}
            }

        # 1. Kiểm tra nhận dạng khuôn mặt (Face Identity Consistency Check) - CHỈ trong cùng 1 phiên
        base_desc = None
        if session_id:
            if session_id in self.liveness_sessions:
                sess = self.liveness_sessions[session_id]
                base_desc = sess.get("base_desc")
                sess["last_activity"] = time.time()
            elif base_frame_input is not None:
                try:
                    base_img = load_image(base_frame_input)
                    base_desc = self.identity_verifier.extract_descriptor(
                        base_img, oval_center=oval_center, oval_axes=head_oval_axes, filter_oval=True
                    )
                except Exception:
                    base_desc = None
            else:
                return {
                    "state": "FAILED",
                    "action": "",
                    "passed": False,
                    "num_faces": 1,
                    "same_person": False,
                    "prompt": "⚠️ CẢNH BÁO: PHIÊN HẾT HẠN!",
                    "error": "Phiên xác thực không tồn tại hoặc đã hết hạn! Vui lòng làm lại từ Bước 1.",
                    "time_left": 0.0,
                    "progress": 0.0,
                    "current_angle": 0.0,
                    "target_threshold": 0.0,
                    "is_matched": False,
                    "pose": {"yaw": 0.0, "pitch": 0.0, "roll": 0.0}
                }
        elif base_frame_input is not None:
            try:
                base_img = load_image(base_frame_input)
                base_desc = self.identity_verifier.extract_descriptor(
                    base_img, oval_center=oval_center, oval_axes=head_oval_axes, filter_oval=True
                )
            except Exception:
                base_desc = None

        # 1. Ước lượng tư thế 3D Pose trước để hỗ trợ bù trừ góc quay khi thẩm định danh tính
        pose_dict = None
        if landmarks and len(landmarks) >= 468:
            _, _, pose_dict = self.pose_validator.validate(landmarks, get_landmark_point, img_w=w, img_h=h)

        pose_angles = None
        if pose_dict:
            pose_angles = (
                float(pose_dict.get("yaw", 0.0)),
                float(pose_dict.get("pitch", 0.0)),
                float(pose_dict.get("roll", 0.0))
            )

        # 1.1 Kiểm tra nhận dạng khuôn mặt (Face Identity Consistency Check) với bù trừ góc quay 2 trục
        same_person = True
        identity_details = None
        if base_desc is not None and landmarks and len(landmarks) >= 468:
            cand_desc = self.identity_verifier.extract_descriptor(
                frame, oval_center=oval_center, oval_axes=head_oval_axes, filter_oval=True
            )
            if cand_desc is not None:
                same_person, match_score, identity_details = self.identity_verifier.verify_identity(
                    base_desc, cand_desc,
                    pose_angles=pose_angles,
                    is_head_challenge=True
                )
                if not same_person:
                    # Tích lũy đếm mismatch liên tiếp để chống rớt frame do chuyển động nhanh hoặc góc quét phức tạp
                    mismatch_count = 1
                    if sess is not None:
                        sess["head_mismatch_count"] = sess.get("head_mismatch_count", 0) + 1
                        mismatch_count = sess["head_mismatch_count"]

                    action_name = self.head_movement_detector.current_action.value if self.head_movement_detector.current_action else "TURN_HEAD"
                    pose_out = {
                        "yaw": round(pose_dict.get("yaw", 0.0), 1),
                        "pitch": round(pose_dict.get("pitch", 0.0), 1),
                        "roll": round(pose_dict.get("roll", 0.0), 1)
                    } if pose_dict else {"yaw": 0.0, "pitch": 0.0, "roll": 0.0}

                    # Nếu mới lệch 1-2 frame: Giữ ở trạng thái WARNING và nhắc nhở, không vội hủy phiên
                    if mismatch_count < 3:
                        return {
                            "state": "WARNING",
                            "action": action_name,
                            "passed": False,
                            "num_faces": 1,
                            "same_person": False,
                            "prompt": "⚠️ VUI LÒNG GIỮ KHUÔN MẶT RÕ RÀNG TRONG OVAL",
                            "error": "Hình ảnh khuôn mặt biến đổi nhanh. Vui lòng căn chỉnh mặt trong khung oval.",
                            "identity_details": identity_details,
                            "time_left": 10.0,
                            "progress": 0.0,
                            "current_angle": round(pose_out.get("yaw", 0.0), 1),
                            "target_threshold": 3.0,
                            "is_matched": False,
                            "pose": pose_out
                        }
                    else:
                        # Mismatch liên tiếp >= 3 frame: Xác nhận đổi người và hủy phiên
                        return {
                            "state": "FAILED",
                            "action": "",
                            "passed": False,
                            "num_faces": 1,
                            "same_person": False,
                            "prompt": "⚠️ CẢNH BÁO: PHÁT HIỆN ĐỔI NGƯỜI!",
                            "error": "CẢNH BÁO: Phát hiện đổi người! Yêu cầu đúng người chụp ảnh ban đầu thực hiện thử thách.",
                            "identity_details": identity_details,
                            "time_left": 0.0,
                            "progress": 0.0,
                            "current_angle": 0.0,
                            "target_threshold": 0.0,
                            "is_matched": False,
                            "pose": pose_out
                        }
                else:
                    if sess is not None:
                        sess["head_mismatch_count"] = 0

        # 1.5 Kiểm tra che mặt (Face Occlusion Defense)
        # Nếu phát hiện che mặt: TUYỆT ĐỐI KHÔNG ĐƯỢC TÍNH GÓC QUAY HEAD YAW HAY TĂNG TIẾN TRÌNH!
        is_occluded, occ_reason, occ_msg = self.occlusion_detector.check_occlusion(
            frame=frame,
            landmarks=landmarks,
            num_faces=num_faces,
            pose_dict=pose_dict
        )
        if is_occluded:
            action_name = self.head_movement_detector.current_action.value if self.head_movement_detector.current_action else "TURN_HEAD"
            return {
                "state": "WARNING",
                "action": action_name,
                "passed": False,
                "num_faces": num_faces,
                "is_occluded": True,
                "occlusion_reason": occ_reason,
                "same_person": True,
                "prompt": "⚠️ CẢNH BÁO: KHÔNG ĐƯỢC CHE MẶT!",
                "error": occ_msg or "CẢNH BÁO: Phát hiện che mặt! Vui lòng không che mặt khi quay đầu.",
                "time_left": 10.0,
                "progress": 0.0,        # KHÔNG ĐỔI TIẾN TRÌNH
                "current_angle": 0.0,   # KHÔNG ĐỔI GÓC QUAY HEAD YAW
                "target_threshold": 3.5,
                "is_matched": False,
                "pose": {"yaw": 0.0, "pitch": 0.0, "roll": 0.0}
            }

        status = self.head_movement_detector.update(pose_dict)
        clean_status = {
            "state": str(status.get("state", "")),
            "action": str(status.get("action", "")),
            "passed": bool(status.get("passed", False)),
            "prompt": str(status.get("prompt", "")),
            "time_left": float(status.get("time_left", 0.0)),
            "progress": float(status.get("progress", 0.0)),
            "current_angle": float(status.get("current_angle", 0.0)),
            "target_threshold": float(status.get("target_threshold", 0.0)),
            "is_matched": bool(status.get("is_matched", False)),
            "same_person": True,
            "num_faces": num_faces
        }
        if pose_dict:
            clean_status["pose"] = {
                "yaw": round(float(pose_dict.get("yaw", 0.0)), 2),
                "pitch": round(float(pose_dict.get("pitch", 0.0)), 2),
                "roll": round(float(pose_dict.get("roll", 0.0)), 2),
            }
        else:
            clean_status["pose"] = {"yaw": 0.0, "pitch": 0.0, "roll": 0.0}

        if clean_status["passed"] and session_id and session_id in self.liveness_sessions:
            self.liveness_sessions[session_id]["head_passed"] = True
            self.liveness_sessions[session_id]["head_frame"] = frame.copy()

        return clean_status

    # =========================================================================
    # 4. QUY TRÌNH TOÀN DIỆN — FULL VERIFY PIPELINE (ENSEMBLE)
    # =========================================================================
    def full_verify(
        self,
        image_input: Union[str, bytes, np.ndarray],
        img_id: Union[int, str] = 1,
        blink_passed: bool = True,
        blink_count: int = 1,
        head_movement_passed: bool = True,
        head_action_name: str = "TURN_LEFT",
        output_dir: Optional[str] = None,
        save_visuals: bool = True,
        apply_oval_mask: bool = True,
        session_id: Optional[str] = None,
        blink_frame_input: Optional[Union[str, bytes, np.ndarray]] = None,
        head_frame_input: Optional[Union[str, bytes, np.ndarray]] = None
    ) -> Dict[str, Any]:
        """
        Thực thi toàn diện quy trình kiểm tra eKYC tĩnh & tổng hợp quyết định:
        - Khung Oval hướng dẫn & làm mờ bối cảnh xung quanh (Bokeh Effect)
        - Face Detection & trích xuất khuôn mặt chính trong Oval
        - MediaPipe 478 Landmark Detection
        - 3D Head Pose Validation
        - Face Alignment & 224x224 Crop
        - ENSEMBLE Anti-Spoofing (YOLO_4 + RF-DETR Small) với IoU Matching
        - Tổng hợp quyết định cuối cùng (7 tiêu chí chuẩn ngân hàng)
        - Lưu artifacts nếu có output_dir
        """
        self._ensure_models_loaded()
        frame = load_image(image_input)
        h_f, w_f = frame.shape[:2]

        # Tọa độ khung Oval trung tâm
        oval_center, oval_axes = get_default_oval_params(w_f, h_f)

        # 0. Làm mờ bối cảnh ngoại vi trừ khung oval nếu apply_oval_mask=True
        if apply_oval_mask:
            processed_frame = get_oval_masked_frame(frame, oval_center, oval_axes, blur_ksize=45, dim_factor=0.35)
        else:
            processed_frame = frame

        # 1. Face Detection (chạy trên processed_frame đã làm mờ ngoại vi để triệt tiêu người thứ 2 ngoài oval)
        detection_input = processed_frame if apply_oval_mask else frame
        raw_faces = self.detector.detect(detection_input, conf=CONF_THRESHOLD_FACE)

        # Chỉ nhận các khuôn mặt nằm trong oval và loại bỏ hoàn toàn các mặt ngoài oval (tolerance=1.0)
        faces_in_oval = filter_faces_in_oval(raw_faces, oval_center, oval_axes, img_w=w_f, img_h=h_f, tolerance=1.0)
        faces = faces_in_oval
        significant_faces = []
        for f in faces_in_oval:
            bx1, by1, bx2, by2 = f["bbox"]
            bw = bx2 - bx1
            bh = by2 - by1
            if bh >= 40 and bw >= 40:
                significant_faces.append(f)
        num_faces = len(significant_faces) if significant_faces else len(faces_in_oval)

        # Chọn Primary Face
        primary_face = None
        if faces:
            def get_face_priority(f):
                bx1, by1, bx2, by2 = f["bbox"]
                area = (bx2 - bx1) * (by2 - by1)
                cx, cy = (bx1 + bx2) / 2.0, (by1 + by2) / 2.0
                dist_center = math.hypot(cx - oval_center[0], cy - oval_center[1])
                return area - (dist_center * 10)

            primary_face = max(faces, key=get_face_priority)

        # Kiểm tra mặt có trong oval
        face_in_oval = bool(primary_face and is_face_in_oval(primary_face["bbox"], oval_center, oval_axes, tolerance=1.0))

        # 2. Landmarks (chạy trên detection_input để triệt tiêu người thứ 2, fallback frame tự nhiên)
        landmarks = self.landmark_detector.detect(
            detection_input, oval_center=oval_center, oval_axes=oval_axes, filter_oval=True, oval_tolerance=1.0
        )
        if not landmarks and detection_input is not frame:
            landmarks = self.landmark_detector.detect(
                frame, oval_center=oval_center, oval_axes=oval_axes, filter_oval=True, oval_tolerance=1.0
            )

        # 3. 3D Pose
        pose_valid = False
        pose_dict = None
        pose_msg = "No Face"
        if landmarks:
            pose_valid, pose_msg, pose_dict = self.pose_validator.validate(landmarks, get_landmark_point, img_w=w_f, img_h=h_f)

        # 4. Face Crop & Align
        aligned_img = None
        face_crop_224 = None
        if primary_face is not None:
            face_crop_224 = self.aligner.crop_face(
                frame,
                bbox=primary_face["bbox"],
                padding=25,
                output_size=(224, 224),
                mode="bbox"
            )
        elif landmarks:
            face_crop_224 = self.aligner.crop_face(
                frame,
                landmarks=landmarks,
                padding=25,
                output_size=(224, 224)
            )

        if landmarks:
            aligned_img = self.aligner.align_face(frame, landmarks)

        if aligned_img is None:
            aligned_img = frame.copy()

        # 4.5 Đánh giá Fail-Fast: Kiểm tra các tiêu chí cơ bản trước khi chạy Ensemble Anti-Spoofing nặng
        face_position_ok = face_in_oval if apply_oval_mask else True
        c_face = (primary_face is not None) and face_position_ok
        c_single = (num_faces == 1)
        c_pose = pose_valid
        c_blink = bool(blink_passed)
        c_head = bool(head_movement_passed)

        # Kiểm tra che mặt sơ bộ
        is_occluded_final, occ_reason_final, occ_msg_final = self.occlusion_detector.check_occlusion(
            frame=frame,
            landmarks=landmarks,
            num_faces=num_faces,
            pose_dict=pose_dict
        )
        c_occlusion_free = not is_occluded_final

        # Tổng hợp lỗi tiền đề (Fail-Fast Gate: Sai 1 bước là REJECT ngay, không chạy tiếp model nặng)
        reasons = []
        if primary_face is None:
            reasons.append("Không tìm thấy khuôn mặt trong ảnh")
        elif apply_oval_mask and not face_in_oval:
            reasons.append("Khuôn mặt nằm ngoài khung oval hướng dẫn")
        elif not c_single:
            reasons.append(f"Phát hiện {num_faces} người trong khung hình (Yêu cầu 1 người duy nhất)")

        if not c_pose:
            reasons.append(f"Góc mặt ảnh chụp bị nghiêng/lệch ({pose_msg})")

        if is_occluded_final:
            reasons.append(occ_msg_final or f"Khuôn mặt bị che khuất hoặc che một phần ({occ_reason_final})!")

        if not c_blink:
            reasons.append("Chưa hoàn thành chớp mắt (Blink)")
        if not c_head:
            reasons.append("Chưa hoàn thành cử động đầu (Head Movement)")

        has_prior_failure = len(reasons) > 0

        # Khởi tạo giá trị mặc định cho Anti-Spoofing & Identity
        all_ensemble_dets, yolo_dets, rfdetr_dets = [], [], []
        ens_latency_ms = 0.0
        best_spoof = None
        primary_spoof_iou = 0.0
        has_any_spoof = False
        is_primary_real = False
        c_both_detected = False
        c_spoof = False
        c_same_person = True
        identity_details = {}

        if not has_prior_failure:
            # 5. ENSEMBLE Anti-Spoofing (YOLO_4 + RF-DETR Small)
            # CHỈ chạy suy luận mô hình nặng khi TẤT CẢ các bước tiền đề đều ĐẠT CHUẨN!
            if check_illumination_quality is not None and primary_face:
                captured_light = check_illumination_quality(frame, bbox=primary_face["bbox"])
                if captured_light.get("mean_luminance", 100.0) < 75.0 and enhance_low_light is not None:
                    input_spoof = enhance_low_light(frame)
                else:
                    input_spoof = frame
            else:
                input_spoof = frame

            t_ens = time.time()
            all_ensemble_dets, yolo_dets, rfdetr_dets = self.ensemble_anti_spoof.predict_ensemble(
                input_spoof,
                conf_threshold=ENSEMBLE_CONF_THRESHOLD,
                iou_thresh=ENSEMBLE_IOU_THRESHOLD,
                w_yolo=ENSEMBLE_W_YOLO,
                w_rfdetr=ENSEMBLE_W_RFDETR,
                strict_spoof_veto=True,
            )
            ens_latency_ms = (time.time() - t_ens) * 1000

            # Lọc spoof detections trong oval (lược bỏ hoàn toàn detections ngoài oval)
            spoofs_in_oval = [sd for sd in all_ensemble_dets if is_face_in_oval(sd["bbox"], oval_center, oval_axes)]
            target_spoofs = spoofs_in_oval

            # Tìm detection khớp nhất với Primary Face
            if primary_face and target_spoofs:
                matching_spoofs = [
                    sd for sd in target_spoofs
                    if calculate_iou(primary_face["bbox"], sd["bbox"]) > 0.15
                ]
                if matching_spoofs:
                    best_spoof = max(matching_spoofs, key=lambda x: x["confidence"])
                    primary_spoof_iou = calculate_iou(primary_face["bbox"], best_spoof["bbox"])
                else:
                    best_spoof = max(target_spoofs, key=lambda x: x["confidence"])
                    primary_spoof_iou = calculate_iou(primary_face["bbox"], best_spoof["bbox"])

            if best_spoof is None and target_spoofs:
                best_spoof = target_spoofs[0]

            has_any_spoof = any(not sd["is_real"] for sd in target_spoofs) if target_spoofs else False
            is_primary_real = bool(best_spoof["is_real"]) if best_spoof else False
            c_both_detected = bool(best_spoof.get("both_detected", False)) if best_spoof else False
            c_spoof = is_primary_real

            if best_spoof is None:
                reasons.append("Không phát hiện được đặc trưng chống giả mạo (Anti-Spoof None)")
            elif not c_both_detected:
                reasons.append(
                    f"Chỉ có 1 model nhận diện ({best_spoof.get('source')}) "
                    f"— Lược bỏ ảnh (Thiếu sự đồng thuận cả 2 model)"
                )
            elif not c_spoof:
                reasons.append(f"Phát hiện giả mạo (SPOOF) với độ tin cậy {best_spoof['confidence']*100:.1f}%")

            # 5.5 Kiểm tra tính liên tục danh tính khuôn mặt giữa các bước (Biometric Face Identity Continuity)
            if session_id and session_id in self.liveness_sessions:
                sess = self.liveness_sessions[session_id]
                base_desc = sess.get("base_desc")
                if base_desc is not None:
                    # 1. Kiểm tra ảnh chụp chớp mắt nếu có trong cùng phiên
                    bf = sess.get("blink_frame")
                    if bf is None and blink_frame_input is not None:
                        try:
                            bf = load_image(blink_frame_input)
                        except Exception:
                            bf = None

                    if bf is not None:
                        cand_b = self.identity_verifier.extract_descriptor(
                            bf, oval_center=oval_center, oval_axes=oval_axes, filter_oval=True
                        )
                        if cand_b is not None:
                            is_same_b, _, dt_b = self.identity_verifier.verify_identity(base_desc, cand_b)
                            identity_details["blink_match"] = dt_b
                            if not is_same_b:
                                c_same_person = False

                    # 2. Kiểm tra ảnh chụp quay đầu nếu có trong cùng phiên
                    hf = sess.get("head_frame")
                    if hf is None and head_frame_input is not None:
                        try:
                            hf = load_image(head_frame_input)
                        except Exception:
                            hf = None

                    if hf is not None:
                        head_oval_axes = (int(oval_axes[0] * 1.25), oval_axes[1])
                        cand_h = self.identity_verifier.extract_descriptor(
                            hf, oval_center=oval_center, oval_axes=head_oval_axes, filter_oval=True
                        )
                        if cand_h is not None:
                            is_same_h, _, dt_h = self.identity_verifier.verify_identity(base_desc, cand_h)
                            identity_details["head_match"] = dt_h
                            if not is_same_h:
                                c_same_person = False

                del self.liveness_sessions[session_id]
                if not c_same_person:
                    reasons.append("Phát hiện tráo đổi người thực hiện thử thách (Face Identity Mismatch)!")
        else:
            # Xóa session khỏi bộ nhớ nếu fail-fast
            if session_id and session_id in self.liveness_sessions:
                del self.liveness_sessions[session_id]

        final_pass = bool(
            c_face and c_single and c_pose and c_spoof
            and c_both_detected and c_blink and c_head and c_same_person
            and c_occlusion_free
        )

        # Xây dựng kết quả chi tiết
        result_report = {
            "image_id": img_id,
            "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
            "final_decision": {
                "approved": final_pass,
                "verdict": "APPROVED" if final_pass else "REJECTED",
                "reasons": reasons
            },
            "criteria": {
                "face_detected": bool(c_face),
                "single_face": bool(c_single),
                "pose_valid": bool(c_pose),
                "anti_spoof_real": bool(c_spoof),
                "both_models_detected": bool(c_both_detected),
                "blink_passed": bool(c_blink),
                "head_movement_passed": bool(c_head),
                "same_person_verified": bool(c_same_person),
                "no_face_occlusion": bool(c_occlusion_free)
            },
            "identity_verification": {
                "same_person_verified": bool(c_same_person),
                "details": identity_details
            },
            "face_detection": {
                "num_faces": num_faces,
                "single_face": bool(num_faces == 1),
                "primary_face": {
                    "bbox": primary_face["bbox"] if primary_face else None,
                    "confidence": round(float(primary_face["confidence"]), 4) if primary_face else 0.0
                } if primary_face else None
            },
            "pose_3d": {
                "is_valid": pose_valid,
                "message": pose_msg,
                "yaw": round(float(pose_dict["yaw"]), 2) if pose_dict else 0.0,
                "pitch": round(float(pose_dict["pitch"]), 2) if pose_dict else 0.0,
                "roll": round(float(pose_dict["roll"]), 2) if pose_dict else 0.0
            },
            "ensemble_anti_spoof": {
                "label": best_spoof["label"] if best_spoof else ("SKIPPED (FAIL-FAST)" if has_prior_failure else "NO_DATA"),
                "is_real": is_primary_real,
                "confidence": round(float(best_spoof["confidence"]), 4) if best_spoof else 0.0,
                "primary_iou": round(float(primary_spoof_iou), 4),
                "has_any_spoof_in_frame": bool(has_any_spoof),
                "source": best_spoof.get("source") if best_spoof else ("Fail-Fast Early Reject" if has_prior_failure else "NONE"),
                "yolo_detail": best_spoof.get("yolo_res") if best_spoof else ("Skipped (Không thực hiện do bước trước lỗi)" if has_prior_failure else "N/A"),
                "rfdetr_detail": best_spoof.get("rfdetr_res") if best_spoof else ("Skipped (Không thực hiện do bước trước lỗi)" if has_prior_failure else "N/A"),
                "agreement": best_spoof.get("agreement") if best_spoof else False,
                "both_detected": bool(c_both_detected),
                "latency_ms": round(ens_latency_ms, 1),
                "models_used": {
                    "model_1": "YOLO_4 (Anti_Spoof_YOLO_4.pt)",
                    "model_2": "RF-DETR Small (Transformer)"
                },
                "all_ensemble_detections": [
                    {
                        "bbox": sd["bbox"],
                        "is_real": sd["is_real"],
                        "label": sd["label"],
                        "confidence": round(float(sd["confidence"]), 4),
                        "source": sd.get("source", ""),
                        "yolo_res": sd.get("yolo_res", "N/A"),
                        "rfdetr_res": sd.get("rfdetr_res", "N/A"),
                        "both_detected": sd.get("both_detected", False),
                    }
                    for sd in all_ensemble_dets
                ]
            },
            "active_liveness": {
                "blink_passed": bool(blink_passed),
                "blink_count": int(blink_count),
                "head_movement_passed": bool(head_movement_passed),
                "head_action": str(head_action_name)
            },
            "oval_guide": {
                "center": [oval_center[0], oval_center[1]],
                "axes": [oval_axes[0], oval_axes[1]],
                "face_in_oval": bool(face_in_oval)
            }
        }

        # 7. Lưu file kết quả nếu được chỉ định thư mục output
        if output_dir and save_visuals:
            sess_out_dir = os.path.join(output_dir, str(img_id))
            os.makedirs(sess_out_dir, exist_ok=True)

            # Lưu all_faces_cropped
            all_faces_dir = os.path.join(sess_out_dir, "all_faces_cropped")
            os.makedirs(all_faces_dir, exist_ok=True)
            for idx_f, f_it in enumerate(faces, 1):
                bx1, by1, bx2, by2 = f_it["bbox"]
                cx1, cy1 = max(0, min(w_f - 1, bx1)), max(0, min(h_f - 1, by1))
                cx2, cy2 = max(0, min(w_f, bx2)), max(0, min(h_f, by2))
                c_img = frame[cy1:cy2, cx1:cx2]
                if c_img.size > 0:
                    cv2.imwrite(os.path.join(all_faces_dir, f"face_{idx_f}.jpg"), c_img)

            # 1: Ảnh khuôn mặt annotated sạch (Làm mờ bối cảnh ngoại vi trừ khung Oval)
            if apply_oval_mask:
                clean_img = get_oval_masked_frame(frame, oval_center, oval_axes, blur_ksize=45, dim_factor=0.35)
                guide_color = (0, 255, 127) if final_pass else (0, 0, 255)
                clean_img = draw_oval_face_guide(
                    clean_img,
                    center=oval_center,
                    axes=oval_axes,
                    is_aligned=final_pass,
                    is_detected=bool(primary_face),
                    color=guide_color
                )
            else:
                clean_img = frame.copy()
            for f_it in faces:
                bx1, by1, bx2, by2 = f_it["bbox"]
                is_p = (primary_face and f_it["bbox"] == primary_face["bbox"])
                box_c = (0, 255, 0) if is_p else (200, 200, 200)
                cv2.rectangle(clean_img, (bx1, by1), (bx2, by2), box_c, 2 if is_p else 1)
                lbl_tag = "PRIMARY FACE" if is_p else "EXTRA FACE"
                cv2.putText(clean_img, lbl_tag, (bx1, max(18, by1 - 6)),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.45, box_c, 1, cv2.LINE_AA)

            if landmarks:
                clean_img = draw_landmarks(clean_img, landmarks)

            # Vẽ Ensemble detection boxes
            for sd in all_ensemble_dets:
                sx1, sy1, sx2, sy2 = sd["bbox"]
                if not sd.get("both_detected", False):
                    scol = (0, 165, 255)   # Cam: Thiếu đồng thuận
                    tag = f"DISCARD: 1 Model ({sd['confidence']*100:.1f}%)"
                elif sd["is_real"]:
                    scol = (0, 255, 0)     # Xanh: Real
                    tag = f"REAL {sd['confidence']*100:.1f}% (Ensemble)"
                else:
                    scol = (0, 0, 255)     # Đỏ: Spoof
                    tag = f"SPOOF {sd['confidence']*100:.1f}% (Ensemble)"

                cv2.rectangle(clean_img, (sx1, sy1), (sx2, sy2), scol, 2)
                cv2.putText(clean_img, tag, (sx1, max(20, sy1 - 8)),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.45, scol, 1, cv2.LINE_AA)
                sub_tag = f"YOLO: {sd.get('yolo_res', 'N/A')} | RF: {sd.get('rfdetr_res', 'N/A')}"
                cv2.putText(clean_img, sub_tag, (sx1, sy2 + 16),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.38, (220, 220, 220), 1, cv2.LINE_AA)

            verdict_badge = "eKYC: APPROVED" if final_pass else "eKYC: REJECTED"
            badge_col = (0, 255, 0) if final_pass else (0, 0, 255)
            badge_w = 240
            cv2.rectangle(clean_img, (w_f - badge_w - 15, 12), (w_f - 15, 52), (15, 18, 24), -1)
            cv2.rectangle(clean_img, (w_f - badge_w - 15, 12), (w_f - 15, 52), badge_col, 2)
            cv2.putText(clean_img, verdict_badge, (w_f - badge_w - 2, 38),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.58, badge_col, 2, cv2.LINE_AA)

            # 2: Dashboard Panel
            dashboard_img = create_pipeline_result_dashboard(
                img_idx=img_id,
                face_info=primary_face,
                num_faces=num_faces,
                pose_info=pose_dict,
                pose_valid=pose_valid,
                anti_spoof_info=best_spoof,
                spoof_iou=primary_spoof_iou,
                blink_passed=blink_passed,
                blink_count=blink_count,
                head_movement_passed=head_movement_passed,
                head_action_name=head_action_name,
                final_pass=final_pass,
                reasons=reasons,
                face_crop=face_crop_224,
                target_height=h_f
            )

            # 3: Side-by-Side
            side_by_side_img = create_side_by_side_result(clean_img, dashboard_img)

            # Lưu file
            cv2.imwrite(os.path.join(sess_out_dir, "1_pipeline_result_clean.jpg"), clean_img)
            cv2.imwrite(os.path.join(sess_out_dir, "1_dashboard_panel.jpg"), dashboard_img)
            cv2.imwrite(os.path.join(sess_out_dir, "1_pipeline_side_by_side.jpg"), side_by_side_img)
            cv2.imwrite(os.path.join(sess_out_dir, "1_pipeline_result.jpg"), side_by_side_img)

            if face_crop_224 is not None:
                cv2.imwrite(os.path.join(sess_out_dir, "2_face_crop_224.jpg"), face_crop_224)

            cv2.imwrite(os.path.join(sess_out_dir, "0_raw_image.jpg"), frame)

            if aligned_img is not None:
                cv2.imwrite(os.path.join(sess_out_dir, "3_aligned_full.jpg"), aligned_img)

            # 4_report.json
            report_file_path = os.path.join(sess_out_dir, "4_report.json")
            with open(report_file_path, "w", encoding="utf-8") as f_rep:
                json.dump(result_report, f_rep, ensure_ascii=False, indent=2, default=json_serialize_helper)

            # Cập nhật batch_summary_ensemble.csv
            batch_csv_path = os.path.join(output_dir, "batch_summary_ensemble.csv")
            csv_exists = os.path.exists(batch_csv_path)
            with open(batch_csv_path, "a", newline="", encoding="utf-8-sig") as f_csv:
                writer = csv.writer(f_csv)
                if not csv_exists:
                    writer.writerow([
                        "Image ID", "Verdict", "Num Faces",
                        "Ensemble Label", "Ensemble Conf",
                        "YOLO Detail", "RF-DETR Detail",
                        "Both Detected", "Agreement",
                        "IoU", "Pose Valid",
                        "Blink", "Head Movement",
                        "Latency (ms)", "Reasons", "Output Folder"
                    ])
                ens_info = result_report["ensemble_anti_spoof"]
                writer.writerow([
                    f"{img_id}.jpg" if not str(img_id).endswith(".jpg") else str(img_id),
                    result_report["final_decision"]["verdict"],
                    num_faces,
                    ens_info["label"],
                    ens_info["confidence"],
                    ens_info["yolo_detail"],
                    ens_info["rfdetr_detail"],
                    "YES" if ens_info["both_detected"] else "NO",
                    "YES" if ens_info["agreement"] else "NO",
                    f"{primary_spoof_iou:.2f}",
                    "PASS" if pose_valid else "FAIL",
                    "PASS" if blink_passed else "FAIL",
                    f"PASS ({head_action_name})" if head_movement_passed else f"FAIL ({head_action_name})",
                    f"{ens_latency_ms:.1f}",
                    "; ".join(reasons) if reasons else "None",
                    sess_out_dir
                ])

        return result_report
