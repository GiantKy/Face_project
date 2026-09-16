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
        ChallengeState
    )
    from .config import (
        FACE_DETECTION_MODEL_PATH,
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
        EAR_EYE_CLOSED_THRESHOLD,
        EAR_EYE_OPEN_THRESHOLD,
        MIN_BLINKS_REQUIRED,
        HEAD_YAW_THRESHOLD,
        HEAD_PITCH_THRESHOLD,
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
        remove_vietnamese_accents
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
        ChallengeState
    )
    from config import (
        FACE_DETECTION_MODEL_PATH,
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
        EAR_EYE_CLOSED_THRESHOLD,
        EAR_EYE_OPEN_THRESHOLD,
        MIN_BLINKS_REQUIRED,
        HEAD_YAW_THRESHOLD,
        HEAD_PITCH_THRESHOLD,
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
        remove_vietnamese_accents
    )


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
        self.rfdetr_model_id = rfdetr_model_id or RFDETR_MODEL_ID
        self.rfdetr_api_key = rfdetr_api_key or RFDETR_API_KEY

        self.detector: Optional[FaceDetector] = None
        self.landmark_detector: Optional[LandmarkDetector] = None
        self.pose_validator: Optional[PoseValidator] = None
        self.aligner: Optional[FaceAligner] = None
        self.ensemble_anti_spoof: Optional[EnsembleAntiSpoofDetector] = None
        self.head_movement_detector: Optional[HeadMovementDetector] = None

        if not lazy_load:
            self.load_models()

    def load_models(self):
        """Khởi tạo và tải trước toàn bộ mô hình AI vào bộ nhớ."""
        print("[EKYCPipelineServer] Đang khởi tạo các mô hình AI (Ensemble Edition)...")
        self.detector = FaceDetector(model_path=self.face_model_path)
        self.landmark_detector = LandmarkDetector()
        self.pose_validator = PoseValidator()
        self.aligner = FaceAligner()
        self.ensemble_anti_spoof = EnsembleAntiSpoofDetector(
            yolo_model_path=self.yolo_antispoof_path,
            rfdetr_model_id=self.rfdetr_model_id,
            rfdetr_api_key=self.rfdetr_api_key,
        )
        self.head_movement_detector = HeadMovementDetector(
            yaw_threshold=HEAD_YAW_THRESHOLD,
            pitch_threshold=HEAD_PITCH_THRESHOLD,
            timeout=CHALLENGE_TIMEOUT_SECONDS
        )
        print("[EKYCPipelineServer] Tải toàn bộ AI Models thành công! (Ensemble Ready)\n")

    def _ensure_models_loaded(self):
        if self.detector is None:
            self.load_models()

    # =========================================================================
    # 1. KIỂM TRA TƯ THẾ & CĂN CHỈNH KHUÔN MẶT (PRE-CAPTURE CHECK)
    # =========================================================================
    def validate_pose(self, image_input: Union[str, bytes, np.ndarray]) -> Dict[str, Any]:
        """
        Kiểm tra tư thế khuôn mặt (trước khi chụp):
        - Phát hiện khuôn mặt và landmarks
        - Đánh giá khoảng cách camera (kích thước mặt)
        - Đánh giá 3 góc Euler (Yaw, Pitch, Roll)
        
        Returns:
            Dict chứa trạng thái pose, góc quay và thông báo hướng dẫn.
        """
        self._ensure_models_loaded()
        frame = load_image(image_input)
        h, w = frame.shape[:2]

        landmarks = self.landmark_detector.detect(frame)
        if not landmarks:
            return {
                "has_face": False,
                "is_valid": False,
                "face_size_h": 0,
                "is_too_far": True,
                "pose": {"yaw": 0.0, "pitch": 0.0, "roll": 0.0},
                "message": "Không tìm thấy khuôn mặt trong khung hình",
                "guide": "Vui lòng đưa khuôn mặt vào giữa khung hình"
            }

        # Đánh giá kích thước khuôn mặt
        ys = [p[1] for p in landmarks]
        face_size_h = max(ys) - min(ys)
        is_too_far = (face_size_h < MIN_FACE_HEIGHT)

        # Đánh giá góc nghiêng 3D Pose
        pose_valid, text_status, pose_dict = self.pose_validator.validate(landmarks, get_landmark_point)
        pose_data = pose_dict if pose_dict else {"yaw": 0.0, "pitch": 0.0, "roll": 0.0}

        is_valid_overall = (pose_valid and not is_too_far)

        if is_too_far:
            guide_msg = "Vui lòng tiến lại gần camera hơn (Khuôn mặt quá nhỏ)"
        elif not pose_valid:
            guide_msg = f"Vui lòng nhìn thẳng vào camera ({text_status})"
        else:
            guide_msg = "Tư thế khuôn mặt đạt chuẩn!"

        return {
            "has_face": True,
            "is_valid": bool(is_valid_overall),
            "face_size_h": int(face_size_h),
            "is_too_far": bool(is_too_far),
            "pose": {
                "yaw": round(float(pose_data.get("yaw", 0.0)), 2),
                "pitch": round(float(pose_data.get("pitch", 0.0)), 2),
                "roll": round(float(pose_data.get("roll", 0.0)), 2),
                "status_text": text_status
            },
            "message": text_status,
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

        # 1. Phát hiện khuôn mặt
        faces = self.detector.detect(frame)
        num_faces = len(faces)

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

        # Tìm Primary Face: Lớn nhất và gần tâm nhất
        def get_face_priority(f):
            bx1, by1, bx2, by2 = f["bbox"]
            area = (bx2 - bx1) * (by2 - by1)
            cx, cy = (bx1 + bx2) / 2.0, (by1 + by2) / 2.0
            dist_center = math.hypot(cx - w_f / 2.0, cy - h_f / 2.0)
            return area - (dist_center * 10)

        primary_face = max(faces, key=get_face_priority)

        # 2. Căn chỉnh và crop 224x224
        landmarks = self.landmark_detector.detect(frame)
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

        # Tìm detection khớp nhất với Primary Face
        best_spoof = None
        primary_spoof_iou = 0.0
        if primary_face and all_ensemble_dets:
            matching_spoofs = [
                sd for sd in all_ensemble_dets
                if calculate_iou(primary_face["bbox"], sd["bbox"]) > 0.15
            ]
            if matching_spoofs:
                best_spoof = max(matching_spoofs, key=lambda x: x["confidence"])
                primary_spoof_iou = calculate_iou(primary_face["bbox"], best_spoof["bbox"])
            else:
                best_spoof = max(all_ensemble_dets, key=lambda x: x["confidence"])
                primary_spoof_iou = calculate_iou(primary_face["bbox"], best_spoof["bbox"])

        if best_spoof is None and all_ensemble_dets:
            best_spoof = all_ensemble_dets[0]

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
        current_blink_state: bool = False
    ) -> Dict[str, Any]:
        """Đo lường chỉ số EAR trên frame và cập nhật trạng thái chớp mắt."""
        self._ensure_models_loaded()
        frame = load_image(frame_input)
        landmarks = self.landmark_detector.detect(frame)

        ear_l, ear_r, ear_avg = compute_eye_aspect_ratio(landmarks) if landmarks else (0.0, 0.0, 0.0)

        new_counter = current_blink_counter
        new_state = current_blink_state

        if ear_avg > 0.05 and ear_avg < EAR_EYE_CLOSED_THRESHOLD:
            if not new_state:
                new_state = True
        elif ear_avg >= EAR_EYE_OPEN_THRESHOLD:
            if new_state:
                new_counter += 1
                new_state = False

        passed = (new_counter >= MIN_BLINKS_REQUIRED)

        return {
            "ear_left": round(ear_l, 4),
            "ear_right": round(ear_r, 4),
            "ear_avg": round(ear_avg, 4),
            "blink_counter": new_counter,
            "blink_state": new_state,
            "passed": bool(passed)
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

    def update_head_challenge(self, frame_input: Union[str, bytes, np.ndarray]) -> Dict[str, Any]:
        """Cập nhật frame cho thử thách quay đầu hiện tại."""
        self._ensure_models_loaded()
        frame = load_image(frame_input)
        landmarks = self.landmark_detector.detect(frame)

        pose_dict = None
        if landmarks:
            _, _, pose_dict = self.pose_validator.validate(landmarks, get_landmark_point)

        status = self.head_movement_detector.update(pose_dict)
        return status

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
        save_visuals: bool = True
    ) -> Dict[str, Any]:
        """
        Thực thi toàn diện quy trình kiểm tra eKYC tĩnh & tổng hợp quyết định:
        - Face Detection & trích xuất khuôn mặt chính
        - MediaPipe 478 Landmark Detection
        - 3D Head Pose Validation
        - Face Alignment & 224x224 Crop
        - ENSEMBLE Anti-Spoofing (YOLO_4 + RF-DETR Small) với IoU Matching
        - Tổng hợp quyết định cuối cùng (7 tiêu chí: Face, Single, Pose, Spoof, BothModels, Blink, Head)
        - Lưu artifacts nếu có output_dir
        """
        self._ensure_models_loaded()
        frame = load_image(image_input)
        h_f, w_f = frame.shape[:2]

        # 1. Face Detection
        faces = self.detector.detect(frame)
        num_faces = len(faces)

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

        # 2. Landmarks
        landmarks = self.landmark_detector.detect(frame)

        # 3. 3D Pose
        pose_valid = False
        pose_dict = None
        pose_msg = "No Face"
        if landmarks:
            pose_valid, pose_msg, pose_dict = self.pose_validator.validate(landmarks, get_landmark_point)

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

        # 5. ENSEMBLE Anti-Spoofing (YOLO_4 + RF-DETR Small)
        t_ens = time.time()
        all_ensemble_dets, yolo_dets, rfdetr_dets = self.ensemble_anti_spoof.predict_ensemble(
            frame,
            conf_threshold=ENSEMBLE_CONF_THRESHOLD,
            iou_thresh=ENSEMBLE_IOU_THRESHOLD,
            w_yolo=ENSEMBLE_W_YOLO,
            w_rfdetr=ENSEMBLE_W_RFDETR,
            strict_spoof_veto=True,
        )
        ens_latency_ms = (time.time() - t_ens) * 1000

        # Tìm detection khớp nhất với Primary Face
        best_spoof = None
        primary_spoof_iou = 0.0
        if primary_face and all_ensemble_dets:
            matching_spoofs = [
                sd for sd in all_ensemble_dets
                if calculate_iou(primary_face["bbox"], sd["bbox"]) > 0.15
            ]
            if matching_spoofs:
                best_spoof = max(matching_spoofs, key=lambda x: x["confidence"])
                primary_spoof_iou = calculate_iou(primary_face["bbox"], best_spoof["bbox"])
            else:
                best_spoof = max(all_ensemble_dets, key=lambda x: x["confidence"])
                primary_spoof_iou = calculate_iou(primary_face["bbox"], best_spoof["bbox"])

        if best_spoof is None and all_ensemble_dets:
            best_spoof = all_ensemble_dets[0]

        has_any_spoof = any(not sd["is_real"] for sd in all_ensemble_dets) if all_ensemble_dets else False
        is_primary_real = bool(best_spoof["is_real"]) if best_spoof else False

        # 6. Đánh giá Final Decision — 7 tiêu chí
        c_face = (primary_face is not None)
        c_single = (num_faces == 1)
        c_pose = pose_valid
        c_spoof = is_primary_real
        # Tiêu chí mới: Cả 2 model phải đồng thuận
        c_both_detected = bool(best_spoof.get("both_detected", False)) if best_spoof else False
        c_blink = blink_passed
        c_head = head_movement_passed

        reasons = []
        if not c_face:
            reasons.append("Không tìm thấy khuôn mặt trong ảnh")
        elif not c_single:
            reasons.append(f"Phát hiện {num_faces} người trong khung hình (Yêu cầu 1 người duy nhất)")

        if not c_pose:
            reasons.append(f"Góc mặt ảnh chụp bị nghiêng/lệch ({pose_msg})")

        if best_spoof is None:
            reasons.append("Không phát hiện được đặc trưng chống giả mạo (Anti-Spoof None)")
        elif not c_both_detected:
            reasons.append(
                f"Chỉ có 1 model nhận diện ({best_spoof.get('source')}) "
                f"— Lược bỏ ảnh (Thiếu sự đồng thuận cả 2 model)"
            )
        elif not c_spoof:
            reasons.append(f"Phát hiện giả mạo (SPOOF) với độ tin cậy {best_spoof['confidence']*100:.1f}%")

        if not c_blink:
            reasons.append("Chưa hoàn thành chớp mắt (Blink)")
        if not c_head:
            reasons.append("Chưa hoàn thành cử động đầu (Head Movement)")

        final_pass = bool(
            c_face and c_single and c_pose and c_spoof
            and c_both_detected and c_blink and c_head
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
                "head_movement_passed": bool(c_head)
            },
            "face_detection": {
                "num_faces": num_faces,
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
                "label": best_spoof["label"] if best_spoof else "NO_DATA",
                "is_real": is_primary_real,
                "confidence": round(float(best_spoof["confidence"]), 4) if best_spoof else 0.0,
                "primary_iou": round(float(primary_spoof_iou), 4),
                "has_any_spoof_in_frame": bool(has_any_spoof),
                "source": best_spoof.get("source") if best_spoof else "NONE",
                "yolo_detail": best_spoof.get("yolo_res") if best_spoof else "N/A",
                "rfdetr_detail": best_spoof.get("rfdetr_res") if best_spoof else "N/A",
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

            # 1: Ảnh khuôn mặt annotated sạch (không bị che khuất bởi HUD)
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
