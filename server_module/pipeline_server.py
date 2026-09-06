"""
Core E-KYC Pipeline Server Engine.
Chuẩn hóa và đóng gói quy trình xử lý từ test_pipeline_full.py thành Class dịch vụ hoàn chỉnh.
Sử dụng YOLO Face Detection và YOLO Anti-Spoofing.
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
        ANTI_SPOOF_YOLO_MODEL_PATH,
        CONF_THRESHOLD_FACE,
        CONF_THRESHOLD_ANTI_SPOOF,
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
    from .anti_spoof_yolo import AntiSpoofYoloDetector
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
        ANTI_SPOOF_YOLO_MODEL_PATH,
        CONF_THRESHOLD_FACE,
        CONF_THRESHOLD_ANTI_SPOOF,
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
    from anti_spoof_yolo import AntiSpoofYoloDetector
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
    E-KYC Server Pipeline Engine sử dụng YOLO (Thuần xử lý ảnh - Headless Server Mode):
    - CHẾ ĐỘ HOẠT ĐỘNG: Hoàn toàn xử lý trên dữ liệu ảnh (Image file path, Base64, Bytes buffer hoặc NumPy array).
    - KHÔNG SỬ DỤNG WEBCAM: Tuyệt đối KHÔNG gọi cv2.VideoCapture(), KHÔNG mở webcam, KHÔNG dùng GUI (cv2.imshow/waitKey).
    - TƯƠNG THÍCH SERVER: Chạy an toàn 100% trên máy chủ headless (Linux/Ubuntu, Docker Container, Cloud VM, Windows Server).
    
    Quy trình xử lý trên ảnh:
    1. Face Detection: YOLO (Face_Detection.pt) trên ảnh đầu vào.
    2. Face Landmarks: MediaPipe 478 points (face_landmarker.task).
    3. 3D Pose Validation: Perspective-n-Point Euler Angles (Yaw, Pitch, Roll).
    4. Face Alignment & Normalization: Affine Transform (224x224 crop).
    5. Anti-Spoofing: YOLO (Anti_Spoof_YOLO.pt) với IoU matching cho Primary Face.
    6. Active Liveness: Đánh giá chỉ số EAR chớp mắt & theo dõi góc quay đầu từ frame ảnh được gửi lên.
    7. Decision Engine: Đánh giá hợp chuẩn FinTech/eKYC ngân hàng và xuất kết quả JSON / File ảnh.
    """

    def __init__(
        self,
        face_model_path: Optional[str] = None,
        anti_spoof_model_path: Optional[str] = None,
        lazy_load: bool = False
    ):
        self.face_model_path = face_model_path or FACE_DETECTION_MODEL_PATH
        self.anti_spoof_model_path = anti_spoof_model_path or ANTI_SPOOF_YOLO_MODEL_PATH

        self.detector: Optional[FaceDetector] = None
        self.landmark_detector: Optional[LandmarkDetector] = None
        self.pose_validator: Optional[PoseValidator] = None
        self.aligner: Optional[FaceAligner] = None
        self.anti_spoof_detector: Optional[AntiSpoofYoloDetector] = None
        self.head_movement_detector: Optional[HeadMovementDetector] = None

        if not lazy_load:
            self.load_models()

    def load_models(self):
        """Khởi tạo và tải trước toàn bộ mô hình AI vào bộ nhớ."""
        print("[EKYCPipelineServer] Đang khởi tạo các mô hình AI...")
        self.detector = FaceDetector(model_path=self.face_model_path)
        self.landmark_detector = LandmarkDetector()
        self.pose_validator = PoseValidator()
        self.aligner = FaceAligner()
        self.anti_spoof_detector = AntiSpoofYoloDetector(model_path=self.anti_spoof_model_path)
        self.head_movement_detector = HeadMovementDetector(
            yaw_threshold=HEAD_YAW_THRESHOLD,
            pitch_threshold=HEAD_PITCH_THRESHOLD,
            timeout=CHALLENGE_TIMEOUT_SECONDS
        )
        print("[EKYCPipelineServer] Tải toàn bộ AI Models thành công!\n")

    def _ensure_models_loaded(self):
        if self.detector is None:
            self.load_models()

    # =========================================================================
    # 1. KIỂM TRA TƯ THẾ & CĂN CHỈNH KHUÔN MẶT (PRE-CAPTURE CHECK)
    # =========================================================================
    def validate_pose(self, image_input: Union[str, bytes, np.ndarray]) -> Dict[str, Any]:
        """
        Kiểm tra tư thế khuôn mặt theo thời gian thực (trước khi chụp):
        - Phát hiện khuôn mặt và landmarks
        - Đánh giá khoảng cách camera (kích thước mặt)
        - Đánh giá 3 góc Euler (Yaw, Pitch, Roll)
        
        Trả về:
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
    # 2. KIỂM TRA CHỐNG GIẢ MẠO TĨNH (PASSIVE ANTI-SPOOFING VỚI YOLO)
    # =========================================================================
    def check_antispoof(
        self,
        image_input: Union[str, bytes, np.ndarray],
        conf_threshold: float = CONF_THRESHOLD_ANTI_SPOOF
    ) -> Dict[str, Any]:
        """
        Thực hiện kiểm tra chống giả mạo khuôn mặt (Passive Anti-Spoofing):
        - Quét toàn bộ ảnh bằng YOLO Anti-Spoof
        - Khớp IoU với khuôn mặt chính
        - Căn chỉnh Affine và Crop khuôn mặt 224x224
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
                "all_spoof_detections": []
            }

        # Tìm Primary Face: Lớn nhất và gần tâm màn hình nhất
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

        # Fallback crop từ BBox nếu không bắt được landmark
        if face_crop_224 is None and primary_face is not None:
            px1, py1, px2, py2 = primary_face["bbox"]
            px1_c, py1_c = max(0, min(w_f - 1, px1)), max(0, min(h_f - 1, py1))
            px2_c, py2_c = max(0, min(w_f, px2)), max(0, min(h_f, py2))
            raw_crop_p = frame[py1_c:py2_c, px1_c:px2_c]
            if raw_crop_p.size > 0:
                face_crop_224 = cv2.resize(raw_crop_p, (224, 224))

        # 3. Quét Anti-Spoofing YOLO trên toàn bộ ảnh gốc và ghép IoU
        spoof_detections = self.anti_spoof_detector.predict(frame, conf_threshold=conf_threshold)

        best_spoof = None
        primary_spoof_iou = 0.0
        if primary_face and spoof_detections:
            for sd in spoof_detections:
                iou = calculate_iou(primary_face["bbox"], sd["bbox"])
                if iou > primary_spoof_iou:
                    primary_spoof_iou = iou
                    best_spoof = sd

        if best_spoof is None and spoof_detections:
            best_spoof = spoof_detections[0]

        is_real = bool(best_spoof["is_real"]) if best_spoof else False
        label = best_spoof["label"] if best_spoof else "UNKNOWN"
        confidence = float(best_spoof["confidence"]) if best_spoof else 0.0
        has_any_spoof_in_frame = any(not sd["is_real"] for sd in spoof_detections) if spoof_detections else False

        crop_b64 = image_to_base64(face_crop_224) if face_crop_224 is not None else None

        return {
            "has_face": True,
            "num_faces": num_faces,
            "is_real": is_real,
            "label": label,
            "confidence": round(confidence, 4),
            "primary_spoof_iou": round(primary_spoof_iou, 4),
            "has_any_spoof_in_frame": bool(has_any_spoof_in_frame),
            "primary_face": {
                "bbox": primary_face["bbox"],
                "confidence": round(float(primary_face["confidence"]), 4)
            },
            "crop_face_base64": crop_b64,
            "all_spoof_detections": [
                {
                    "bbox": sd["bbox"],
                    "is_real": sd["is_real"],
                    "label": sd["label"],
                    "confidence": round(float(sd["confidence"]), 4)
                }
                for sd in spoof_detections
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
    # 4. QUY TRÌNH TOÀN DIỆN (FULL VERIFY PIPELINE 4)
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
        - YOLO Anti-Spoofing với IoU Matching
        - Tổng hợp quyết định cuối cùng (6 tiêu chí: Face, Single, Pose, Spoof, Blink, Head)
        - Lưu các artifacts: 1_pipeline_result.jpg, 1_pipeline_result_clean.jpg,
          2_face_crop_224.jpg, 3_aligned_full.jpg, 4_report.json (nếu có output_dir)
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

        # 4. Cắt khuôn mặt chính thẳng đứng tự nhiên từ Bounding Box của YOLO trên ảnh gốc
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

        # Căn chỉnh xoay mắt nếu cần ảnh đối soát
        if landmarks:
            aligned_img = self.aligner.align_face(frame, landmarks)

        if aligned_img is None:
            aligned_img = frame.copy()

        # 5. Anti-Spoofing YOLO
        spoof_detections = self.anti_spoof_detector.predict(frame, conf_threshold=CONF_THRESHOLD_ANTI_SPOOF)
        best_spoof = None
        primary_spoof_iou = 0.0
        if primary_face and spoof_detections:
            for sd in spoof_detections:
                iou = calculate_iou(primary_face["bbox"], sd["bbox"])
                if iou > primary_spoof_iou:
                    primary_spoof_iou = iou
                    best_spoof = sd

        if best_spoof is None and spoof_detections:
            best_spoof = spoof_detections[0]

        has_any_spoof_in_frame = any(not sd["is_real"] for sd in spoof_detections) if spoof_detections else False
        is_primary_real = bool(best_spoof["is_real"]) if best_spoof else False

        # 6. Đánh giá Final Decision theo chuẩn 6 tiêu chí
        c_face = (primary_face is not None)
        c_single = (num_faces == 1)
        c_pose = pose_valid
        c_spoof = is_primary_real
        c_blink = blink_passed
        c_head = head_movement_passed

        reasons = []
        if not c_face:
            reasons.append("Không tìm thấy khuôn mặt trong ảnh")
        elif not c_single:
            reasons.append(f"Phát hiện {num_faces} người trong khung hình (Yêu cầu 1 người duy nhất)")

        if not c_pose:
            reasons.append(f"Góc mặt ảnh chụp bị nghiêng/lệch ({pose_msg})")

        if not c_spoof:
            reasons.append("Phát hiện giả mạo Anti-Spoof (Fake/Spoof)")

        if not c_blink:
            reasons.append("Chưa hoàn thành chớp mắt (Blink)")
        if not c_head:
            reasons.append("Chưa hoàn thành cử động đầu (Head Movement)")

        final_pass = bool(c_face and c_single and c_pose and c_spoof and c_blink and c_head)

        # Xây dựng cấu trúc kết quả chi tiết
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
            "anti_spoof_yolo": {
                "label": best_spoof["label"] if best_spoof else "NO_DATA",
                "is_real": is_primary_real,
                "confidence": round(float(best_spoof["confidence"]), 4) if best_spoof else 0.0,
                "primary_iou": round(float(primary_spoof_iou), 4),
                "has_any_spoof_in_frame": bool(has_any_spoof_in_frame)
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

            # 1: Ảnh khuôn mặt sạch đã annotate (Bounding box, landmarks tinh tế, không có bảng HUD che)
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

            if spoof_detections:
                for sd in spoof_detections:
                    sx1, sy1, sx2, sy2 = sd["bbox"]
                    scol = (0, 255, 0) if sd["is_real"] else (0, 0, 255)
                    cv2.rectangle(clean_img, (sx1, sy1), (sx2, sy2), scol, 2)
                    cv2.putText(clean_img, f"{sd['label']} {sd['confidence']*100:.1f}%",
                                (sx1, max(25, sy1 - 10)), cv2.FONT_HERSHEY_SIMPLEX, 0.60, scol, 2)

            verdict_badge = "eKYC: APPROVED" if final_pass else "eKYC: REJECTED"
            badge_col = (0, 255, 0) if final_pass else (0, 0, 255)
            badge_w = 210
            cv2.rectangle(clean_img, (w_f - badge_w - 15, 12), (w_f - 15, 48), (15, 18, 24), -1)
            cv2.rectangle(clean_img, (w_f - badge_w - 15, 12), (w_f - 15, 48), badge_col, 2)
            cv2.putText(clean_img, verdict_badge, (w_f - badge_w - 2, 36),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.55, badge_col, 2, cv2.LINE_AA)

            # 2: Tạo Canvas bảng thông số Dashboard độc lập (Window 2)
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

            # 3: Tạo ảnh ghép 2 Window song song cạnh nhau (Side-by-Side)
            side_by_side_img = create_side_by_side_result(clean_img, dashboard_img)

            # File 1A: 1_pipeline_result_clean.jpg (Khuôn mặt rõ ràng, không bị che khuất)
            cv2.imwrite(os.path.join(sess_out_dir, "1_pipeline_result_clean.jpg"), clean_img)

            # File 1B: 1_dashboard_panel.jpg (Bảng thông số độc lập độ phân giải cao)
            cv2.imwrite(os.path.join(sess_out_dir, "1_dashboard_panel.jpg"), dashboard_img)

            # File 1C: 1_pipeline_side_by_side.jpg (Ghép 2 window cạnh nhau, trực quan)
            cv2.imwrite(os.path.join(sess_out_dir, "1_pipeline_side_by_side.jpg"), side_by_side_img)

            # File 1: 1_pipeline_result.jpg (Mặc định xuất dạng 2 window song song để không che mặt)
            cv2.imwrite(os.path.join(sess_out_dir, "1_pipeline_result.jpg"), side_by_side_img)

            # 2: 2_face_crop_224.jpg
            if face_crop_224 is not None:
                cv2.imwrite(os.path.join(sess_out_dir, "2_face_crop_224.jpg"), face_crop_224)

            # 0: 0_raw_image.jpg (Lưu ảnh gốc đầu vào phục vụ đối soát)
            cv2.imwrite(os.path.join(sess_out_dir, "0_raw_image.jpg"), frame)

            # 3: 3_aligned_full.jpg
            if aligned_img is not None:
                cv2.imwrite(os.path.join(sess_out_dir, "3_aligned_full.jpg"), aligned_img)

            # 4: 4_report.json
            report_file_path = os.path.join(sess_out_dir, "4_report.json")
            with open(report_file_path, "w", encoding="utf-8") as f_rep:
                json.dump(result_report, f_rep, ensure_ascii=False, indent=2, default=json_serialize_helper)

            # 5: Cập nhật file tổng kết batch_summary_v4.csv
            batch_csv_path = os.path.join(output_dir, "batch_summary_v4.csv")
            csv_exists = os.path.exists(batch_csv_path)
            with open(batch_csv_path, "a", newline="", encoding="utf-8-sig") as f_csv:
                writer = csv.writer(f_csv)
                if not csv_exists:
                    writer.writerow([
                        "Image ID", "Verdict", "Num Faces", "Primary Spoof", "Spoof Conf",
                        "IoU", "Pose Valid", "Blink", "Head Movement", "Reasons", "Output Folder"
                    ])
                writer.writerow([
                    f"{img_id}.jpg" if not str(img_id).endswith(".jpg") else str(img_id),
                    result_report["final_decision"]["verdict"],
                    num_faces,
                    best_spoof["label"] if best_spoof else "NONE",
                    best_spoof["confidence"] if best_spoof else 0.0,
                    f"{primary_spoof_iou:.2f}",
                    "PASS" if pose_valid else "FAIL",
                    "PASS" if blink_passed else "FAIL",
                    f"PASS ({head_action_name})" if head_movement_passed else f"FAIL ({head_action_name})",
                    "; ".join(reasons) if reasons else "None",
                    sess_out_dir
                ])

        return result_report
