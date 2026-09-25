"""
Phát hiện che mặt và bảo vệ chống che giấu khuôn mặt (Face Occlusion & Mask/Hand Defense).
Ngăn chặn gian lận eKYC khi người dùng:
1. Dùng bàn tay che miệng, che mắt, che mũi, hoặc che một phần khuôn mặt.
2. Đeo khẩu trang (y tế, vải), trùm khăn, đeo kính râm che mắt.
3. Dùng vật cản (giấy, sách, điện thoại, bìa cứng) che trước camera.
"""

from typing import List, Tuple, Optional, Dict, Any, Union
import numpy as np
import cv2


class FaceOcclusionDetector:
    """
    Bộ phát hiện che mặt đa tầng độ chính xác cao:
    - Tầng 1: Đánh giá số lượng và sự hiện diện của 468 điểm MediaPipe Landmarks.
    - Tầng 2: Tỷ lệ hình học giải phẫu học khuôn mặt (Anatomical Geometry Proportions).
    - Tầng 3: Phân tích độ tương phản, sắc tố môi (Lip Redness / YCrCb Cr) và kết cấu vùng miệng/mũi/mắt.
    - Tầng 4: Đối chiếu đồng thuận với YOLO Face Bounding Box (nếu có).
    """

    def __init__(self):
        pass

    def check_occlusion(
        self,
        frame: np.ndarray,
        landmarks: Optional[List[Tuple[int, int]]],
        num_faces: int = 1,
        yolo_faces: Optional[List[Dict[str, Any]]] = None,
        pose_dict: Optional[Dict[str, float]] = None
    ) -> Tuple[bool, str, str]:
        """
        Kiểm tra khuôn mặt có đang bị che khuất hay không.

        Args:
            frame: Ảnh BGR gốc.
            landmarks: Danh sách tọa độ pixel (x, y) của 468/478 landmarks.
            num_faces: Số lượng khuôn mặt đếm được từ detector.
            yolo_faces: Danh sách kết quả từ YOLO Face Detector (nếu có).
            pose_dict: Góc quay 3D Euler (yaw, pitch, roll) để điều chỉnh dung sai phối cảnh.

        Returns:
            Tuple[is_occluded, reason_code, message]
            - is_occluded: True nếu phát hiện bị che mặt, False nếu mặt thông thoáng rõ nét.
            - reason_code: Mã kỹ thuật lý do che mặt.
            - message: Thông báo hướng dẫn người dùng.
        """
        if frame is None or frame.size == 0:
            return True, "EMPTY_FRAME", "Không nhận được khung hình camera"

        # Tầng 1: Kiểm tra tính hiện diện của khuôn mặt và landmarks
        if num_faces == 0 or landmarks is None or len(landmarks) < 468:
            return True, "NO_FACE_OR_LANDMARKS", "Không tìm thấy khuôn mặt hoặc mặt bị che hoàn toàn"

        h, w = frame.shape[:2]

        # Tọa độ các mốc then chốt
        try:
            p_le = np.array(landmarks[33], dtype=np.float32)      # Đuôi mắt trái
            p_re = np.array(landmarks[263], dtype=np.float32)     # Đuôi mắt phải
            p_nose = np.array(landmarks[4], dtype=np.float32)     # Đầu mũi
            p_lip_t = np.array(landmarks[13], dtype=np.float32)   # Môi trên
            p_lip_b = np.array(landmarks[14], dtype=np.float32)   # Môi dưới
            p_ml = np.array(landmarks[61], dtype=np.float32)      # Khóe môi trái
            p_mr = np.array(landmarks[291], dtype=np.float32)     # Khóe môi phải
            p_chin = np.array(landmarks[152], dtype=np.float32)   # Đáy cằm
            p_forehead = np.array(landmarks[10], dtype=np.float32) # Tâm trán
        except (IndexError, TypeError):
            return True, "MALFORMED_LANDMARKS", "Dữ liệu mốc khuôn mặt không hoàn chỉnh"

        d_eyes = float(np.linalg.norm(p_le - p_re))
        e_mid = (p_le + p_re) / 2.0
        m_mid = (p_lip_t + p_lip_b) / 2.0
        w_mouth = float(np.linalg.norm(p_ml - p_mr))
        d_nose_mouth = float(np.linalg.norm(p_nose - m_mid))
        d_nose_eyes = float(np.linalg.norm(p_nose - e_mid))
        d_mouth_chin = float(np.linalg.norm(m_mid - p_chin))
        h_face = float(np.linalg.norm(p_forehead - p_chin))

        # Kiểm tra kích thước mặt hợp lệ
        if d_eyes < 18 or h_face < 35:
            return True, "FACE_TOO_SMALL", "Khuôn mặt quá nhỏ hoặc bị che khuất một phần"

        # Đánh giá góc nghiêng để nới lỏng dung sai khi quay mặt
        yaw = float(pose_dict.get("yaw", 0.0)) if pose_dict else 0.0
        is_turning = abs(yaw) > 12.0

        # Tầng 2: Tỷ lệ hình học giải phẫu học khuôn mặt (Geometric Consistency)
        ratio_nm_ne = d_nose_mouth / max(1.0, d_nose_eyes)
        ratio_wm_de = w_mouth / max(1.0, d_eyes)
        ratio_mc_nm = d_mouth_chin / max(1.0, d_nose_mouth)

        min_nm_ne = 0.22 if is_turning else 0.28
        max_nm_ne = 2.40 if is_turning else 2.10

        if ratio_nm_ne < min_nm_ne or ratio_nm_ne > max_nm_ne:
            return True, "MOUTH_NOSE_GEOMETRY_ANOMALOUS", "CẢNH BÁO: Phát hiện che miệng hoặc che mũi!"

        if ratio_wm_de < 0.22 or ratio_wm_de > 1.30:
            return True, "MOUTH_WIDTH_ANOMALOUS", "CẢNH BÁO: Phát hiện che một phần miệng!"

        if not is_turning and (ratio_mc_nm < 0.20 or ratio_mc_nm > 3.8):
            return True, "CHIN_MOUTH_GEOMETRY_ANOMALOUS", "CẢNH BÁO: Phát hiện che cằm hoặc che miệng!"

        # Tầng 3: Phân tích vùng miệng (Mouth ROI) - Chống khẩu trang / Bàn tay che miệng
        mx1 = max(0, int(min(p_ml[0], p_mr[0])))
        mx2 = min(w, int(max(p_ml[0], p_mr[0])))
        my1 = max(0, int(min(p_lip_t[1], landmarks[0][1]) - 2))
        my2 = min(h, int(max(p_lip_b[1], landmarks[17][1]) + 2))

        m_crop = frame[my1:my2, mx1:mx2]
        if m_crop.size >= 16:
            m_gray = cv2.cvtColor(m_crop, cv2.COLOR_BGR2GRAY)
            std_m = float(np.std(m_gray))
            contrast_m = int(np.max(m_gray)) - int(np.min(m_gray))

            # Chuyển đổi sang YCrCb để đo lường sắc tố môi tự nhiên (Cr của môi luôn cao hơn da)
            ycrcb = cv2.cvtColor(m_crop, cv2.COLOR_BGR2YCrCb)
            cr = ycrcb[:, :, 1]
            lip_ratio = float(np.sum(cr > 142) / max(1, cr.size))

            # Nếu vùng miệng đồng nhất (khẩu trang màu, vải, giấy)
            if std_m < 6.8 and contrast_m < 22:
                return True, "MOUTH_COVERED_MASK", "CẢNH BÁO: Phát hiện đeo khẩu trang hoặc che miệng!"

            # Nếu dùng bàn tay hoặc vật cản che miệng làm mất đường vân môi tự nhiên
            if lip_ratio < 0.035 and (std_m < 11.5 or contrast_m < 32):
                return True, "MOUTH_COVERED_HAND", "CẢNH BÁO: Phát hiện dùng tay hoặc vật thể che miệng!"

        # Tầng 4: Phân tích vùng hốc mũi (Nostrils ROI) - Chống che mũi
        nx1 = max(0, int(min(landmarks[98][0], landmarks[327][0]) - 3))
        nx2 = min(w, int(max(landmarks[98][0], landmarks[327][0]) + 3))
        ny1 = max(0, int(min(landmarks[2][1], landmarks[4][1]) - 3))
        ny2 = min(h, int(max(landmarks[2][1], landmarks[4][1]) + 3))

        n_crop = frame[ny1:ny2, nx1:nx2]
        if n_crop.size >= 12:
            n_gray = cv2.cvtColor(n_crop, cv2.COLOR_BGR2GRAY)
            n_std = float(np.std(n_gray))
            n_contrast = int(np.max(n_gray)) - int(np.min(n_gray))
            if n_std < 4.5 and n_contrast < 16:
                return True, "NOSE_COVERED", "CẢNH BÁO: Phát hiện che mũi hoặc đeo khẩu trang!"

        # Tầng 5: Phân tích vùng mắt (Eyes ROI) - Chống kính râm tối màu / Bàn tay che mắt
        for name, idxs in [("left_eye", [33, 133, 159, 145]), ("right_eye", [362, 263, 386, 374])]:
            exs = [landmarks[i][0] for i in idxs]
            eys = [landmarks[i][1] for i in idxs]
            ecrop = frame[max(0, min(eys) - 2):min(h, max(eys) + 2), max(0, min(exs) - 2):min(w, max(exs) + 2)]
            if ecrop.size >= 12:
                egray = cv2.cvtColor(ecrop, cv2.COLOR_BGR2GRAY)
                e_contrast = int(np.max(egray)) - int(np.min(egray))
                e_std = float(np.std(egray))
                if e_contrast < 15 and e_std < 5.0:
                    eye_label = "mắt trái" if name == "left_eye" else "mắt phải"
                    return True, f"{name.upper()}_COVERED", f"CẢNH BÁO: Phát hiện che {eye_label} hoặc đeo kính râm!"

        # Tầng 6: Đối chiếu đồng thuận với YOLO Face Detector (nếu có cung cấp)
        if yolo_faces is not None:
            if len(yolo_faces) == 0:
                # MediaPipe thấy landmarks nhưng YOLO không thấy mặt -> Mặt bị che mất cấu trúc tổng thể
                return True, "FACE_OCCLUDED_YOLO", "CẢNH BÁO: Khuôn mặt bị che khuất (YOLO không nhận diện được)!"

        return False, "OK", ""
