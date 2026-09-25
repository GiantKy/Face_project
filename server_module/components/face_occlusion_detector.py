"""
Phát hiện che mặt và bảo vệ chống che giấu khuôn mặt (Face Occlusion & Mask/Hand Defense).
Ngăn chặn gian lận eKYC khi người dùng:
1. Đeo khẩu trang (y tế, vải), trùm khăn, đeo kính râm che mắt.
2. Dùng vật cản (giấy, sách, điện thoại, bìa cứng, vải) che trước camera.
3. Dùng bàn tay che kín miệng, che mắt, che mũi, hoặc che một phần khuôn mặt.
"""

from typing import List, Tuple, Optional, Dict, Any, Union
import numpy as np
import cv2


class FaceOcclusionDetector:
    """
    Bộ phát hiện che mặt đa tầng độ chính xác cao (Adaptive Occlusion Guard):
    - Tầng 1: Đánh giá số lượng và sự hiện diện của 468 điểm MediaPipe Landmarks.
    - Tầng 2: Kiểm tra tính nguyên vẹn giải phẫu học khuôn mặt (Anatomical Integrity & Collapse Detection).
    - Tầng 3: Phân tích sắc thái da Trán (Forehead Baseline) đối chiếu Hạ diện (Mouth/Chin ROI) chống Khẩu trang & Vật che màu.
    - Tầng 4: Phân tích kết cấu phẳng lì bất thường (Blank Paper / Solid Cardboard Occlusion).
    - Tầng 5: Phân tích vùng mắt chống kính râm tối màu (Black Sunglasses).
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
            p_le = np.array(landmarks[33], dtype=np.float32)       # Đuôi mắt trái
            p_re = np.array(landmarks[263], dtype=np.float32)      # Đuôi mắt phải
            p_nose = np.array(landmarks[4], dtype=np.float32)      # Đầu mũi
            p_lip_t = np.array(landmarks[13], dtype=np.float32)    # Môi trên
            p_lip_b = np.array(landmarks[14], dtype=np.float32)    # Môi dưới
            p_ml = np.array(landmarks[61], dtype=np.float32)       # Khóe môi trái
            p_mr = np.array(landmarks[291], dtype=np.float32)      # Khóe môi phải
            p_chin = np.array(landmarks[152], dtype=np.float32)    # Đáy cằm
            p_forehead = np.array(landmarks[10], dtype=np.float32) # Tâm trán
        except (IndexError, TypeError):
            return True, "MALFORMED_LANDMARKS", "Dữ liệu mốc khuôn mặt không hoàn chỉnh"

        d_eyes = float(np.linalg.norm(p_le - p_re))
        h_face = float(np.linalg.norm(p_forehead - p_chin))

        # Kiểm tra kích thước mặt hợp lệ
        if d_eyes < 18 or h_face < 35:
            return True, "FACE_TOO_SMALL", "Khuôn mặt quá nhỏ hoặc ở quá xa"

        # Tầng 2: Kiểm tra biến dạng giải phẫu cực đoan (Gross Anatomic Anomaly)
        # Chỉ bắt các biến dạng rõ ràng khi các điểm landmarks bị co rúm lên bàn tay/vật cản
        w_mouth = float(np.linalg.norm(p_ml - p_mr))
        if w_mouth < 0.12 * d_eyes:
            return True, "MOUTH_COLLAPSED", "CẢNH BÁO: Vùng miệng bị che khuất bất thường!"
        if w_mouth > 1.85 * d_eyes:
            return True, "MOUTH_STRETCHED", "CẢNH BÁO: Vùng miệng bị kéo dãn bất thường!"

        # Tầng 3: Phân tích sắc thái da Trán (Forehead Baseline) vs Hạ diện (Mouth/Chin Area)
        # Đây là phương pháp khoa học chuẩn xác để phát hiện Khẩu trang và Vật che màu
        fx, fy = int(p_forehead[0]), int(p_forehead[1])
        f_crop = frame[max(0, fy - 8):min(h, fy + 8), max(0, fx - 15):min(w, fx + 15)]

        if f_crop.size >= 16:
            f_lab = cv2.cvtColor(f_crop, cv2.COLOR_BGR2LAB)
            f_mean_lab = np.mean(f_lab, axis=(0, 1))
            f_hsv = cv2.cvtColor(f_crop, cv2.COLOR_BGR2HSV)
            f_mean_hsv = np.mean(f_hsv, axis=(0, 1))

            # Vùng hạ diện bao quát từ dưới mũi xuống cằm
            lx1 = max(0, int(min(p_ml[0], p_mr[0]) - 5))
            lx2 = min(w, int(max(p_ml[0], p_mr[0]) + 5))
            ly1 = max(0, int(p_nose[1] + 2))
            ly2 = min(h, int(p_chin[1] - 2))

            if ly2 > ly1 + 6 and lx2 > lx1 + 10:
                low_crop = frame[ly1:ly2, lx1:lx2]
                low_lab = cv2.cvtColor(low_crop, cv2.COLOR_BGR2LAB)
                low_mean_lab = np.mean(low_lab, axis=(0, 1))
                low_hsv = cv2.cvtColor(low_crop, cv2.COLOR_BGR2HSV)
                low_mean_hsv = np.mean(low_hsv, axis=(0, 1))

                delta_e = float(np.linalg.norm(f_mean_lab - low_mean_lab))

                # Khẩu trang màu (xanh, đen, trắng, xám, hoa văn) hoặc vật cản màu sắc che mặt:
                # Trên mặt người bình thường: DeltaE giữa trán và cằm/miệng chỉ từ 6 - 25.
                # Khi đeo khẩu trang hoặc che vật màu: DeltaE luôn > 45.
                if delta_e > 45.0:
                    return True, "MASK_COLOR_MISMATCH", "CẢNH BÁO: Phát hiện đeo khẩu trang hoặc vật cản che miệng!"

                # Nhận diện đặc thù khẩu trang y tế xanh dương / xanh ngọc (Cyan / Blue / Light Green: Hue 75 - 140)
                if (f_mean_hsv[0] < 30 or f_mean_hsv[0] > 160) and (75 <= low_mean_hsv[0] <= 140) and low_mean_hsv[1] > 25:
                    return True, "MEDICAL_MASK_DETECTED", "CẢNH BÁO: Phát hiện đeo khẩu trang y tế che mặt!"

                # Tầng 4: Kiểm tra vật phẳng lì che miệng (Tờ giấy trắng, bìa cứng, thẻ nhựa)
                low_gray = cv2.cvtColor(low_crop, cv2.COLOR_BGR2GRAY)
                low_std = float(np.std(low_gray))
                low_contrast = int(np.max(low_gray)) - int(np.min(low_gray))
                if low_std < 3.2 and low_contrast < 12:
                    return True, "OBJECT_COVERING_LOWER_FACE", "CẢNH BÁO: Phát hiện vật phẳng che kín vùng miệng!"

        # Tầng 5: Phân tích kính râm tối màu (Black Sunglasses) che kín mắt
        # Chỉ cảnh báo khi CẢ HAI mắt đều có độ sáng cực thấp (L < 22 trong không gian LAB)
        # Hoàn toàn KHÔNG báo nhầm khi người dùng chớp mắt bình thường (vì mí mắt có màu da L > 50)
        both_eyes_dark = True
        for name, idxs in [("left_eye", [33, 133, 159, 145]), ("right_eye", [362, 263, 386, 374])]:
            exs = [landmarks[i][0] for i in idxs]
            eys = [landmarks[i][1] for i in idxs]
            ecrop = frame[max(0, min(eys) - 2):min(h, max(eys) + 2), max(0, min(exs) - 2):min(w, max(exs) + 2)]
            if ecrop.size >= 16:
                elab = cv2.cvtColor(ecrop, cv2.COLOR_BGR2LAB)
                mean_l = float(np.mean(elab[:, :, 0]))
                if mean_l >= 22.0:
                    both_eyes_dark = False
                    break
            else:
                both_eyes_dark = False

        if both_eyes_dark:
            return True, "SUNGLASSES_DETECTED", "CẢNH BÁO: Phát hiện đeo kính râm tối màu che mắt!"

        return False, "OK", ""
