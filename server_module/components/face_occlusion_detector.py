"""
Face Occlusion & Mask / Eyeglasses Defense Module (Chính Sách A - Nghiêm Ngặt).
Ngăn chặn gian lận eKYC và đảm bảo tính nguyên vẹn sinh trắc học:
1. MẮT KÍNH (EYEGLASSES & SUNGLASSES - CHÍNH SÁCH A):
   - Bắt buộc tháo TOÀN BỘ mọi loại kính (kính cận trong suốt, kính thuốc, kính râm, kính đen, kính màu).
   - Phát hiện kính râm/kính đen qua tỷ lệ độ sáng thích ứng giữa hốc mắt và da trán.
   - Phát hiện kính cận trong suốt qua phân tích viền cạnh cầu nối sống mũi (Nose Bridge Edge) và gọng thái dương.
   - Phát hiện lóa sáng tròng kính (Specular Glare) che khuất con ngươi.
2. KHẨU TRANG (FACE MASK DEFENSE):
   - Bắt buộc tháo 100% khẩu trang (y tế xanh/trắng/đen, khẩu trang vải, khẩu trang 3D/KN95, khẩu trang màu nude).
   - Phát hiện qua độ lệch màu LAB Delta E thích ứng, lọc dải màu y tế, phân tích tương phản viền môi và kết cấu vải.
3. VẬT CHE CHẮN (OBJECT / HAND OCCLUSION):
   - Phát hiện bàn tay, tờ giấy, điện thoại, bìa cứng che miệng/mắt.
"""

from typing import List, Tuple, Optional, Dict, Any, Union
import numpy as np
import cv2

try:
    from server_module.config import (
        STRICT_GLASSES_POLICY,
        CHECK_CLEAR_GLASSES,
        CHECK_GLASSES_GLARE,
        SUNGLASSES_RATIO_THRESH,
        GLASSES_BRIDGE_EDGE_THRESH,
        MASK_DELTA_E_THRESH,
        MASK_LIP_CONTRAST_THRESH
    )
except ImportError:
    try:
        from config import (
            STRICT_GLASSES_POLICY,
            CHECK_CLEAR_GLASSES,
            CHECK_GLASSES_GLARE,
            SUNGLASSES_RATIO_THRESH,
            GLASSES_BRIDGE_EDGE_THRESH,
            MASK_DELTA_E_THRESH,
            MASK_LIP_CONTRAST_THRESH
        )
    except ImportError:
        STRICT_GLASSES_POLICY = True
        CHECK_CLEAR_GLASSES = True
        CHECK_GLASSES_GLARE = True
        SUNGLASSES_RATIO_THRESH = 0.58
        GLASSES_BRIDGE_EDGE_THRESH = 16.0
        MASK_DELTA_E_THRESH = 28.0
        MASK_LIP_CONTRAST_THRESH = 12.0


class FaceOcclusionDetector:
    """
    Bộ phát hiện che mặt đa tầng độ chính xác cao theo Chính Sách A:
    - Tầng 1: Đánh giá số lượng và sự hiện diện của 468 điểm MediaPipe Landmarks.
    - Tầng 2: Kiểm tra kính râm tối màu & kính màu (Sunglasses / Tinted Glasses).
    - Tầng 3: Kiểm tra lóa sáng tròng kính (Eyeglass Glare / Reflection).
    - Tầng 4: Kiểm tra gọng kính cận trong suốt (Clear Prescription Eyeglasses - Chính Sách A).
    - Tầng 5: Kiểm tra khẩu trang y tế, khẩu trang vải, khẩu trang màu và khẩu trang nude (Mask Defense).
    - Tầng 6: Kiểm tra vật phẳng lì che miệng (Paper / Cardboard / Hand Occlusion).
    """

    def __init__(
        self,
        strict_glasses: bool = STRICT_GLASSES_POLICY,
        check_clear_glasses: bool = CHECK_CLEAR_GLASSES,
        check_glare: bool = CHECK_GLASSES_GLARE,
        sunglasses_ratio: float = SUNGLASSES_RATIO_THRESH,
        bridge_edge_thresh: float = GLASSES_BRIDGE_EDGE_THRESH,
        mask_delta_e: float = MASK_DELTA_E_THRESH,
        mask_lip_contrast: float = MASK_LIP_CONTRAST_THRESH
    ):
        self.strict_glasses = strict_glasses
        self.check_clear_glasses = check_clear_glasses
        self.check_glare = check_glare
        self.sunglasses_ratio = sunglasses_ratio
        self.bridge_edge_thresh = bridge_edge_thresh
        self.mask_delta_e = mask_delta_e
        self.mask_lip_contrast = mask_lip_contrast

    def check_occlusion(
        self,
        frame: np.ndarray,
        landmarks: Optional[List[Tuple[int, int]]],
        num_faces: int = 1,
        yolo_faces: Optional[List[Dict[str, Any]]] = None,
        pose_dict: Optional[Dict[str, float]] = None
    ) -> Tuple[bool, str, str]:
        """
        Kiểm tra khuôn mặt có đang bị che khuất, đeo khẩu trang hoặc đeo mắt kính hay không.

        Args:
            frame: Ảnh BGR gốc nguyên bản (chưa qua CLAHE hay làm mờ).
            landmarks: Danh sách tọa độ pixel (x, y) của 468/478 landmarks.
            num_faces: Số lượng khuôn mặt đếm được từ detector.
            yolo_faces: Danh sách kết quả từ YOLO Face Detector (nếu có).
            pose_dict: Góc quay 3D Euler (yaw, pitch, roll) để điều chỉnh dung sai phối cảnh.

        Returns:
            Tuple[is_occluded, reason_code, message]
            - is_occluded: True nếu phát hiện bị che mặt/đeo kính/đeo khẩu trang, False nếu mặt thông thoáng rõ nét.
            - reason_code: Mã kỹ thuật lý do vi phạm.
            - message: Thông báo hướng dẫn người dùng khắc phục.
        """
        if frame is None or frame.size == 0:
            return True, "EMPTY_FRAME", "Không nhận được khung hình camera"

        # ---------------------------------------------------------------------
        # TẦNG 1: KIỂM TRA SỐ LƯỢNG MẶT VÀ MỐC LANDMARKS
        # ---------------------------------------------------------------------
        if num_faces == 0 or landmarks is None or len(landmarks) < 468:
            return True, "NO_FACE_OR_LANDMARKS", "Không tìm thấy khuôn mặt hoặc mặt bị che hoàn toàn"

        h, w = frame.shape[:2]

        try:
            # Mắt trái
            p_le_out = np.array(landmarks[33], dtype=np.float32)   # Đuôi mắt trái
            p_le_in  = np.array(landmarks[133], dtype=np.float32)  # Khóe mắt trái
            p_le_top = np.array(landmarks[159], dtype=np.float32)  # Mí trên mắt trái
            p_le_bot = np.array(landmarks[145], dtype=np.float32)  # Mí dưới mắt trái

            # Mắt phải
            p_re_out = np.array(landmarks[263], dtype=np.float32)  # Đuôi mắt phải
            p_re_in  = np.array(landmarks[362], dtype=np.float32)  # Khóe mắt phải
            p_re_top = np.array(landmarks[386], dtype=np.float32)  # Mí trên mắt phải
            p_re_bot = np.array(landmarks[374], dtype=np.float32)  # Mí dưới mắt phải

            # Mũi & Sống mũi
            p_nose_tip = np.array(landmarks[4], dtype=np.float32)   # Đầu mũi
            p_bridge_mid = np.array(landmarks[168], dtype=np.float32) # Điểm giữa 2 mắt
            p_bridge_top = np.array(landmarks[6], dtype=np.float32)   # Sống mũi trên

            # Miệng & Môi
            p_lip_t = np.array(landmarks[13], dtype=np.float32)     # Môi trên
            p_lip_b = np.array(landmarks[14], dtype=np.float32)     # Môi dưới
            p_philtrum = np.array(landmarks[164], dtype=np.float32) # Nhân trung
            p_ml = np.array(landmarks[61], dtype=np.float32)        # Khóe môi trái
            p_mr = np.array(landmarks[291], dtype=np.float32)       # Khóe môi phải
            p_chin = np.array(landmarks[152], dtype=np.float32)     # Đáy cằm
            p_forehead = np.array(landmarks[10], dtype=np.float32)  # Tâm trán
        except (IndexError, TypeError):
            return True, "MALFORMED_LANDMARKS", "Dữ liệu mốc khuôn mặt không hoàn chỉnh"

        d_eyes = float(np.linalg.norm(p_le_out - p_re_out))
        h_face = float(np.linalg.norm(p_forehead - p_chin))

        if d_eyes < 18 or h_face < 35:
            return True, "FACE_TOO_SMALL", "Khuôn mặt quá nhỏ hoặc ở quá xa camera"

        # ---------------------------------------------------------------------
        # LẤY MẪU ĐỘ SÁNG & MÀU SẮC DA TRÁN CHUẨN (FOREHEAD BASELINE)
        # ---------------------------------------------------------------------
        fx, fy = int(p_forehead[0]), int(p_forehead[1])
        f_rad_x = max(10, int(d_eyes * 0.16))
        f_rad_y = max(6, int(h_face * 0.06))
        f_crop = frame[max(0, fy - f_rad_y):min(h, fy + f_rad_y), max(0, fx - f_rad_x):min(w, fx + f_rad_x)]

        forehead_mean_l = 100.0
        f_mean_lab = np.array([100.0, 128.0, 128.0], dtype=np.float32)
        f_mean_hsv = np.array([0.0, 0.0, 100.0], dtype=np.float32)

        if f_crop.size >= 16:
            f_lab = cv2.cvtColor(f_crop, cv2.COLOR_BGR2LAB)
            f_mean_lab = np.mean(f_lab, axis=(0, 1))
            forehead_mean_l = float(f_mean_lab[0])
            f_hsv = cv2.cvtColor(f_crop, cv2.COLOR_BGR2HSV)
            f_mean_hsv = np.mean(f_hsv, axis=(0, 1))

        # ---------------------------------------------------------------------
        # TẦNG 2: KIỂM TRA KÍNH RÂM / KÍNH ĐEN / KÍNH MÀU (SUNGLASSES DETECTION)
        # ---------------------------------------------------------------------
        eye_l_values = []
        eye_crops = []
        for (p_out, p_in, p_top, p_bot) in [
            (p_le_out, p_le_in, p_le_top, p_le_bot),
            (p_re_in, p_re_out, p_re_top, p_re_bot)
        ]:
            ex1 = max(0, int(min(p_out[0], p_in[0]) - d_eyes * 0.05))
            ex2 = min(w, int(max(p_out[0], p_in[0]) + d_eyes * 0.05))
            ey1 = max(0, int(min(p_top[1], p_bot[1]) - d_eyes * 0.08))
            ey2 = min(h, int(max(p_top[1], p_bot[1]) + d_eyes * 0.08))

            if (ex2 - ex1) >= 6 and (ey2 - ey1) >= 6:
                ecrop = frame[ey1:ey2, ex1:ex2]
                eye_crops.append(ecrop)
                elab = cv2.cvtColor(ecrop, cv2.COLOR_BGR2LAB)
                eye_l_values.append(float(np.mean(elab[:, :, 0])))

        if len(eye_l_values) == 2:
            mean_eye_l = (eye_l_values[0] + eye_l_values[1]) / 2.0
            eye_ratio = mean_eye_l / max(10.0, forehead_mean_l)

            # Điều kiện 1: Hốc mắt tối bất thường so với da trán (tỷ lệ < 0.58)
            # Điều kiện 2: Cả 2 mắt đều tối sẫm (L < 40.0)
            if (eye_ratio < self.sunglasses_ratio and mean_eye_l < 65.0) or (eye_l_values[0] < 38.0 and eye_l_values[1] < 38.0):
                return True, "SUNGLASSES_DETECTED", "CẢNH BÁO: Phát hiện đeo kính râm/kính đen!"

        # ---------------------------------------------------------------------
        # TẦNG 3: KIỂM TRA LÓA SÁNG TRÒNG KÍNH (EYEGLASS GLARE DETECTION)
        # ---------------------------------------------------------------------
        if self.check_glare and len(eye_crops) == 2:
            for ecrop in eye_crops:
                egray = cv2.cvtColor(ecrop, cv2.COLOR_BGR2GRAY)
                glare_pixels = np.sum(egray >= 238)
                glare_ratio = glare_pixels / float(egray.size)
                if glare_ratio >= 0.10:  # Tròng kính bị lóa chói hơn 10% diện tích mắt
                    return True, "GLASSES_GLARE_DETECTED", "CẢNH BÁO: Tròng kính bị chói lóa ánh sáng!"

        # ---------------------------------------------------------------------
        # TẦNG 4: KIỂM TRA GỌNG KÍNH CẬN TRONG SUỐT (CLEAR EYEGLASS FRAMES - CHÍNH SÁCH A)
        # ---------------------------------------------------------------------
        if self.strict_glasses and self.check_clear_glasses:
            # 1. Quét cầu gọng kính vắt ngang sống mũi (Bridge of Nose ROI giữa 2 mắt)
            bx1 = max(0, int(min(p_le_in[0], p_re_in[0]) + 1))
            bx2 = min(w, int(max(p_le_in[0], p_re_in[0]) - 1))
            by1 = max(0, int(min(p_bridge_top[1], p_bridge_mid[1]) - d_eyes * 0.08))
            by2 = min(h, int(max(p_bridge_top[1], p_bridge_mid[1]) + d_eyes * 0.14))

            if (bx2 - bx1) >= 8 and (by2 - by1) >= 8:
                bridge_crop = frame[by1:by2, bx1:bx2]
                gray_bridge = cv2.cvtColor(bridge_crop, cv2.COLOR_BGR2GRAY)
                # Lọc cạnh ngang (Sobel Y): Cầu gọng kính luôn tạo thành các cạnh ngang sắc nét vắt qua sống mũi
                sobel_y = cv2.Sobel(gray_bridge, cv2.CV_64F, 0, 1, ksize=3)
                bridge_edge_energy = float(np.mean(np.abs(sobel_y)))

                # Canny Edge Density
                canny_bridge = cv2.Canny(gray_bridge, 50, 140)
                bridge_edge_density = float(np.mean(canny_bridge > 0) * 100.0)

                # Da sống mũi trần tự nhiên có độ biến thiên cạnh thấp (edge_energy < 12.0)
                # Khi có gọng kính (nhựa, kim loại, titan), edge_energy > 16.0 hoặc edge_density > 14%
                if bridge_edge_energy >= self.bridge_edge_thresh or bridge_edge_density >= 14.5:
                    return True, "CLEAR_GLASSES_DETECTED", "CẢNH BÁO: Phát hiện đang đeo mắt kính!"

            # 2. Quét viền gọng thái dương 2 bên đuôi mắt (Temple Frame Outer Edge)
            temple_edge_detected = 0
            for p_outer, direction in [(p_le_out, -1), (p_re_out, 1)]:
                tx1 = max(0, int(p_outer[0] - d_eyes * 0.12 if direction == -1 else p_outer[0] - d_eyes * 0.04))
                tx2 = min(w, int(p_outer[0] + d_eyes * 0.04 if direction == -1 else p_outer[0] + d_eyes * 0.12))
                ty1 = max(0, int(p_outer[1] - d_eyes * 0.08))
                ty2 = min(h, int(p_outer[1] + d_eyes * 0.08))
                if (tx2 - tx1) >= 6 and (ty2 - ty1) >= 6:
                    tcrop = cv2.cvtColor(frame[ty1:ty2, tx1:tx2], cv2.COLOR_BGR2GRAY)
                    sobel_x = cv2.Sobel(tcrop, cv2.CV_64F, 1, 0, ksize=3)
                    if float(np.mean(np.abs(sobel_x))) >= 20.0:
                        temple_edge_detected += 1

            if temple_edge_detected >= 2:
                return True, "CLEAR_GLASSES_DETECTED", "CẢNH BÁO: Phát hiện gọng kính bên thái dương!"

        # ---------------------------------------------------------------------
        # TẦNG 5: KIỂM TRA KHẨU TRANG (FACE MASK DEFENSE - ĐA DẠNG MÀU SẮC)
        # ---------------------------------------------------------------------
        # Vùng hạ diện từ chân mũi xuống cằm
        lx1 = max(0, int(min(p_ml[0], p_mr[0]) - d_eyes * 0.10))
        lx2 = min(w, int(max(p_ml[0], p_mr[0]) + d_eyes * 0.10))
        ly1 = max(0, int(p_nose_tip[1] + d_eyes * 0.04))
        ly2 = min(h, int(p_chin[1] - 2))

        if ly2 > ly1 + 8 and lx2 > lx1 + 12:
            low_crop = frame[ly1:ly2, lx1:lx2]
            low_lab = cv2.cvtColor(low_crop, cv2.COLOR_BGR2LAB)
            low_mean_lab = np.mean(low_lab, axis=(0, 1))
            low_hsv = cv2.cvtColor(low_crop, cv2.COLOR_BGR2HSV)
            low_mean_hsv = np.mean(low_hsv, axis=(0, 1))

            delta_e = float(np.linalg.norm(f_mean_lab - low_mean_lab))

            # 1. Khẩu trang màu thông thường (Xanh, xám, trắng, hoa văn): Delta E > 28.0
            if delta_e >= self.mask_delta_e:
                return True, "MASK_COLOR_MISMATCH", "CẢNH BÁO: Phát hiện đeo khẩu trang hoặc vật che miệng! Vui lòng THÁO KHẨU TRANG."

            # 2. Khẩu trang y tế chuyên dụng (Màu xanh dương / xanh ngọc / Cyan / Light Green: Hue 75 - 140)
            if (f_mean_hsv[0] < 30 or f_mean_hsv[0] > 160) and (70 <= low_mean_hsv[0] <= 142) and low_mean_hsv[1] >= 20:
                return True, "MEDICAL_MASK_DETECTED", "CẢNH BÁO: Phát hiện đeo khẩu trang y tế che mặt! Vui lòng THÁO KHẨU TRANG."

            # 3. Khẩu trang tối màu / Đen:
            if low_mean_lab[0] < 42.0 and forehead_mean_l >= 60.0:
                return True, "DARK_MASK_DETECTED", "CẢNH BÁO: Phát hiện khẩu trang tối màu che miệng! Vui lòng THÁO KHẨU TRANG."

            # 4. Khẩu trang màu da / Nude Mask (Kiểm tra mất viền môi & tương phản nhân trung)
            # Trên da thật: môi luôn có sắc hồng/đỏ rõ nét so với nhân trung
            px1 = max(0, int(p_philtrum[0] - d_eyes * 0.06))
            px2 = min(w, int(p_philtrum[0] + d_eyes * 0.06))
            py1 = max(0, int(p_philtrum[1] - d_eyes * 0.04))
            py2 = min(h, int(p_philtrum[1] + d_eyes * 0.04))

            lip_x1 = max(0, int(p_lip_t[0] - d_eyes * 0.06))
            lip_x2 = min(w, int(p_lip_t[0] + d_eyes * 0.06))
            lip_y1 = max(0, int(p_lip_t[1] - d_eyes * 0.03))
            lip_y2 = min(h, int(p_lip_b[1] + d_eyes * 0.03))

            if (px2 - px1) >= 4 and (py2 - py1) >= 4 and (lip_x2 - lip_x1) >= 4 and (lip_y2 - lip_y1) >= 4:
                p_crop_lab = cv2.cvtColor(frame[py1:py2, px1:px2], cv2.COLOR_BGR2LAB)
                lip_crop_lab = cv2.cvtColor(frame[lip_y1:lip_y2, lip_x1:lip_x2], cv2.COLOR_BGR2LAB)
                p_mean = np.mean(p_crop_lab, axis=(0, 1))
                lip_mean = np.mean(lip_crop_lab, axis=(0, 1))
                lip_contrast = float(np.linalg.norm(p_mean - lip_mean))

                # Phân tích độ lệch chuẩn kết cấu hạ diện
                low_gray = cv2.cvtColor(low_crop, cv2.COLOR_BGR2GRAY)
                low_std = float(np.std(low_gray))
                low_contrast = int(np.max(low_gray)) - int(np.min(low_gray))

                # Nếu viền môi phẳng lì tiệp màu hoàn toàn với da nhân trung và kết cấu mịn bất thường
                if lip_contrast < self.mask_lip_contrast and low_std < 5.5 and low_contrast < 22:
                    return True, "NUDE_MASK_DETECTED", "CẢNH BÁO: Phát hiện khẩu trang tiệp màu da che miệng! Vui lòng THÁO KHẨU TRANG."

            # -----------------------------------------------------------------
            # TẦNG 6: KIỂM TRA VẬT PHẲNG CHE MIỆNG (PAPER / CARDBOARD / HAND)
            # -----------------------------------------------------------------
            low_gray = cv2.cvtColor(low_crop, cv2.COLOR_BGR2GRAY)
            low_std = float(np.std(low_gray))
            low_contrast = int(np.max(low_gray)) - int(np.min(low_gray))
            if low_std < 3.2 and low_contrast < 12:
                return True, "OBJECT_COVERING_LOWER_FACE", "CẢNH BÁO: Phát hiện vật cản che kín vùng miệng! Vui lòng bỏ vật cản ra."

        # Biến dạng hình học môi
        w_mouth = float(np.linalg.norm(p_ml - p_mr))
        if w_mouth < 0.12 * d_eyes:
            return True, "MOUTH_COLLAPSED", "CẢNH BÁO: Vùng miệng bị co cụm che khuất! Vui lòng không che mặt."

        return False, "OK", ""
