"""
Utilities module for E-KYC Server Module.
Cung cấp các hàm tiền xử lý ảnh, giải mã Base64/Bytes, tính IoU, tính EAR và vẽ HUD kết quả.
"""

import os
import io
import base64
import math
import unicodedata
from typing import Union, Tuple, List, Optional, Dict, Any
import numpy as np
import cv2


def remove_vietnamese_accents(text: str) -> str:
    """Chuyển đổi văn bản tiếng Việt có dấu thành không dấu để cv2.putText hiển thị không bị lỗi font."""
    if not text:
        return ""
    text = str(text).replace("đ", "d").replace("Đ", "D")
    nfkd = unicodedata.normalize("NFKD", text)
    return "".join([c for c in nfkd if not unicodedata.combining(c)])


def load_image(image_input: Union[str, bytes, np.ndarray]) -> np.ndarray:
    """
    Nạp ảnh đa năng từ nhiều nguồn đầu vào:
    - Đường dẫn file ảnh (str / Path)
    - Chuỗi Base64 (có hoặc không có prefix data:image/...;base64,)
    - Raw bytes buffer
    - Đối tượng cv2 numpy.ndarray sẵn có
    
    Trả về:
        np.ndarray: Ảnh định dạng BGR chuẩn của OpenCV.
    """
    if isinstance(image_input, np.ndarray):
        return image_input.copy()

    if isinstance(image_input, bytes):
        nparr = np.frombuffer(image_input, np.uint8)
        img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
        if img is None:
            raise ValueError("Không thể giải mã mảng bytes thành hình ảnh OpenCV hợp lệ.")
        return img

    if isinstance(image_input, str):
        # 1. Kiểm tra xem có phải chuỗi Base64 không
        if image_input.startswith("data:image") or ";base64," in image_input or len(image_input) > 500:
            raw_base64 = image_input
            if ";base64," in raw_base64:
                raw_base64 = raw_base64.split(";base64,")[1]
            try:
                img_bytes = base64.b64decode(raw_base64)
                nparr = np.frombuffer(img_bytes, np.uint8)
                img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
                if img is not None:
                    return img
            except Exception as e:
                pass

        # 2. Nếu không phải Base64 hoặc giải mã thất bại, thử đọc như đường dẫn file
        if os.path.exists(image_input):
            img = cv2.imread(image_input)
            if img is not None:
                return img
            raise ValueError(f"Không thể đọc file ảnh từ đường dẫn: {image_input}")

        raise ValueError("Đầu vào image_input không phải là đường dẫn file hợp lệ, chuỗi Base64 hoặc bytes.")

    raise TypeError(f"Kiểu dữ liệu {type(image_input)} không được hỗ trợ để nạp ảnh.")


def image_to_base64(image: np.ndarray, ext: str = ".jpg", quality: int = 90) -> str:
    """Chuyển đổi ảnh OpenCV numpy array sang chuỗi Base64."""
    if image is None or image.size == 0:
        return ""
    encode_params = [int(cv2.IMWRITE_JPEG_QUALITY), quality] if ext.lower() in [".jpg", ".jpeg"] else []
    success, buffer = cv2.imencode(ext, image, encode_params)
    if not success:
        return ""
    b64_str = base64.b64encode(buffer).decode("utf-8")
    return f"data:image/jpeg;base64,{b64_str}"


def calculate_iou(boxA: List[int], boxB: List[int]) -> float:
    """Tính Intersection over Union (IoU) giữa 2 bounding box [x1, y1, x2, y2]."""
    xA = max(boxA[0], boxB[0])
    yA = max(boxA[1], boxB[1])
    xB = min(boxA[2], boxB[2])
    yB = min(boxA[3], boxB[3])

    interArea = max(0, xB - xA) * max(0, yB - yA)
    boxAArea = (boxA[2] - boxA[0]) * (boxA[3] - boxA[1])
    boxBArea = (boxB[2] - boxB[0]) * (boxB[3] - boxB[1])

    iou = interArea / float(boxAArea + boxBArea - interArea + 1e-6)
    return float(iou)


def calc_dist(p1: Tuple[float, float], p2: Tuple[float, float]) -> float:
    """Tính khoảng cách Euclidean giữa 2 điểm (x, y)."""
    return math.hypot(p1[0] - p2[0], p1[1] - p2[1])


def compute_eye_aspect_ratio(landmarks: List[Tuple[int, int]]) -> Tuple[float, float, float]:
    """
    Tính EAR (Eye Aspect Ratio) từ danh sách 478 MediaPipe face landmarks.
    Trả về: (ear_left, ear_right, ear_avg)
    """
    if not landmarks or len(landmarks) < 468:
        return 0.0, 0.0, 0.0

    l_top = (calc_dist(landmarks[160], landmarks[144]) + calc_dist(landmarks[158], landmarks[153])) / 2.0
    l_width = calc_dist(landmarks[33], landmarks[133])
    ear_left = (l_top / l_width) if l_width > 0 else 0.0

    r_top = (calc_dist(landmarks[385], landmarks[380]) + calc_dist(landmarks[387], landmarks[373])) / 2.0
    r_width = calc_dist(landmarks[362], landmarks[263])
    ear_right = (r_top / r_width) if r_width > 0 else 0.0

    ear_avg = (ear_left + ear_right) / 2.0
    return float(ear_left), float(ear_right), float(ear_avg)


def json_serialize_helper(obj: Any) -> Any:
    """Chuyển đổi kiểu dữ liệu numpy/OpenCV sang Python native types để xuất JSON sạch."""
    if isinstance(obj, (np.bool_, bool)):
        return bool(obj)
    if isinstance(obj, (np.integer, int)):
        return int(obj)
    if isinstance(obj, (np.floating, float)):
        return float(obj)
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    return str(obj)


def draw_ui_card(image: np.ndarray, x: int, y: int, w: int, h: int, bg_color=(15, 15, 20), alpha=0.85):
    """Vẽ khung card bán trong suốt làm nền HUD."""
    overlay = image.copy()
    cv2.rectangle(overlay, (x, y), (x + w, y + h), bg_color, -1)
    cv2.addWeighted(overlay, alpha, image, 1 - alpha, 0, image)
    cv2.rectangle(image, (x, y), (x + w, y + h), (100, 100, 100), 1)


def draw_pipeline_result_hud(
    image: np.ndarray,
    img_idx: Any,
    face_info: Optional[Dict[str, Any]],
    num_faces: int,
    pose_info: Optional[Dict[str, Any]],
    pose_valid: bool,
    anti_spoof_info: Optional[Dict[str, Any]],
    spoof_iou: float,
    blink_passed: bool,
    blink_count: int,
    head_movement_passed: bool,
    head_action_name: str,
    final_pass: bool,
    reasons: List[str]
) -> np.ndarray:
    """Vẽ bảng HUD kết quả eKYC trực quan lên ảnh theo chuẩn test_pipeline_full."""
    h, w = image.shape[:2]
    vis = image.copy()

    clean_reasons = [remove_vietnamese_accents(r) for r in reasons] if (not final_pass and reasons) else []
    num_reasons = len(clean_reasons)
    extra_h = max(0, num_reasons * 22) if num_reasons > 0 else 0

    card_w = min(540, w - 20)
    card_h = min(h - 25, 235 + extra_h)
    draw_ui_card(vis, 15, 15, card_w, card_h, bg_color=(15, 15, 20), alpha=0.88)

    cv2.putText(vis, f"E-KYC SERVER PIPELINE REPORT (ID: {img_idx})", (25, 42),
                cv2.FONT_HERSHEY_SIMPLEX, 0.60, (0, 230, 255), 2, cv2.LINE_AA)
    cv2.line(vis, (25, 50), (15 + card_w - 20, 50), (80, 80, 80), 1)

    # 1. Face Detect
    if face_info:
        if num_faces == 1:
            f_txt = f"1. Face Detect   : 1 FACE (CONF: {face_info['confidence']:.2f}) -> PASS"
            f_col = (0, 255, 0)
        else:
            f_txt = f"1. Face Detect   : MULTI-FACE ({num_faces} FACES) -> WARNING"
            f_col = (0, 165, 255)
    else:
        f_txt = "1. Face Detect   : NO FACE DETECTED -> FAIL"
        f_col = (0, 0, 255)
    cv2.putText(vis, f_txt, (25, 72), cv2.FONT_HERSHEY_SIMPLEX, 0.44, f_col, 1, cv2.LINE_AA)

    # 2. Pose 3D
    if pose_info:
        yaw = pose_info.get("yaw", 0.0)
        pitch = pose_info.get("pitch", 0.0)
        roll = pose_info.get("roll", 0.0)
        p_stat = "PASS" if pose_valid else "FAIL"
        p_txt = f"2. Head Pose [{p_stat}] : Y:{yaw:+.1f} P:{pitch:+.1f} R:{roll:+.1f}"
        p_col = (0, 255, 0) if pose_valid else (0, 0, 255)
    else:
        p_txt = "2. Head Pose     : UNKNOWN"
        p_col = (0, 0, 255)
    cv2.putText(vis, p_txt, (25, 95), cv2.FONT_HERSHEY_SIMPLEX, 0.44, p_col, 1, cv2.LINE_AA)

    # 3. Anti-Spoof
    if anti_spoof_info:
        as_lbl = anti_spoof_info.get("label", "UNKNOWN")
        as_conf = anti_spoof_info.get("confidence", 0.0)
        as_col = (0, 255, 0) if anti_spoof_info.get("is_real", False) else (0, 0, 255)
        iou_str = f" | IoU:{spoof_iou:.2f}" if spoof_iou > 0 else ""
        as_txt = f"3. Anti-Spoof YOLO : {as_lbl} ({as_conf*100:.1f}%{iou_str})"
    else:
        as_txt = "3. Anti-Spoof YOLO : NO DATA"
        as_col = (0, 165, 255)
    cv2.putText(vis, as_txt, (25, 118), cv2.FONT_HERSHEY_SIMPLEX, 0.44, as_col, 1, cv2.LINE_AA)

    # 4. Blink Liveness
    b_txt = f"4. Blink Liveness: PASS ({blink_count} blinks)" if blink_passed else f"4. Blink Liveness: FAIL ({blink_count} blinks)"
    b_col = (0, 255, 0) if blink_passed else (0, 0, 255)
    cv2.putText(vis, b_txt, (25, 141), cv2.FONT_HERSHEY_SIMPLEX, 0.44, b_col, 1, cv2.LINE_AA)

    # 5. Head Movement Liveness
    h_act = str(head_action_name).upper()
    hm_txt = f"5. Head Movement : PASS [{h_act}]" if head_movement_passed else f"5. Head Movement : FAIL [{h_act}]"
    hm_col = (0, 255, 0) if head_movement_passed else (0, 0, 255)
    cv2.putText(vis, hm_txt, (25, 164), cv2.FONT_HERSHEY_SIMPLEX, 0.44, hm_col, 1, cv2.LINE_AA)

    cv2.line(vis, (25, 180), (15 + card_w - 20, 180), (80, 80, 80), 1)

    # 6. Final Decision
    verdict_text = "eKYC: APPROVED (HOP LE)" if final_pass else "eKYC: REJECTED (TU CHOI)"
    verdict_col = (0, 255, 0) if final_pass else (0, 0, 255)
    cv2.putText(vis, verdict_text, (25, 208),
                cv2.FONT_HERSHEY_SIMPLEX, 0.65, verdict_col, 2, cv2.LINE_AA)

    if not final_pass and clean_reasons:
        cv2.putText(vis, "Ly do tu choi:", (25, 230),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.44, (0, 200, 255), 1, cv2.LINE_AA)
        start_y = 250
        line_spacing = 20
        for idx_r, r_text in enumerate(clean_reasons[:5]):
            if len(r_text) > 65:
                r_text = r_text[:62] + "..."
            line_txt = f" * {r_text}"
            cv2.putText(vis, line_txt, (25, start_y + idx_r * line_spacing),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.40, (140, 210, 255), 1, cv2.LINE_AA)

    return vis
