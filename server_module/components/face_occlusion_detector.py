"""
Face Occlusion, Eyeglasses & Mask Defense Module (AI Deep Learning).
Sử dụng mô hình AI Deep Learning YOLO26 Nano từ Roboflow:
- Model ID: glass-and-mask-q5de1/2
- Chạy 100% OFFLINE từ cache cục bộ (GPU DirectML / CUDA / CPU)
- Phát hiện chính xác: glass (đeo kính), mask (đeo khẩu trang), no_glass, no_mask.
- Chỉ đưa ra thông báo cảnh báo, không vẽ bounding box đè lên mắt/mặt.
"""

from typing import List, Tuple, Optional, Dict, Any, Union
import numpy as np
import cv2
import os

try:
    from server_module.config import STRICT_GLASSES_POLICY
except ImportError:
    try:
        from config import STRICT_GLASSES_POLICY
    except ImportError:
        STRICT_GLASSES_POLICY = True


class FaceOcclusionDetector:
    """
    Bộ phát hiện Kính & Khẩu trang bằng mô hình AI Deep Learning YOLO26n.
    Độ chính xác cao, loại bỏ hoàn toàn các thuật toán heuristic đo màu/cạnh gây báo động giả.
    Chỉ xuất cảnh báo thông báo (không vẽ bounding box lên mắt/khẩu trang).
    """

    def __init__(
        self,
        model_id: str = "glass-and-mask-q5de1/2",
        api_key: str = "lGvF9eLaX4ZhhERgN5u2",
        conf_threshold: float = 0.38,
        strict_glasses: bool = STRICT_GLASSES_POLICY
    ):
        self.model_id = model_id
        self.api_key = api_key
        self.conf_threshold = conf_threshold
        self.strict_glasses = strict_glasses
        self.ai_model = None
        self._init_ai_model()

    def _init_ai_model(self):
        try:
            from inference import get_model
            base_dir = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
            cache_dir = os.path.join(base_dir, "models", "roboflow")
            os.environ["MODEL_CACHE_DIR"] = cache_dir
            self.ai_model = get_model(model_id=self.model_id, api_key=self.api_key)
        except Exception as e:
            print(f"[WARN] FaceOcclusionDetector: Không thể khởi tạo model AI '{self.model_id}': {e}")
            self.ai_model = None

    def check_occlusion(
        self,
        frame: np.ndarray,
        landmarks: Optional[List[Tuple[int, int]]] = None,
        num_faces: int = 1,
        yolo_faces: Optional[List[Dict[str, Any]]] = None,
        pose_dict: Optional[Dict[str, float]] = None
    ) -> Tuple[bool, str, str]:
        """
        Kiểm tra khuôn mặt có đang đeo kính mắt hoặc đeo khẩu trang hay không bằng mô hình AI.

        Args:
            frame: Ảnh BGR gốc.
            landmarks: Danh sách tọa độ pixel (x, y) của landmarks (tùy chọn).
            num_faces: Số lượng khuôn mặt đếm được từ detector.
            yolo_faces: Danh sách kết quả từ YOLO Face Detector (nếu có).
            pose_dict: Góc quay 3D Euler (yaw, pitch, roll).

        Returns:
            Tuple[is_occluded, reason_code, message]
            - is_occluded: True nếu phát hiện đeo kính hoặc khẩu trang, False nếu mặt thông thoáng.
            - reason_code: Mã kỹ thuật (GLASS_DETECTED, MASK_DETECTED, OK).
            - message: Thông báo cảnh báo hướng dẫn người dùng.
        """
        if frame is None or frame.size == 0:
            return True, "EMPTY_FRAME", "Không nhận được khung hình camera"

        if num_faces == 0:
            return True, "NO_FACE", "Không tìm thấy khuôn mặt"

        # ---------------------------------------------------------------------
        # KIỂM TRA BẰNG MÔ HÌNH AI DEEP LEARNING (YOLO26n)
        # ---------------------------------------------------------------------
        if self.ai_model is not None:
            try:
                preds = self.ai_model.infer(frame)
                pred_list = []
                if isinstance(preds, list) and len(preds) > 0:
                    pred_list = getattr(preds[0], "predictions", [])
                elif hasattr(preds, "predictions"):
                    pred_list = preds.predictions

                has_glass = False
                has_mask = False

                for p in pred_list:
                    cls_name = str(getattr(p, "class_name", "")).lower().strip()
                    conf = float(getattr(p, "confidence", 0.0))
                    if conf >= self.conf_threshold:
                        if cls_name == "glass":
                            has_glass = True
                        elif cls_name == "mask":
                            has_mask = True

                if has_glass and has_mask:
                    return True, "GLASS_AND_MASK_DETECTED", "CẢNH BÁO: Phát hiện đang đeo kính mắt và khẩu trang!"
                elif has_glass:
                    return True, "GLASS_DETECTED", "CẢNH BÁO: Phát hiện đang đeo kính mắt!"
                elif has_mask:
                    return True, "MASK_DETECTED", "CẢNH BÁO: Phát hiện đang đeo khẩu trang!"

                # Nếu AI xác nhận không kính và không khẩu trang
                return False, "OK", ""
            except Exception as e:
                pass

        # Kiểm tra tối thiểu: nếu mất landmarks hoàn toàn khi num_faces > 0
        if landmarks is None or len(landmarks) < 100:
            return True, "FACE_OCCLUDED", "Khuôn mặt bị che khuất"

        return False, "OK", ""
