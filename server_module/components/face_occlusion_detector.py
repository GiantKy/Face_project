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
        api_key: Optional[str] = None,
        conf_threshold: float = 0.55,
        strict_glasses: bool = STRICT_GLASSES_POLICY
    ):
        self.model_id = model_id
        self.api_key = api_key if api_key is not None else os.environ.get("ROBOFLOW_API_KEY", "")
        self.conf_threshold = conf_threshold
        self.strict_glasses = strict_glasses
        self.ai_model = None
        self._last_check_time = 0.0
        self._cached_result = (False, "OK", "")
        self._init_ai_model()

    def _init_ai_model(self):
        try:
            from inference import get_model
            _current_file = os.path.abspath(__file__)
            _comp_dir = os.path.dirname(_current_file)
            _server_dir = os.path.dirname(_comp_dir)
            _root_dir = os.path.dirname(_server_dir)

            cache_dir = os.path.join(_server_dir, "models", "roboflow")
            if not os.path.exists(cache_dir):
                alt_dir = os.path.join(_root_dir, "models", "roboflow")
                if os.path.exists(alt_dir):
                    cache_dir = alt_dir
            target_cache = os.path.join(cache_dir, "glass-and-mask-q5de1", "2")
            alt_cache = os.path.join(cache_dir, "Mask_Glass", "2")
            if not os.path.exists(target_cache) and os.path.exists(alt_cache):
                import shutil
                os.makedirs(os.path.dirname(target_cache), exist_ok=True)
                shutil.copytree(alt_cache, target_cache, dirs_exist_ok=True)
            elif os.path.exists(target_cache) and not os.path.exists(alt_cache):
                import shutil
                os.makedirs(os.path.dirname(alt_cache), exist_ok=True)
                shutil.copytree(target_cache, alt_cache, dirs_exist_ok=True)

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
        pose_dict: Optional[Dict[str, float]] = None,
        force_fresh: bool = False,
        oval_center: Optional[Tuple[int, int]] = None,
        oval_axes: Optional[Tuple[int, int]] = None,
        filter_oval: bool = True
    ) -> Tuple[bool, str, str]:
        """
        Kiểm tra khuôn mặt có đang đeo kính mắt hoặc đeo khẩu trang hay không bằng mô hình AI.
        Hỗ trợ lọc oval: chỉ kiểm tra khuôn mặt nằm trong khung oval, bỏ qua các mặt ngoài oval.
        """
        if frame is None or frame.size == 0:
            return True, "EMPTY_FRAME", "Không nhận được khung hình camera"

        if num_faces == 0:
            return True, "NO_FACE", "Không tìm thấy khuôn mặt"

        import time
        now = time.time()
        # Dùng cache nếu các frame tới quá dồn dập trong vòng 150ms
        if not force_fresh and (now - self._last_check_time < 0.15):
            return self._cached_result

        # ---------------------------------------------------------------------
        # KIỂM TRA BẰNG MÔ HÌNH AI DEEP LEARNING (YOLO26n)
        # ---------------------------------------------------------------------
        if self.ai_model is not None:
            try:
                # Tối ưu kích thước ảnh đầu vào để giảm thiểu độ trễ tối đa
                h, w = frame.shape[:2]
                max_dim = max(h, w)
                scale = 1.0
                if max_dim > 416:
                    scale = 416.0 / max_dim
                    infer_frame = cv2.resize(frame, (int(w * scale), int(h * scale)), interpolation=cv2.INTER_LINEAR)
                else:
                    infer_frame = frame

                preds = self.ai_model.infer(infer_frame)
                pred_list = []
                if isinstance(preds, list) and len(preds) > 0:
                    pred_list = getattr(preds[0], "predictions", [])
                elif hasattr(preds, "predictions"):
                    pred_list = preds.predictions

                # Lọc không gian nếu có oval: chỉ xét detection nằm trong khung oval
                valid_preds = []
                for p in pred_list:
                    if filter_oval and oval_center is not None and oval_axes is not None:
                        cx, cy = oval_center
                        ax, ay = oval_axes
                        scale_factor = scale if (max_dim > 416 and scale > 0) else 1.0
                        px = float(getattr(p, "x", 0.0)) / scale_factor
                        py = float(getattr(p, "y", 0.0)) / scale_factor
                        if ax > 0 and ay > 0 and (px > 0 or py > 0):
                            norm_x = (px - cx) / float(ax * 1.25)
                            norm_y = (py - cy) / float(ay * 1.25)
                            if (norm_x ** 2 + norm_y ** 2) > 1.0:
                                continue  # Bỏ qua detection ở ngoài vùng oval khuôn mặt
                    valid_preds.append(p)

                glass_confs = [float(getattr(p, "confidence", 0.0)) for p in valid_preds if str(getattr(p, "class_name", "")).lower().strip() == "glass"]
                no_glass_confs = [float(getattr(p, "confidence", 0.0)) for p in valid_preds if str(getattr(p, "class_name", "")).lower().strip() == "no_glass"]
                mask_confs = [float(getattr(p, "confidence", 0.0)) for p in valid_preds if str(getattr(p, "class_name", "")).lower().strip() == "mask"]
                no_mask_confs = [float(getattr(p, "confidence", 0.0)) for p in valid_preds if str(getattr(p, "class_name", "")).lower().strip() == "no_mask"]

                max_glass = max(glass_confs, default=0.0)
                max_no_glass = max(no_glass_confs, default=0.0)
                max_mask = max(mask_confs, default=0.0)
                max_no_mask = max(no_mask_confs, default=0.0)

                # Logic phân định Kính mắt:
                # Nếu model phát hiện no_glass mạnh (>= 0.50) hoặc no_glass >= glass -> Chắc chắn KHÔNG ĐEO KÍNH
                if max_no_glass >= 0.50 or max_no_glass >= max_glass:
                    has_glass = False
                else:
                    has_glass = (max_glass >= self.conf_threshold and max_glass > max_no_glass + 0.08)

                # Logic phân định Khẩu trang:
                if max_no_mask >= 0.50 or max_no_mask >= max_mask:
                    has_mask = False
                else:
                    has_mask = (max_mask >= self.conf_threshold and max_mask > max_no_mask + 0.08)

                self._last_check_time = now

                if has_glass and has_mask:
                    res = (True, "GLASS_AND_MASK_DETECTED", "CẢNH BÁO: Phát hiện đang đeo kính mắt và khẩu trang!")
                elif has_glass:
                    res = (True, "GLASS_DETECTED", "CẢNH BÁO: Phát hiện đang đeo kính mắt!")
                elif has_mask:
                    res = (True, "MASK_DETECTED", "CẢNH BÁO: Phát hiện đang đeo khẩu trang!")
                else:
                    # Nếu AI xác nhận không kính và không khẩu trang
                    res = (False, "OK", "")

                self._cached_result = res
                return res
            except Exception as e:
                pass

        # Kiểm tra tối thiểu: nếu mất landmarks hoàn toàn khi num_faces > 0
        if landmarks is None or len(landmarks) < 100:
            res = (True, "FACE_OCCLUDED", "Khuôn mặt bị che khuất")
            self._cached_result = res
            return res

        res = (False, "OK", "")
        self._cached_result = res
        return res
