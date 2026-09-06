"""
YOLO Anti-Spoofing Detector for Server Module.
Được trích xuất và chuẩn hóa trực tiếp từ test_pipeline_full.py sử dụng mô hình Anti_Spoof_YOLO.pt.
"""

import os
import glob
from typing import List, Dict, Any, Optional
import numpy as np
from ultralytics import YOLO

from .config import ANTI_SPOOF_YOLO_MODEL_PATH, CONF_THRESHOLD_ANTI_SPOOF, SERVER_MODULE_DIR


class AntiSpoofYoloDetector:
    """
    Bộ nhận diện Anti-Spoofing bằng YOLO (Anti_Spoof_YOLO.pt).
    Hỗ trợ cả model dạng phân loại (Classification) lẫn phát hiện đối tượng (Object Detection).
    """

    def __init__(self, model_path: Optional[str] = None):
        if model_path is None or not os.path.exists(model_path):
            internal_models_dir = os.path.join(SERVER_MODULE_DIR, "models")
            root_models_dir = os.path.join(PROJECT_ROOT, "models")
            candidate_files = [
                ANTI_SPOOF_YOLO_MODEL_PATH,
                os.path.join(internal_models_dir, "Anti_Spoof_YOLO.pt"),
                os.path.join(root_models_dir, "Anti_Spoof_YOLO.pt")
            ]
            self.model_path = None
            for p in candidate_files:
                if p and os.path.exists(p):
                    self.model_path = p
                    break

            if self.model_path is None:
                pts = glob.glob(os.path.join(internal_models_dir, "*Anti_Spoof*.pt")) + \
                      glob.glob(os.path.join(root_models_dir, "*Anti_Spoof*.pt"))
                if pts:
                    self.model_path = pts[0]
                else:
                    raise FileNotFoundError(
                        f"Không tìm thấy model Anti_Spoof YOLO tại: {ANTI_SPOOF_YOLO_MODEL_PATH} hoặc {root_models_dir}"
                    )
        else:
            self.model_path = model_path

        # Tải mô hình YOLO
        self.model = YOLO(self.model_path)
        self.classes = self.model.names
        print(f"[OK] AntiSpoofYoloDetector loaded: {self.model_path} (classes: {self.classes})")

    def predict(self, frame: np.ndarray, conf_threshold: float = CONF_THRESHOLD_ANTI_SPOOF) -> List[Dict[str, Any]]:
        """
        Chạy inference chống giả mạo trên khung hình.
        
        Tham số:
            frame: Ảnh đầu vào chuẩn BGR OpenCV.
            conf_threshold: Ngưỡng tự tin tối thiểu (mặc định 0.25).
            
        Trả về:
            Danh sách các detection:
            [
                {
                    "bbox": [x1, y1, x2, y2],
                    "is_real": True/False,
                    "label": "REAL" / "SPOOF",
                    "confidence": float,
                    "raw_class": str
                }, ...
            ]
        """
        if frame is None or frame.size == 0:
            return []

        results = self.model(frame, verbose=False, conf=conf_threshold)
        detections = []

        h, w = frame.shape[:2]

        for r in results:
            # 1. Trường hợp mô hình Classification (có r.probs)
            if hasattr(r, "probs") and r.probs is not None:
                probs = r.probs.data.cpu().numpy()
                top1 = int(r.probs.top1)
                score = float(probs[top1])
                label = self.classes.get(top1, f"class_{top1}").lower()
                is_real = ("real" in label)
                detections.append({
                    "bbox": [0, 0, w, h],
                    "is_real": bool(is_real),
                    "label": "REAL" if is_real else "SPOOF",
                    "confidence": float(score),
                    "raw_class": label
                })

            # 2. Trường hợp mô hình Object Detection (có r.boxes)
            elif hasattr(r, "boxes") and len(r.boxes) > 0:
                for box in r.boxes:
                    x1, y1, x2, y2 = map(int, box.xyxy[0])
                    conf = float(box.conf[0])
                    cls_id = int(box.cls[0])
                    label = self.classes.get(cls_id, f"class_{cls_id}").lower()
                    is_real = ("real" in label)
                    detections.append({
                        "bbox": [x1, y1, x2, y2],
                        "is_real": bool(is_real),
                        "label": "REAL" if is_real else "SPOOF",
                        "confidence": float(conf),
                        "raw_class": label
                    })

        return detections
