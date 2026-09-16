"""
Ensemble Anti-Spoofing Detector for Server Module.
Kết hợp song song 2 model:
  - Model 1: YOLO_4 (Anti_Spoof_YOLO_4.pt) — Object Detection local
  - Model 2: RF-DETR Small (Transformer) — Roboflow Inference

Ensemble Fusion Strategy:
  1. IoU Bounding Box Matching: Ghép cặp detection giữa 2 model
  2. Weighted Soft-Voting: Tính xác suất REAL trung bình có trọng số
  3. Strict Spoof Veto: Nếu 1 model cảnh báo SPOOF với conf >= 0.68 → VETO thành SPOOF

Trích xuất và chuẩn hóa từ test_pipeline_ensemble_full.py cho chế độ Server headless.
"""

import os
import sys
import glob
import warnings
from typing import List, Dict, Any, Optional, Tuple
import numpy as np

from ultralytics import YOLO

try:
    from .config import (
        ANTI_SPOOF_YOLO4_MODEL_PATH,
        RFDETR_MODEL_ID,
        RFDETR_API_KEY,
        ROBOFLOW_CACHE_DIR,
        MODELS_DIR,
        ENSEMBLE_CONF_THRESHOLD,
        ENSEMBLE_IOU_THRESHOLD,
        ENSEMBLE_W_YOLO,
        ENSEMBLE_W_RFDETR,
        ENSEMBLE_SPOOF_VETO_THRESHOLD,
    )
    from .utils import calculate_iou
except (ImportError, ValueError):
    from config import (
        ANTI_SPOOF_YOLO4_MODEL_PATH,
        RFDETR_MODEL_ID,
        RFDETR_API_KEY,
        ROBOFLOW_CACHE_DIR,
        MODELS_DIR,
        ENSEMBLE_CONF_THRESHOLD,
        ENSEMBLE_IOU_THRESHOLD,
        ENSEMBLE_W_YOLO,
        ENSEMBLE_W_RFDETR,
        ENSEMBLE_SPOOF_VETO_THRESHOLD,
    )
    from utils import calculate_iou


class EnsembleAntiSpoofDetector:
    """
    Bộ nhận diện Anti-Spoofing Ensemble (YOLO_4 + RF-DETR Small).
    
    Kết hợp 2 mô hình để tăng độ chính xác và giảm false positive:
    - YOLO_4: Object detection trên ảnh tĩnh (local inference)
    - RF-DETR Small: Transformer trích xuất vân ảnh & đặc trưng sâu (Roboflow inference)
    
    Chiến lược Ensemble:
    - IoU Matching: Ghép cặp bounding box từ 2 model (IoU >= 0.40)
    - Soft Voting: Tính trung bình có trọng số xác suất REAL
    - Spoof Veto: Nếu bất kỳ model nào phát hiện SPOOF với conf >= 0.68 → tự động VETO
    - Nếu chỉ 1 model nhận diện (không có đồng thuận) → đánh dấu UNCERTAIN
    """

    def __init__(
        self,
        yolo_model_path: Optional[str] = None,
        rfdetr_model_id: Optional[str] = None,
        rfdetr_api_key: Optional[str] = None,
    ):
        """
        Khởi tạo Ensemble Detector.
        
        Args:
            yolo_model_path: Đường dẫn tới file Anti_Spoof_YOLO_4.pt. Nếu None, dùng config mặc định.
            rfdetr_model_id: Model ID của RF-DETR trên Roboflow. Nếu None, dùng config mặc định.
            rfdetr_api_key: API key Roboflow. Nếu None, dùng config mặc định hoặc env var.
        """
        # =====================================================================
        # 1. Load Model YOLO_4
        # =====================================================================
        self.yolo_path = yolo_model_path or ANTI_SPOOF_YOLO4_MODEL_PATH

        if not os.path.exists(self.yolo_path):
            # Fallback: Tìm bất kỳ file Anti_Spoof*.pt nào trong server_module/models/
            pts = glob.glob(os.path.join(MODELS_DIR, "*Anti_Spoof*.pt"))
            self.yolo_path = pts[0] if pts else ANTI_SPOOF_YOLO4_MODEL_PATH

        print(f"[EnsembleAntiSpoof] Loading Model 1 (YOLO_4): {self.yolo_path}")
        self.yolo_model = YOLO(self.yolo_path)
        self.yolo_classes = self.yolo_model.names
        print(f"[EnsembleAntiSpoof] YOLO_4 loaded — classes: {self.yolo_classes}")

        # =====================================================================
        # 2. Load Model RF-DETR Small
        # =====================================================================
        self.rfdetr_model_id = rfdetr_model_id or RFDETR_MODEL_ID
        self.rfdetr_api_key = rfdetr_api_key or RFDETR_API_KEY
        self.rfdetr_model = None
        self.rfdetr_available = False

        # Thiết lập cache cho Roboflow
        os.environ["MODEL_CACHE_DIR"] = ROBOFLOW_CACHE_DIR
        os.makedirs(ROBOFLOW_CACHE_DIR, exist_ok=True)

        try:
            from inference import get_model
            print(f"[EnsembleAntiSpoof] Loading Model 2 (RF-DETR Small): {self.rfdetr_model_id}")
            self.rfdetr_model = get_model(
                model_id=self.rfdetr_model_id,
                api_key=self.rfdetr_api_key
            )
            self.rfdetr_available = True
            print("[EnsembleAntiSpoof] RF-DETR Small loaded successfully!")
        except Exception as e:
            print(f"[EnsembleAntiSpoof] WARNING: Không thể load RF-DETR Small: {e}")
            print("[EnsembleAntiSpoof] Sẽ fallback về chế độ YOLO_4 đơn lẻ (single model).")
            self.rfdetr_available = False

        mode_str = "Ensemble (YOLO_4 + RF-DETR)" if self.rfdetr_available else "YOLO_4 Only (Fallback)"
        print(f"[EnsembleAntiSpoof] Ready — Mode: {mode_str}\n")

    def _predict_yolo(
        self,
        frame: np.ndarray,
        conf_threshold: float = ENSEMBLE_CONF_THRESHOLD
    ) -> List[Dict[str, Any]]:
        """
        Chạy inference YOLO_4 trên ảnh.
        
        Returns:
            Danh sách detection, mỗi item chứa: bbox, is_real, label, confidence, p_real, raw_class
        """
        results = self.yolo_model(frame, verbose=False, conf=conf_threshold)
        detections = []

        for r in results:
            if hasattr(r, "boxes") and len(r.boxes) > 0:
                for box in r.boxes:
                    x1, y1, x2, y2 = map(int, box.xyxy[0])
                    conf = float(box.conf[0])
                    cls_id = int(box.cls[0])
                    raw_label = self.yolo_classes.get(cls_id, f"class_{cls_id}").lower()
                    is_real = ("real" in raw_label)
                    p_real = conf if is_real else (1.0 - conf)
                    detections.append({
                        "bbox": [x1, y1, x2, y2],
                        "is_real": is_real,
                        "label": "REAL" if is_real else "SPOOF",
                        "confidence": conf,
                        "p_real": p_real,
                        "raw_class": raw_label,
                    })

        return detections

    def _predict_rfdetr(
        self,
        frame: np.ndarray,
        conf_threshold: float = ENSEMBLE_CONF_THRESHOLD
    ) -> List[Dict[str, Any]]:
        """
        Chạy inference RF-DETR Small trên ảnh.
        
        Returns:
            Danh sách detection, mỗi item chứa: bbox, is_real, label, confidence, p_real, raw_class
        """
        if not self.rfdetr_available or self.rfdetr_model is None:
            return []

        h, w = frame.shape[:2]
        preds = self.rfdetr_model.infer(frame)

        pred_list = []
        if isinstance(preds, list) and len(preds) > 0:
            pred_list = getattr(preds[0], "predictions", [])
        elif hasattr(preds, "predictions"):
            pred_list = preds.predictions

        detections = []
        for p in pred_list:
            cls_name = str(getattr(p, "class_name", "")).lower().strip()
            conf = float(getattr(p, "confidence", 0.0))

            if "background" in cls_name or conf < conf_threshold:
                continue

            cx = float(getattr(p, "x", 0.0))
            cy = float(getattr(p, "y", 0.0))
            pw = float(getattr(p, "width", 0.0))
            ph = float(getattr(p, "height", 0.0))

            # Normalize nếu tọa độ ở dạng [0, 1]
            if 0.0 <= cx <= 1.0 and 0.0 <= pw <= 1.0:
                cx *= w
                cy *= h
                pw *= w
                ph *= h

            x1 = max(0, int(cx - pw / 2.0))
            y1 = max(0, int(cy - ph / 2.0))
            x2 = min(w, int(cx + pw / 2.0))
            y2 = min(h, int(cy + ph / 2.0))

            is_real = ("real" in cls_name)
            p_real = conf if is_real else (1.0 - conf)
            detections.append({
                "bbox": [x1, y1, x2, y2],
                "is_real": is_real,
                "label": "REAL" if is_real else "SPOOF",
                "confidence": conf,
                "p_real": p_real,
                "raw_class": cls_name,
            })

        return detections

    def predict_ensemble(
        self,
        frame: np.ndarray,
        conf_threshold: float = ENSEMBLE_CONF_THRESHOLD,
        iou_thresh: float = ENSEMBLE_IOU_THRESHOLD,
        w_yolo: float = ENSEMBLE_W_YOLO,
        w_rfdetr: float = ENSEMBLE_W_RFDETR,
        strict_spoof_veto: bool = True,
        spoof_veto_threshold: float = ENSEMBLE_SPOOF_VETO_THRESHOLD,
    ) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]], List[Dict[str, Any]]]:
        """
        Chạy Ensemble 2 model và ghép kết quả:
          1. Chạy YOLO_4 và RF-DETR đồng thời
          2. IoU matching giữa các detection
          3. Soft voting trọng số cho xác suất REAL
          4. Strict Spoof Veto nếu 1 model cảnh báo SPOOF mạnh
        
        Args:
            frame: Ảnh đầu vào BGR (numpy array).
            conf_threshold: Ngưỡng confidence tối thiểu cho mỗi model.
            iou_thresh: Ngưỡng IoU để ghép cặp detection.
            w_yolo: Trọng số YOLO trong soft voting.
            w_rfdetr: Trọng số RF-DETR trong soft voting.
            strict_spoof_veto: Bật/tắt Spoof Veto.
            spoof_veto_threshold: Ngưỡng confidence để kích hoạt Spoof Veto.
        
        Returns:
            Tuple gồm 3 danh sách:
            - final_dets: Kết quả ensemble đã ghép cặp
            - yolo_dets: Kết quả gốc từ YOLO_4
            - rfdetr_dets: Kết quả gốc từ RF-DETR
        """
        if frame is None or frame.size == 0:
            return [], [], []

        yolo_dets = self._predict_yolo(frame, conf_threshold=conf_threshold)
        rfdetr_dets = self._predict_rfdetr(frame, conf_threshold=conf_threshold)

        # Nếu RF-DETR không khả dụng, fallback sang YOLO_4 đơn lẻ
        if not self.rfdetr_available or not rfdetr_dets:
            fallback_dets = []
            for y_det in yolo_dets:
                fallback_dets.append({
                    "bbox": y_det["bbox"],
                    "is_real": y_det["is_real"],
                    "label": y_det["label"],
                    "confidence": round(float(y_det["confidence"]), 4),
                    "source": "YOLO_Only (RF-DETR Unavailable)",
                    "yolo_res": f"{y_det['label']} ({y_det['confidence']*100:.1f}%)",
                    "rfdetr_res": "N/A",
                    "agreement": False,
                    "both_detected": False,
                })
            return fallback_dets, yolo_dets, rfdetr_dets

        # =====================================================================
        # ENSEMBLE FUSION: IoU Matching + Soft Voting + Spoof Veto
        # =====================================================================
        final_dets = []
        matched_rf = set()

        for y_det in yolo_dets:
            y_box = y_det["bbox"]
            best_iou = 0.0
            best_idx = -1

            for idx, r_det in enumerate(rfdetr_dets):
                if idx in matched_rf:
                    continue
                iou = calculate_iou(y_box, r_det["bbox"])
                if iou > best_iou:
                    best_iou = iou
                    best_idx = idx

            if best_iou >= iou_thresh and best_idx != -1:
                # ĐÃ GHÉP CẶP: Cả 2 model đều phát hiện cùng vùng
                matched_rf.add(best_idx)
                r_det = rfdetr_dets[best_idx]

                # Hợp nhất Bounding Box (trung bình 2 model)
                f_box = [int((y_box[i] + r_det["bbox"][i]) / 2) for i in range(4)]

                # Soft Voting xác suất REAL
                p_real_ensemble = (
                    y_det["p_real"] * w_yolo + r_det["p_real"] * w_rfdetr
                ) / (w_yolo + w_rfdetr)

                # Strict Spoof Veto
                if strict_spoof_veto and (
                    (not y_det["is_real"] and y_det["confidence"] >= spoof_veto_threshold) or
                    (not r_det["is_real"] and r_det["confidence"] >= spoof_veto_threshold)
                ):
                    is_real = False
                    conf = max(
                        y_det["confidence"] if not y_det["is_real"] else 0.0,
                        r_det["confidence"] if not r_det["is_real"] else 0.0,
                    )
                else:
                    is_real = (p_real_ensemble >= 0.50)
                    conf = p_real_ensemble if is_real else (1.0 - p_real_ensemble)

                final_dets.append({
                    "bbox": f_box,
                    "is_real": is_real,
                    "label": "REAL" if is_real else "SPOOF",
                    "confidence": round(float(conf), 4),
                    "source": "Ensemble (Both Models Matched)",
                    "yolo_res": f"{y_det['label']} ({y_det['confidence']*100:.1f}%)",
                    "rfdetr_res": f"{r_det['label']} ({r_det['confidence']*100:.1f}%)",
                    "agreement": y_det["is_real"] == r_det["is_real"],
                    "both_detected": True,
                })
            else:
                # Chỉ YOLO bắt được — thiếu đồng thuận RF-DETR
                final_dets.append({
                    "bbox": y_box,
                    "is_real": False,
                    "label": "UNCERTAIN",
                    "confidence": round(float(y_det["confidence"]), 4),
                    "source": "YOLO_Only (No Consensus)",
                    "yolo_res": f"{y_det['label']} ({y_det['confidence']*100:.1f}%)",
                    "rfdetr_res": "N/A",
                    "agreement": False,
                    "both_detected": False,
                })

        # Các detection chỉ RF-DETR bắt được — thiếu đồng thuận YOLO
        for idx, r_det in enumerate(rfdetr_dets):
            if idx not in matched_rf:
                final_dets.append({
                    "bbox": r_det["bbox"],
                    "is_real": False,
                    "label": "UNCERTAIN",
                    "confidence": round(float(r_det["confidence"]), 4),
                    "source": "RFDETR_Only (No Consensus)",
                    "yolo_res": "N/A",
                    "rfdetr_res": f"{r_det['label']} ({r_det['confidence']*100:.1f}%)",
                    "agreement": False,
                    "both_detected": False,
                })

        return final_dets, yolo_dets, rfdetr_dets

    def predict(
        self,
        frame: np.ndarray,
        conf_threshold: float = ENSEMBLE_CONF_THRESHOLD,
    ) -> List[Dict[str, Any]]:
        """
        Shortcut method tương thích API cũ (AntiSpoofYoloDetector.predict).
        Trả về danh sách ensemble detections (chỉ final_dets).
        """
        final_dets, _, _ = self.predict_ensemble(frame, conf_threshold=conf_threshold)
        return final_dets
