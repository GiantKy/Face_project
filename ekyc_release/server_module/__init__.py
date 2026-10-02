"""
E-KYC Server Module Package — Ensemble Edition.
Gói module xử lý Computer Vision & Deep Learning chuẩn ngân hàng cho Server.
Sử dụng Ensemble Anti-Spoofing (YOLO_4 + RF-DETR Small) theo chuẩn test_pipeline_ensemble_full.
"""

from .pipeline_server import EKYCPipelineServer
from .ensemble_anti_spoof import EnsembleAntiSpoofDetector
from .anti_spoof_yolo import AntiSpoofYoloDetector
try:
    from .app import app
except ImportError:
    app = None
from .utils import (
    load_image,
    image_to_base64,
    calculate_iou,
    compute_eye_aspect_ratio,
    remove_vietnamese_accents,
    draw_pipeline_result_hud,
    create_pipeline_result_dashboard,
    create_side_by_side_result,
    show_dual_window_result
)
from .config import (
    FACE_DETECTION_MODEL_PATH,
    ANTI_SPOOF_YOLO_MODEL_PATH,
    ANTI_SPOOF_YOLO4_MODEL_PATH,
    FACE_LANDMARKER_MODEL_PATH,
    RFDETR_MODEL_ID,
    CONF_THRESHOLD_FACE,
    ENSEMBLE_CONF_THRESHOLD,
    ENSEMBLE_IOU_THRESHOLD,
)

__all__ = [
    # Core Pipeline
    "EKYCPipelineServer",
    "app",
    # Anti-Spoof Detectors
    "EnsembleAntiSpoofDetector",
    "AntiSpoofYoloDetector",
    # Utilities
    "load_image",
    "image_to_base64",
    "calculate_iou",
    "compute_eye_aspect_ratio",
    "remove_vietnamese_accents",
    "draw_pipeline_result_hud",
    "create_pipeline_result_dashboard",
    "create_side_by_side_result",
    "show_dual_window_result",
    # Config
    "FACE_DETECTION_MODEL_PATH",
    "ANTI_SPOOF_YOLO_MODEL_PATH",
    "ANTI_SPOOF_YOLO4_MODEL_PATH",
    "FACE_LANDMARKER_MODEL_PATH",
    "RFDETR_MODEL_ID",
    "CONF_THRESHOLD_FACE",
    "ENSEMBLE_CONF_THRESHOLD",
    "ENSEMBLE_IOU_THRESHOLD",
]

__version__ = "2.0.0"
