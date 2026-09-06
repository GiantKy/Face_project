"""
E-KYC Server Module Package.
Gói module xử lý Computer Vision & Deep Learning chuẩn ngân hàng cho Server.
Sử dụng YOLO Face Detection và YOLO Anti-Spoofing theo chuẩn test_pipeline_full.
"""

from .pipeline_server import EKYCPipelineServer
from .anti_spoof_yolo import AntiSpoofYoloDetector
from .utils import (
    load_image,
    image_to_base64,
    calculate_iou,
    compute_eye_aspect_ratio,
    remove_vietnamese_accents,
    draw_pipeline_result_hud
)
from .config import (
    FACE_DETECTION_MODEL_PATH,
    ANTI_SPOOF_YOLO_MODEL_PATH,
    FACE_LANDMARKER_MODEL_PATH,
    CONF_THRESHOLD_FACE,
    CONF_THRESHOLD_ANTI_SPOOF
)

__all__ = [
    "EKYCPipelineServer",
    "AntiSpoofYoloDetector",
    "load_image",
    "image_to_base64",
    "calculate_iou",
    "compute_eye_aspect_ratio",
    "remove_vietnamese_accents",
    "draw_pipeline_result_hud",
    "FACE_DETECTION_MODEL_PATH",
    "ANTI_SPOOF_YOLO_MODEL_PATH",
    "FACE_LANDMARKER_MODEL_PATH",
    "CONF_THRESHOLD_FACE",
    "CONF_THRESHOLD_ANTI_SPOOF",
]

__version__ = "1.0.0"
