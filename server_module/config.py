"""
Config module for E-KYC Server Module.
Chứa các thiết lập đường dẫn weights, ngưỡng nhận diện và tham số thuật toán.
"""

import os
from pathlib import Path

# Đường dẫn thư mục gốc của project (Face-Project/)
SERVER_MODULE_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(SERVER_MODULE_DIR)

# Thư mục models nội bộ bên trong server_module và thư mục models gốc
MODELS_INTERNAL_DIR = os.path.join(SERVER_MODULE_DIR, "models")
MODELS_ROOT_DIR = os.path.join(PROJECT_ROOT, "models")

def _resolve_model_path(model_filename: str) -> str:
    """Tìm đường dẫn file model: ưu tiên server_module/models, dự phòng models/."""
    p1 = os.path.join(MODELS_INTERNAL_DIR, model_filename)
    if os.path.exists(p1):
        return p1
    p2 = os.path.join(MODELS_ROOT_DIR, model_filename)
    if os.path.exists(p2):
        return p2
    return p1

# Đường dẫn các mô hình AI theo chuẩn test_pipeline_full
FACE_DETECTION_MODEL_PATH = _resolve_model_path("Face_Detection.pt")
ANTI_SPOOF_YOLO_MODEL_PATH = _resolve_model_path("Anti_Spoof_YOLO.pt")
FACE_LANDMARKER_MODEL_PATH = _resolve_model_path("face_landmarker.task")

# Thư mục lưu kết quả mặc định
DEFAULT_DATA_RAW_DIR = os.path.join(PROJECT_ROOT, "data_raw")
DEFAULT_OUTPUT_DIR = os.path.join(PROJECT_ROOT, "output")

# Ngưỡng Face Detection
CONF_THRESHOLD_FACE = 0.5

# Ngưỡng Anti-Spoofing YOLO
CONF_THRESHOLD_ANTI_SPOOF = 0.25

# Ngưỡng tư thế 3D Pose (Euler Angles: Yaw, Pitch, Roll)
POSE_MAX_YAW = 25.0       # Độ xoay ngang tối đa cho phép
POSE_MAX_PITCH = 20.0     # Độ ngước lên/cúi xuống tối đa cho phép
POSE_MAX_ROLL = 15.0      # Độ nghiêng đầu tối đa cho phép
MIN_FACE_HEIGHT = 170     # Chiều cao khuôn mặt tối thiểu trong khung hình (tránh ngồi quá xa)

# Ngưỡng Liveness Blink (Eye Aspect Ratio - EAR)
EAR_EYE_CLOSED_THRESHOLD = 0.18   # Dưới ngưỡng này coi như mắt nhắm
EAR_EYE_OPEN_THRESHOLD = 0.22     # Trên ngưỡng này coi như mắt mở
MIN_BLINKS_REQUIRED = 1

# Ngưỡng thử thách chuyển động đầu (Head Movement Challenge)
HEAD_YAW_THRESHOLD = 16.0
HEAD_PITCH_THRESHOLD = 12.0
CHALLENGE_TIMEOUT_SECONDS = 7.0
