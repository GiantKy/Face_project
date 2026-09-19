"""
Config module for E-KYC Server Module.
Chứa các thiết lập đường dẫn weights, ngưỡng nhận diện và tham số thuật toán.
"""

import os
from pathlib import Path

# Đường dẫn thư mục gốc
SERVER_MODULE_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(SERVER_MODULE_DIR)

# =====================================================================
# THƯ MỤC MODELS: Chỉ sử dụng server_module/models/ (tự chứa)
# =====================================================================
MODELS_DIR = os.path.join(SERVER_MODULE_DIR, "models")

# Đường dẫn các mô hình AI — tất cả nằm trong server_module/models/
FACE_DETECTION_MODEL_PATH = os.path.join(MODELS_DIR, "Face_Detection.pt")
ANTI_SPOOF_YOLO_MODEL_PATH = os.path.join(MODELS_DIR, "Anti_Spoof_YOLO.pt")
FACE_LANDMARKER_MODEL_PATH = os.path.join(MODELS_DIR, "face_landmarker.task")

# =====================================================================
# ENSEMBLE ANTI-SPOOF: YOLO_4 + RF-DETR Small
# =====================================================================
# Model 1: YOLO_4 (Anti_Spoof_YOLO_4.pt) — Object Detection local
ANTI_SPOOF_YOLO4_MODEL_PATH = os.path.join(MODELS_DIR, "Anti_Spoof_YOLO_4.pt")

# Model 2: RF-DETR Small — Roboflow Inference (Transformer)
RFDETR_MODEL_ID = "k-thi-gia-s-workspace/face-spoof-detection-liika-owgrl-1-rfdetr-small-t1"
RFDETR_API_KEY = os.environ.get("ROBOFLOW_API_KEY", "ydUs8YBnVWjyjFFVvcpx")

# Roboflow model cache — nằm trong server_module/models/roboflow/
ROBOFLOW_CACHE_DIR = os.path.join(MODELS_DIR, "roboflow")

# Ensemble fusion parameters
ENSEMBLE_CONF_THRESHOLD = 0.30       # Ngưỡng confidence tối thiểu cho mỗi model
ENSEMBLE_IOU_THRESHOLD = 0.40        # Ngưỡng IoU để ghép cặp detection giữa 2 model
ENSEMBLE_W_YOLO = 0.5                # Trọng số YOLO trong Soft Voting
ENSEMBLE_W_RFDETR = 0.5              # Trọng số RF-DETR trong Soft Voting
ENSEMBLE_SPOOF_VETO_THRESHOLD = 0.68 # Ngưỡng Spoof Veto: nếu 1 model phát hiện SPOOF >= ngưỡng → VETO

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
EAR_EYE_CLOSED_THRESHOLD = 0.20   # Dưới ngưỡng này coi như mắt nhắm (bắt trọn chớp mắt tự nhiên)
EAR_EYE_OPEN_THRESHOLD = 0.22     # Trên ngưỡng này coi như mắt mở
MIN_BLINKS_REQUIRED = 1

# Ngưỡng thử thách chuyển động đầu (Head Movement Challenge)
HEAD_YAW_THRESHOLD = 16.0
HEAD_PITCH_THRESHOLD = 12.0
HEAD_DELTA_YAW_THRESHOLD = 8.0     # Ngưỡng nhích nhẹ đầu tối thiểu (8 độ chuyển động thực tế từ mốc ban đầu)
HEAD_DELTA_PITCH_THRESHOLD = 7.0   # Ngưỡng nhích nhẹ gật đầu tối thiểu
CHALLENGE_TIMEOUT_SECONDS = 7.0

# =====================================================================
# NODE.JS BACKEND INTEGRATION & WEBHOOK
# =====================================================================
NODEJS_WEBHOOK_URL = os.environ.get("NODEJS_WEBHOOK_URL", "http://127.0.0.1:3000/api/ekyc/result")
NODEJS_WEBHOOK_TIMEOUT = float(os.environ.get("NODEJS_WEBHOOK_TIMEOUT", "5.0"))
