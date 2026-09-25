"""
Pydantic Schemas for E-KYC FastAPI Server.
Định nghĩa cấu trúc dữ liệu Request/Response chuẩn hóa cho giao tiếp giữa Node.js, Web Client và AI Engine.
"""

from typing import List, Optional, Dict, Any
from pydantic import BaseModel, Field


# =============================================================================
# REQUEST SCHEMAS
# =============================================================================

class VerifyJsonRequest(BaseModel):
    """Payload gửi lên qua JSON Base64."""
    image_base64: str = Field(..., description="Chuỗi ảnh Base64 (có hoặc không có tiền tố data:image/...)")
    img_id: Optional[str] = Field(default="1", description="Mã định danh giao dịch hoặc ID ảnh")
    user_id: Optional[str] = Field(default=None, description="Mã định danh người dùng (tùy chọn)")
    blink_passed: bool = Field(default=True, description="Trạng thái chớp mắt đạt chuẩn (từ client/liveness check)")
    blink_count: int = Field(default=1, description="Số lần chớp mắt ghi nhận")
    head_passed: bool = Field(default=True, description="Trạng thái quay đầu đạt chuẩn")
    head_action: str = Field(default="TURN_LEFT", description="Hành động quay đầu: TURN_LEFT, TURN_RIGHT, NOD_UP, NOD_DOWN")
    return_annotated_image: bool = Field(default=True, description="Có trả về ảnh Base64 đã vẽ HUD/khung nhận diện hay không")
    return_crop_image: bool = Field(default=True, description="Có trả về ảnh khuôn mặt đã crop 224x224 Base64 hay không")
    apply_oval_mask: bool = Field(default=True, description="Làm mờ bối cảnh ngoại vi trừ khung Oval ở giữa")
    output_dir: Optional[str] = Field(default=None, description="Đường dẫn thư mục lưu ảnh/báo cáo trên ổ cứng server (nếu muốn lưu)")


class PoseValidateJsonRequest(BaseModel):
    """Payload kiểm tra tư thế qua JSON Base64."""
    image_base64: str = Field(..., description="Chuỗi ảnh Base64")


class AntiSpoofJsonRequest(BaseModel):
    """Payload kiểm tra chống giả mạo qua JSON Base64."""
    image_base64: str = Field(..., description="Chuỗi ảnh Base64")
    conf_threshold: Optional[float] = Field(default=0.30, description="Ngưỡng confidence tối thiểu")


# =============================================================================
# RESPONSE SCHEMAS
# =============================================================================

class FaceBBox(BaseModel):
    bbox: List[int] = Field(..., description="Bounding box [x1, y1, x2, y2]")
    confidence: float = Field(..., description="Độ tin cậy nhận diện khuôn mặt (0.0 - 1.0)")


class FaceDetectionDetail(BaseModel):
    num_faces: int = Field(..., description="Tổng số khuôn mặt tìm thấy")
    primary_face: Optional[FaceBBox] = Field(None, description="Thông tin khuôn mặt chính (lớn nhất và gần tâm)")


class Pose3DDetail(BaseModel):
    is_valid: bool = Field(..., description="Tư thế khuôn mặt có đạt chuẩn không")
    message: str = Field(..., description="Thông điệp đánh giá góc quay")
    yaw: float = Field(..., description="Góc quay ngang (-: trái, +: phải)")
    pitch: float = Field(..., description="Góc ngửa/cúi (-: ngửa, +: cúi)")
    roll: float = Field(..., description="Góc nghiêng đầu sang vai")


class EnsembleModelDetection(BaseModel):
    bbox: List[int]
    is_real: bool
    label: str
    confidence: float
    source: str
    yolo_res: str
    rfdetr_res: str
    both_detected: bool


class EnsembleAntiSpoofDetail(BaseModel):
    label: str = Field(..., description="Nhãn phân loại: REAL hoặc SPOOF")
    is_real: bool = Field(..., description="Khuôn mặt chính là người thật (True) hay giả mạo (False)")
    confidence: float = Field(..., description="Độ tin cậy của ensemble (0.0 - 1.0)")
    primary_iou: float = Field(..., description="Độ trùng khớp IoU giữa BBox khuôn mặt chính và BBox Anti-Spoof")
    has_any_spoof_in_frame: bool = Field(..., description="Có phát hiện dấu hiệu giả mạo nào trong toàn khung hình không")
    source: str = Field(..., description="Nguồn kết quả: ENSEMBLE, YOLO_ONLY, RFDETR_ONLY")
    yolo_detail: str = Field(..., description="Chi tiết nhãn và điểm của mô hình YOLO_4")
    rfdetr_detail: str = Field(..., description="Chi tiết nhãn và điểm của mô hình RF-DETR Small")
    agreement: bool = Field(..., description="Hai mô hình có cùng phân loại REAL/SPOOF không")
    both_detected: bool = Field(..., description="Cả 2 mô hình có cùng nhận diện được khuôn mặt này không")
    latency_ms: float = Field(..., description="Thời gian suy luận của cụm Ensemble (ms)")
    models_used: Dict[str, str] = Field(default_factory=dict, description="Tên các mô hình được dùng")
    all_ensemble_detections: List[EnsembleModelDetection] = Field(default_factory=list)


class ActiveLivenessDetail(BaseModel):
    blink_passed: bool
    blink_count: int
    head_movement_passed: bool
    head_action: str


class CriteriaDetail(BaseModel):
    face_detected: bool = Field(..., description="1. Tìm thấy khuôn mặt")
    single_face: bool = Field(..., description="2. Chỉ có 1 người duy nhất trong ảnh")
    pose_valid: bool = Field(..., description="3. Góc mặt thẳng, khoảng cách chuẩn")
    anti_spoof_real: bool = Field(..., description="4. Mặt chính là người thật")
    both_models_detected: bool = Field(..., description="5. Cả 2 mô hình đồng thuận nhận diện")
    blink_passed: bool = Field(..., description="6. Hoàn thành chớp mắt")
    head_movement_passed: bool = Field(..., description="7. Hoàn thành quay đầu")


class FinalDecision(BaseModel):
    approved: bool = Field(..., description="Kết luận cuối cùng: Đạt hay Không đạt")
    verdict: str = Field(..., description="APPROVED hoặc REJECTED")
    reasons: List[str] = Field(default_factory=list, description="Danh sách lý do bị từ chối nếu có")


class VerifyResponse(BaseModel):
    """Response chuẩn hóa gửi về cho Server Node.js và Web Client."""
    success: bool = Field(default=True, description="API xử lý thành công")
    image_id: str = Field(..., description="ID ảnh hoặc ID giao dịch")
    timestamp: str = Field(..., description="Thời điểm xử lý")
    approved: bool = Field(..., description="Kết quả xác thực: True (Hợp lệ) / False (Từ chối)")
    verdict: str = Field(..., description="APPROVED hoặc REJECTED")
    is_real: bool = Field(..., description="Người thật (True) hay Giả mạo (False)")
    confidence: float = Field(..., description="Độ tin cậy chống giả mạo (0.0 - 1.0)")
    reasons: List[str] = Field(default_factory=list, description="Lý do từ chối (nếu rejected)")
    criteria: CriteriaDetail = Field(..., description="Chi tiết 7 tiêu chuẩn đánh giá eKYC")
    face_detection: FaceDetectionDetail = Field(..., description="Dữ liệu phát hiện khuôn mặt")
    pose_3d: Pose3DDetail = Field(..., description="Dữ liệu tư thế 3D")
    ensemble_anti_spoof: EnsembleAntiSpoofDetail = Field(..., description="Dữ liệu Ensemble Anti-Spoofing chi tiết")
    active_liveness: ActiveLivenessDetail = Field(..., description="Dữ liệu kiểm tra cử động")
    crop_face_base64: Optional[str] = Field(None, description="Ảnh khuôn mặt crop 224x224 Base64 chuẩn hóa (lưu DB)")
    annotated_image_base64: Optional[str] = Field(None, description="Ảnh đã vẽ HUD và khung nhận diện (hiển thị UI)")
    dual_window_image_base64: Optional[str] = Field(None, description="Ảnh Dual Window (Side-by-Side) kết hợp ảnh khuôn mặt và dashboard AI 5 phần")
    oval_guide: Optional[Dict[str, Any]] = Field(None, description="Tọa độ và thông tin khung oval hướng dẫn")
    processing_time_ms: float = Field(..., description="Tổng thời gian xử lý toàn bộ quy trình (ms)")


class PoseValidateResponse(BaseModel):
    success: bool = True
    has_face: bool
    is_valid: bool
    face_in_oval: bool = False
    is_aligned_good: bool = False
    face_size_h: int
    is_too_far: bool
    is_too_close: bool = False
    is_off_center: bool = False
    off_center_hint: str = ""
    oval_guide: Optional[Dict[str, Any]] = None
    pose: Dict[str, Any]
    message: str
    guide: str
    processing_time_ms: float


class AntiSpoofResponse(BaseModel):
    success: bool = True
    has_face: bool
    num_faces: int
    is_real: bool
    label: str
    confidence: float
    primary_spoof_iou: float
    primary_face: Optional[Dict[str, Any]] = None
    crop_face_base64: Optional[str] = None
    ensemble_detail: Optional[Dict[str, Any]] = None
    all_ensemble_detections: List[Dict[str, Any]] = Field(default_factory=list)
    processing_time_ms: float


class HealthResponse(BaseModel):
    status: str = "healthy"
    engine: str = "E-KYC Server Module (Ensemble Edition: YOLO_4 + RF-DETR Small)"
    models_loaded: bool
    device: str
    version: str
