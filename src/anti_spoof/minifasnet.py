import sys
import os
import glob
import cv2
import numpy as np
import torch
import torch.nn as nn
from typing import Optional, Union, Tuple, Dict, Any, List

if hasattr(sys.stdout, "reconfigure"):
    try:
        sys.stdout.reconfigure(encoding="utf-8")
        sys.stderr.reconfigure(encoding="utf-8")
    except Exception:
        pass

# Tự động tìm thư mục gốc Face-Project
# __file__ = src/anti_spoof/minifasnet.py → 3 lần dirname để về Face-Project/
BASE_DIR = os.path.dirname(
    os.path.dirname(
        os.path.dirname(os.path.abspath(__file__))
    )
)


# =============================================================================
# 1. ARCHITECTURE COMPONENTS (MiniFASNetV2)
# =============================================================================
class Conv_block(nn.Module):
    """
    Standard Convolution + BatchNorm + PReLU block with grouping support
    """
    def __init__(self, in_c: int, out_c: int, kernel=(1, 1), stride=(1, 1), padding=(0, 0), groups=1):
        super().__init__()
        self.conv = nn.Conv2d(
            in_c, out_c,
            kernel_size=kernel,
            groups=groups,
            stride=stride,
            padding=padding,
            bias=False
        )
        self.bn = nn.BatchNorm2d(out_c)
        self.prelu = nn.PReLU()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.prelu(self.bn(self.conv(x)))


class Linear_block(nn.Module):
    """
    Convolution + BatchNorm (no activation)
    """
    def __init__(self, in_c: int, out_c: int, kernel=(1, 1), stride=(1, 1), padding=(0, 0), groups=1):
        super().__init__()
        self.conv = nn.Conv2d(
            in_c, out_c,
            kernel_size=kernel,
            groups=groups,
            stride=stride,
            padding=padding,
            bias=False
        )
        self.bn = nn.BatchNorm2d(out_c)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.bn(self.conv(x))


class Depth_Wise(nn.Module):
    """
    DepthWise Separable Convolution block
    """
    def __init__(self, in_c: int, out_c: int, residual: bool = False, kernel=(3, 3), stride=(2, 2), padding=(1, 1), groups=1):
        super().__init__()
        self.residual = residual
        self.conv = Conv_block(in_c, out_c=groups, kernel=(1, 1), padding=(0, 0), stride=(1, 1))
        self.conv_dw = Conv_block(groups, groups, groups=groups, kernel=kernel, padding=padding, stride=stride)
        self.project = Linear_block(groups, out_c, kernel=(1, 1), padding=(0, 0), stride=(1, 1))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        short_cut = x
        x = self.conv(x)
        x = self.conv_dw(x)
        x = self.project(x)
        if self.residual:
            return short_cut + x
        return x


class Residual(nn.Module):
    """
    Residual block containing multiple Conv_blocks
    """
    def __init__(self, c: int, num_block: int, groups: int, kernel=(3, 3), stride=(1, 1), padding=(1, 1)):
        super().__init__()
        modules = [
            Conv_block(c, c, kernel=kernel, stride=stride, padding=padding, groups=groups)
            for _ in range(num_block)
        ]
        self.model = nn.Sequential(*modules)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.model(x)


class MiniFASNetV2(nn.Module):
    """
    MiniFASNetV2 (Silent-Face-Anti-Spoofing lightweight backbone)
    Classes: 0 = FAKE / SPOOF, 1 = REAL
    """
    def __init__(self, embedding_size: int = 128, num_classes: int = 2, img_channel: int = 3):
        super().__init__()
        self.conv1 = nn.Sequential(
            nn.Conv2d(img_channel, 32, kernel_size=3, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(32),
            nn.PReLU()
        )
        self.conv2 = Conv_block(32, 64, kernel=3, stride=1, padding=1, groups=32)
        self.res1 = Residual(64, 1, groups=64)
        self.conv3 = Conv_block(64, 128, kernel=3, stride=2, padding=1, groups=64)
        self.res2 = Residual(128, 2, groups=128)
        self.conv4 = Conv_block(128, 128, kernel=3, stride=2, padding=1, groups=128)
        self.res3 = Residual(128, 2, groups=128)
        self.conv5 = Conv_block(128, 256, kernel=3, stride=2, padding=1, groups=128)
        self.res4 = Residual(256, 1, groups=256)
        self.fc = nn.Linear(256, embedding_size)
        self.classifier = nn.Linear(embedding_size, num_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.res1(x)
        x = self.conv3(x)
        x = self.res2(x)
        x = self.conv4(x)
        x = self.res3(x)
        x = self.conv5(x)
        x = self.res4(x)
        x = torch.mean(x, dim=[2, 3])
        feat = self.fc(x)
        return self.classifier(feat)


# =============================================================================
# 2. MODEL QUALITY CHECK
# =============================================================================
MIN_BATCHES_TRAINED = 1000  # Ngưỡng tối thiểu để model được coi là đã train đủ


def check_model_quality(state_dict: dict) -> Dict[str, Any]:
    """
    Kiểm tra chất lượng model dựa trên phân tích weights.
    Trả về dict chứa thông tin chẩn đoán.
    """
    info: Dict[str, Any] = {
        "is_usable": True,
        "warnings": [],
        "num_batches_trained": 0,
        "classifier_bias": None,
    }

    # Kiểm tra num_batches_tracked (số batch đã train)
    bt_keys = [k for k in state_dict.keys() if "num_batches_tracked" in k]
    if bt_keys:
        batches = state_dict[bt_keys[0]].item()
        info["num_batches_trained"] = batches
        if batches < MIN_BATCHES_TRAINED:
            info["is_usable"] = False
            info["warnings"].append(
                f"Model chỉ train được {batches} batches (cần tối thiểu ~{MIN_BATCHES_TRAINED}). "
                f"Model chưa converge và sẽ không phân biệt được Real/Fake chính xác!"
            )

    # Kiểm tra classifier bias
    if "classifier.bias" in state_dict:
        bias = state_dict["classifier.bias"].tolist()
        info["classifier_bias"] = bias
        # Nếu bias lệch quá nhiều về 1 class → model chưa học đúng
        bias_diff = abs(bias[1] - bias[0])
        cls_w = state_dict.get("classifier.weight", None)
        if cls_w is not None:
            w_std = cls_w.std().item()
            if w_std < 0.08 and bias_diff > 0.05:
                info["warnings"].append(
                    f"Classifier weights rất nhỏ (std={w_std:.4f}) với bias lệch "
                    f"({bias}) → model chưa học được feature phân biệt."
                )

    return info


def find_default_minifasnet_model() -> str:
    """
    Tự động tìm kiếm file model MiniFASNet trong thư mục models/,
    ưu tiên các file hiện hành: Anti_Spoof_minifasnet.pth hoặc các phiên bản tương đương.
    """
    candidate_files = [
        "Anti_Spoof_minifasnet.pth",
        "Anti_Spoof_minifasnetv2.pth",
        "Anti_Spoof_minifasnetv2_(4).pth",
        "Anti_Spoof_minifasnetv2_(3).pth",
        "Anti_Spoof_minifasnetv2_(2).pth",
        "Anti_Spoof_minifasnetv2_(1).pth",
        "best_minifasnetv2.pth",
        "2.7_80x80_MiniFASNetV2.pth"
    ]

    for fname in candidate_files:
        path = os.path.join(BASE_DIR, "models", fname)
        if os.path.exists(path):
            return path

    all_pth = glob.glob(os.path.join(BASE_DIR, "models", "*minifas*.pth"))
    if all_pth:
        return max(all_pth, key=os.path.getmtime)

    return os.path.join(BASE_DIR, "models", "Anti_Spoof_minifasnet.pth")


def load_minifasnet_model(model_path: Optional[str] = None, device: Optional[torch.device] = None) -> Tuple[MiniFASNetV2, Dict[str, Any]]:
    """
    Tải weights và khởi tạo model MiniFASNetV2.
    In thông tin chi tiết tên model, kích thước và trạng thái khi khởi chạy.
    Returns: (model, quality_info)
    """
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    if model_path is None:
        model_path = find_default_minifasnet_model()

    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Không tìm thấy file model MiniFASNet tại: {model_path}")

    model_name = os.path.basename(model_path)
    file_size_kb = os.path.getsize(model_path) / 1024.0
    print(f"\n[INFO] 📦 Đang nạp Model MiniFASNet: {model_name} ({file_size_kb:.1f} KB)")
    print(f"[INFO] 📍 Đường dẫn model: {model_path}")

    model = MiniFASNetV2()
    state_dict = torch.load(model_path, map_location=device)

    # Hỗ trợ checkpoint được bọc dict {'model_state_dict': ...}, {'state_dict': ...} hoặc trực tiếp OrderedDict
    if isinstance(state_dict, dict):
        if "model_state_dict" in state_dict:
            state_dict = state_dict["model_state_dict"]
        elif "state_dict" in state_dict:
            state_dict = state_dict["state_dict"]
        elif "model" in state_dict:
            state_dict = state_dict["model"]
        elif "net" in state_dict:
            state_dict = state_dict["net"]

    # Xử lý prefix 'module.' nếu train bằng DataParallel
    cleaned_state_dict = {}
    for k, v in state_dict.items():
        name = k[7:] if k.startswith("module.") else k
        cleaned_state_dict[name] = v

    # Kiểm tra chất lượng model trước khi load
    quality_info = check_model_quality(cleaned_state_dict)

    model.load_state_dict(cleaned_state_dict, strict=True)
    model.to(device)
    model.eval()

    batches = quality_info.get("num_batches_trained", 0)
    status_str = "✓ SẴN SÀNG" if quality_info.get("is_usable", False) else "✗ CHƯA SẴN SÀNG"
    print(f"[INFO] ✅ Tải weights {model_name} thành công | Batches: {batches} | Trạng thái: {status_str}\n")

    return model, quality_info


# =============================================================================
# 3. HIGH-LEVEL DETECTOR CLASS
# =============================================================================
class AntiSpoofMiniFASNet:
    """
    Bộ nhận diện Anti-Spoofing sử dụng mạng MiniFASNetV2
    Nhận diện khuôn mặt là ảnh Thật (REAL) hay Giả mạo (FAKE/SPOOF: in ảnh, màn hình điện thoại, video...)
    """
    def __init__(
        self,
        model_path: Optional[str] = None,
        input_size: Tuple[int, int] = (80, 80),
        scale_factor: float = 2.7,
        real_threshold: float = 0.5,
        device: Optional[Union[str, torch.device]] = None
    ):
        if device is None:
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        elif isinstance(device, str):
            self.device = torch.device(device)
        else:
            self.device = device

        self.input_size = input_size
        self.scale_factor = scale_factor
        self.real_threshold = real_threshold
        self.model_path = model_path if model_path else find_default_minifasnet_model()
        self.model_name = os.path.basename(self.model_path)
        self.model, self.quality_info = load_minifasnet_model(model_path=self.model_path, device=self.device)

        # In cảnh báo nếu model chưa train đủ
        if not self.quality_info["is_usable"]:
            print("\n" + "!" * 80)
            print(" CẢNH BÁO: MODEL MINIFASNET CHƯA TRAIN ĐỦ ")
            print("!" * 80)
            for w in self.quality_info["warnings"]:
                print(f"  ⚠ {w}")
            print(f"  → Batches trained: {self.quality_info['num_batches_trained']}")
            print(f"  → Kết quả dự đoán SẼ KHÔNG CHÍNH XÁC!")
            print(f"  → Vui lòng train lại model với đủ dữ liệu trước khi sử dụng.")
            print("!" * 80 + "\n")

    @property
    def is_model_ready(self) -> bool:
        """Trả về True nếu model đã train đủ để sử dụng"""
        return self.quality_info["is_usable"]

    def preprocess_crop(self, face_img: np.ndarray) -> torch.Tensor:
        """
        Chuẩn hóa ảnh khuôn mặt cắt về kích thước input_size (80, 80) và chuyển tensor.
        Preprocessing PHẢI KHỚP với code training:
          1. Resize về input_size (80x80)
          2. BGR → RGB
          3. ToTensor: [0, 255] → [0, 1] + HWC → CHW
          4. ImageNet Normalize: mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]
        """
        if face_img is None or face_img.size == 0:
            raise ValueError("Ảnh crop khuôn mặt rỗng!")

        resized = cv2.resize(face_img, self.input_size)
        # BGR → RGB
        resized = cv2.cvtColor(resized, cv2.COLOR_BGR2RGB)
        # ToTensor: [0, 255] → [0, 1] + HWC → CHW
        tensor = resized.astype(np.float32) / 255.0
        tensor = tensor.transpose((2, 0, 1))  # HWC → CHW
        # ImageNet Normalize (khớp với transforms.Normalize trong code training)
        mean = np.array([0.485, 0.456, 0.406], dtype=np.float32).reshape(3, 1, 1)
        std = np.array([0.229, 0.224, 0.225], dtype=np.float32).reshape(3, 1, 1)
        tensor = (tensor - mean) / std
        tensor = torch.from_numpy(tensor).unsqueeze(0).to(self.device)
        return tensor

    def crop_face(self, frame: np.ndarray, bbox: List[int], scale: Optional[float] = None) -> np.ndarray:
        """
        Cắt khuôn mặt từ frame theo bounding box [x1, y1, x2, y2] có hỗ trợ mở rộng tỷ lệ (scale)
        """
        x1, y1, x2, y2 = bbox
        h, w = frame.shape[:2]
        current_scale = scale if scale is not None else self.scale_factor

        if current_scale == 1.0:
            cx1 = max(0, min(w - 1, x1))
            cy1 = max(0, min(h - 1, y1))
            cx2 = max(0, min(w, x2))
            cy2 = max(0, min(h, y2))
            crop = frame[cy1:cy2, cx1:cx2]
        else:
            # Scale bbox quanh tâm mặt
            box_w = x2 - x1
            box_h = y2 - y1
            center_x = x1 + box_w / 2.0
            center_y = y1 + box_h / 2.0

            new_w = box_w * current_scale
            new_h = box_h * current_scale

            cx1 = int(center_x - new_w / 2.0)
            cy1 = int(center_y - new_h / 2.0)
            cx2 = int(cx1 + new_w)
            cy2 = int(cy1 + new_h)

            # Padding nếu vượt ngoài khung hình
            pad_left = max(0, -cx1)
            pad_top = max(0, -cy1)
            pad_right = max(0, cx2 - w)
            pad_bottom = max(0, cy2 - h)

            src_x1 = max(0, cx1)
            src_y1 = max(0, cy1)
            src_x2 = min(w, cx2)
            src_y2 = min(h, cy2)

            crop = frame[src_y1:src_y2, src_x1:src_x2]
            if pad_left > 0 or pad_top > 0 or pad_right > 0 or pad_bottom > 0:
                crop = cv2.copyMakeBorder(
                    crop,
                    pad_top, pad_bottom, pad_left, pad_right,
                    cv2.BORDER_CONSTANT,
                    value=[0, 0, 0]
                )

        return crop

    def predict_crop(self, face_crop: np.ndarray) -> Dict[str, Any]:
        """
        Dự đoán Anti-Spoof trực tiếp từ ảnh cắt khuôn mặt
        Returns:
            Dict chứa:
              - is_real: True nếu REAL, False nếu FAKE
              - label: 'REAL' hoặc 'FAKE'
              - confidence: xác suất của nhãn dự đoán (0.0 -> 1.0)
              - real_score: xác suất là REAL (0.0 -> 1.0)
              - fake_score: xác suất là FAKE (0.0 -> 1.0)
              - raw_logits: logits đầu ra từ model
        """
        tensor = self.preprocess_crop(face_crop)

        with torch.no_grad():
            logits = self.model(tensor)
            probs = torch.softmax(logits, dim=1).cpu().numpy()[0]

        fake_score = float(probs[0])
        real_score = float(probs[1])

        is_real = real_score >= self.real_threshold
        label = "REAL" if is_real else "FAKE"
        confidence = real_score if is_real else fake_score

        return {
            "is_real": is_real,
            "label": label,
            "confidence": confidence,
            "real_score": real_score,
            "fake_score": fake_score,
            "raw_logits": logits.cpu().numpy()[0].tolist()
        }

    def predict_face(self, frame: np.ndarray, bbox: List[int], scale: Optional[float] = None) -> Dict[str, Any]:
        """
        Cắt và dự đoán Anti-Spoof cho 1 bbox khuôn mặt trong frame
        """
        crop = self.crop_face(frame, bbox, scale=scale)
        if crop.size == 0:
            return {
                "is_real": False,
                "label": "UNKNOWN",
                "confidence": 0.0,
                "real_score": 0.0,
                "fake_score": 1.0,
                "face_crop": crop,
                "raw_logits": [0.0, 0.0]
            }

        result = self.predict_crop(crop)
        result["face_crop"] = crop
        result["bbox"] = bbox
        return result
