import sys
import os
import json
import glob
import datetime
import cv2
import numpy as np
import torch
from typing import Optional, Union, Tuple, Dict, Any, List

if hasattr(sys.stdout, "reconfigure"):
    try:
        sys.stdout.reconfigure(encoding="utf-8")
        sys.stderr.reconfigure(encoding="utf-8")
    except Exception:
        pass

# Tự động tìm thư mục gốc Face-Project
BASE_DIR = os.path.dirname(
    os.path.dirname(
        os.path.dirname(os.path.abspath(__file__))
    )
)


def find_default_mobilenetv2_model() -> str:
    """
    Tự động tìm thư mục chứa model MobileNetV2 trong thư mục models/
    """
    candidate = os.path.join(BASE_DIR, "models", "Model_MobilenetV2")
    if os.path.exists(candidate):
        return candidate

    models_dir = os.path.join(BASE_DIR, "models")
    if os.path.exists(models_dir):
        for item in os.listdir(models_dir):
            item_path = os.path.join(models_dir, item)
            if os.path.isdir(item_path) and "mobilenet" in item.lower():
                return item_path

    raise FileNotFoundError(
        f"Không tìm thấy thư mục model MobileNetV2 trong {os.path.join(BASE_DIR, 'models')}!"
    )


def resolve_mobilenetv2_paths(model_path_or_dir: Optional[str] = None) -> Tuple[str, str]:
    """
    Tự động tìm cặp (weight_file, config_file) MỚI NHẤT trong thư mục model.
    Hỗ trợ cả các tên file được đánh số thứ tự tải về:
      - model (1).safetensors, model (2).safetensors...
      - config (1).json, config (2).json...
    Ưu tiên file có thời gian chỉnh sửa mới nhất (LastWriteTime).
    """
    target = model_path_or_dir if model_path_or_dir else find_default_mobilenetv2_model()

    if os.path.isfile(target):
        # Người dùng truyền thẳng 1 file weights
        weight_file = os.path.abspath(target)
        directory = os.path.dirname(weight_file)
    else:
        directory = os.path.abspath(target)
        # Quét tất cả file .safetensors và .bin trong thư mục
        weight_candidates = glob.glob(os.path.join(directory, "*.safetensors"))
        if not weight_candidates:
            weight_candidates = glob.glob(os.path.join(directory, "*.bin"))

        if not weight_candidates:
            raise FileNotFoundError(f"Không tìm thấy file weights (.safetensors/.bin) nào trong {directory}!")

        # Sắp xếp theo thời gian sửa đổi mới nhất
        weight_candidates.sort(key=lambda p: os.path.getmtime(p), reverse=True)
        weight_file = weight_candidates[0]

    # Tìm file config tương ứng
    config_candidates = glob.glob(os.path.join(directory, "config*.json"))
    if not config_candidates:
        config_candidates = glob.glob(os.path.join(directory, "*.json"))

    if config_candidates:
        # Sắp xếp theo thời gian mới nhất
        config_candidates.sort(key=lambda p: os.path.getmtime(p), reverse=True)
        config_file = config_candidates[0]
    else:
        config_file = os.path.join(directory, "config.json")

    return weight_file, config_file


class AntiSpoofMobileNetV2:
    """
    Bộ nhận diện Face Anti-Spoofing sử dụng mạng MobileNetV2 (Hugging Face Transformers / Safetensors)
    Phân loại khuôn mặt:
      - REAL / LIVE (0): Người thật trước camera
      - FAKE / SPOOF (1): Giả mạo (ảnh in giấy, màn hình điện thoại/máy tính, video phát lại, mặt nạ...)
    """
    def __init__(
        self,
        model_path: Optional[str] = None,
        input_size: Optional[Tuple[int, int]] = None,
        scale_factor: float = 1.2,
        real_threshold: float = 0.5,
        device: Optional[Union[str, torch.device]] = None
    ):
        if device is None:
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        elif isinstance(device, str):
            self.device = torch.device(device)
        else:
            self.device = device

        # Tự động phân giải đường dẫn weights và config mới nhất
        self.weight_file, self.config_file = resolve_mobilenetv2_paths(model_path)
        self.model_path = os.path.dirname(self.weight_file)
        self.weight_name = os.path.basename(self.weight_file)
        self.config_name = os.path.basename(self.config_file)
        self.model_name = self.weight_name

        self.scale_factor = scale_factor
        self.real_threshold = real_threshold

        mtime = datetime.datetime.fromtimestamp(os.path.getmtime(self.weight_file)).strftime("%d/%m/%Y %H:%M:%S")
        print(f"[INFO] 🔍 Đã chọn weights MobileNetV2 mới nhất: {self.weight_name} (Sửa đổi: {mtime})")

        # Đọc config
        self.config = self._load_config()
        configured_size = self.config.get("image_size", 224) if self.config else 224
        self.input_size = input_size if input_size is not None else (configured_size, configured_size)

        # Mapping nhãn từ config
        self.id2label = self.config.get("id2label", {"0": "live", "1": "spoof"}) if self.config else {"0": "live", "1": "spoof"}
        self.live_index = 0
        self.spoof_index = 1
        for k, v in self.id2label.items():
            if str(v).lower() in ["live", "real"]:
                self.live_index = int(k)
            elif str(v).lower() in ["spoof", "fake"]:
                self.spoof_index = int(k)

        # Tải mô hình
        self.model = self._load_model()
        print(f"[INFO] AntiSpoofMobileNetV2 đã sẵn sàng trên thiết bị [{self.device.type.upper()}] | Input: {self.input_size[0]}x{self.input_size[1]}")

    def _load_config(self) -> Optional[Dict[str, Any]]:
        """Đọc file json config"""
        if os.path.exists(self.config_file):
            try:
                with open(self.config_file, "r", encoding="utf-8") as f:
                    return json.load(f)
            except Exception as e:
                print(f"[WARNING] Không thể đọc config file {self.config_file}: {e}")
        return None

    def _load_model(self) -> torch.nn.Module:
        """
        Tải mô hình MobileNetV2ForImageClassification từ config và file weights mới nhất
        """
        print(f"[INFO] Đang nạp weights từ: {self.weight_file} ...")
        from transformers import MobileNetV2Config, MobileNetV2ForImageClassification
        from safetensors.torch import load_file

        try:
            if os.path.exists(self.config_file):
                hf_config = MobileNetV2Config.from_json_file(self.config_file)
                model = MobileNetV2ForImageClassification(hf_config)
            else:
                model = MobileNetV2ForImageClassification.from_pretrained(self.model_path)

            if self.weight_file.endswith(".safetensors"):
                weights = load_file(self.weight_file)
                model.load_state_dict(weights)
            elif self.weight_file.endswith(".bin"):
                weights = torch.load(self.weight_file, map_location="cpu")
                model.load_state_dict(weights)

            model.to(self.device)
            model.eval()
            print(f"[INFO] [OK] Tải model MobileNetV2 ({self.weight_name}) thành công!")
            return model
        except Exception as e:
            # Fallback về phương thức chuẩn nếu có trục trặc
            print(f"[WARNING] Thử load qua from_pretrained do: {e}")
            model = MobileNetV2ForImageClassification.from_pretrained(self.model_path)
            model.to(self.device)
            model.eval()
            return model

    def preprocess_crop(self, face_img: np.ndarray) -> torch.Tensor:
        """
        Tiền xử lý ảnh cắt khuôn mặt:
          1. Resize về kích thước chuẩn input_size (mặc định 224x224)
          2. Chuyển hệ màu BGR -> RGB
          3. Chuẩn hóa [0, 255] -> [0.0, 1.0] và HWC -> CHW
          4. ImageNet Normalize: mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]
        """
        if face_img is None or face_img.size == 0:
            raise ValueError("Ảnh crop khuôn mặt rỗng!")

        resized = cv2.resize(face_img, self.input_size, interpolation=cv2.INTER_LINEAR)
        rgb = cv2.cvtColor(resized, cv2.COLOR_BGR2RGB)

        tensor = rgb.astype(np.float32) / 255.0
        tensor = tensor.transpose((2, 0, 1))

        mean = np.array([0.485, 0.456, 0.406], dtype=np.float32).reshape(3, 1, 1)
        std = np.array([0.229, 0.224, 0.225], dtype=np.float32).reshape(3, 1, 1)
        tensor = (tensor - mean) / std

        tensor = torch.from_numpy(tensor).unsqueeze(0).to(self.device)
        return tensor

    def crop_face(self, frame: np.ndarray, bbox: List[int], scale: Optional[float] = None) -> np.ndarray:
        """
        Cắt khuôn mặt từ frame theo bounding box [x1, y1, x2, y2]
        Hỗ trợ mở rộng tỷ lệ (scale factor) quanh tâm khuôn mặt kèm padding an toàn
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
            box_w = x2 - x1
            box_h = y2 - y1
            center_x = x1 + box_w / 2.0
            center_y = y1 + box_h / 2.0

            new_w = box_w * current_scale
            new_h = box_h * current_scale

            cx1 = int(round(center_x - new_w / 2.0))
            cy1 = int(round(center_y - new_h / 2.0))
            cx2 = int(round(cx1 + new_w))
            cy2 = int(round(cy1 + new_h))

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
        """
        tensor = self.preprocess_crop(face_crop)

        with torch.no_grad():
            outputs = self.model(tensor)
            logits = outputs.logits
            probs = torch.softmax(logits, dim=-1).cpu().numpy()[0]

        real_score = float(probs[self.live_index])
        fake_score = float(probs[self.spoof_index])

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
        Cắt và dự đoán Anti-Spoof cho 1 bbox khuôn mặt [x1, y1, x2, y2] trong frame
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
                "raw_logits": [0.0, 0.0],
                "bbox": bbox
            }

        result = self.predict_crop(crop)
        result["face_crop"] = crop
        result["bbox"] = bbox
        return result
