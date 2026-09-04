# -*- coding: utf-8 -*-
"""
=============================================================================
Module: MiniFASNetV2 Official Architecture (Silent-Face-Anti-Spoofing)
=============================================================================
Kiến trúc chính thức của MiniFASNet (Minivision Silent-Face-Anti-Spoofing)
Tương thích 100% với các file weights pre-trained mã nguồn mở (1.8MB) như:
  - Anti_Spoof_minifasnetv2_(Copy).pth
  - 2.7_80x80_MiniFASNetV2.pth
  - 4_0_0_80x80_MiniFASNetV1SE.pth

Phân loại 3 classes:
  - Class 0: Spoof Attack (2D: ảnh in, giấy...)
  - Class 1: REAL / LIVE  (Khuôn mặt thật)
  - Class 2: Spoof Attack (3D: màn hình, video, mặt nạ...)
=============================================================================
"""

import os
import glob
import cv2
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import (
    Linear, Conv2d, BatchNorm1d, BatchNorm2d,
    PReLU, ReLU, Sigmoid, AdaptiveAvgPool2d, Sequential, Module
)
from typing import Optional, Union, Tuple, Dict, Any, List

# Tự động tìm thư mục gốc Face-Project linh hoạt
def _find_face_project_root():
    p = os.path.abspath(__file__)
    for _ in range(5):
        p = os.path.dirname(p)
        if os.path.exists(os.path.join(p, "models")):
            return p
    return os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

BASE_DIR = _find_face_project_root()


# =============================================================================
# 1. CORE BUILDING BLOCKS (OFFICIAL SILENT-FACE-ANTI-SPOOFING)
# =============================================================================
class L2Norm(Module):
    def forward(self, x):
        return F.normalize(x)


class Flatten(Module):
    def forward(self, x):
        return x.view(x.size(0), -1)


class Conv_block(Module):
    def __init__(self, in_c, out_c, kernel=(1, 1), stride=(1, 1), padding=(0, 0), groups=1):
        super(Conv_block, self).__init__()
        self.conv = Conv2d(
            in_c, out_c, kernel_size=kernel, groups=groups,
            stride=stride, padding=padding, bias=False
        )
        self.bn = BatchNorm2d(out_c)
        self.prelu = PReLU(out_c)

    def forward(self, x):
        return self.prelu(self.bn(self.conv(x)))


class Linear_block(Module):
    def __init__(self, in_c, out_c, kernel=(1, 1), stride=(1, 1), padding=(0, 0), groups=1):
        super(Linear_block, self).__init__()
        self.conv = Conv2d(
            in_c, out_channels=out_c, kernel_size=kernel,
            groups=groups, stride=stride, padding=padding, bias=False
        )
        self.bn = BatchNorm2d(out_c)

    def forward(self, x):
        return self.bn(self.conv(x))


class Depth_Wise(Module):
    def __init__(self, c1, c2, c3, residual=False, kernel=(3, 3), stride=(2, 2), padding=(1, 1), groups=1):
        super(Depth_Wise, self).__init__()
        c1_in, c1_out = c1
        c2_in, c2_out = c2
        c3_in, c3_out = c3
        self.conv = Conv_block(c1_in, out_c=c1_out, kernel=(1, 1), padding=(0, 0), stride=(1, 1))
        self.conv_dw = Conv_block(c2_in, c2_out, groups=c2_in, kernel=kernel, padding=padding, stride=stride)
        self.project = Linear_block(c3_in, c3_out, kernel=(1, 1), padding=(0, 0), stride=(1, 1))
        self.residual = residual

    def forward(self, x):
        short_cut = x
        x = self.conv(x)
        x = self.conv_dw(x)
        x = self.project(x)
        if self.residual:
            return short_cut + x
        return x


class Residual(Module):
    def __init__(self, c1, c2, c3, num_block, groups, kernel=(3, 3), stride=(1, 1), padding=(1, 1)):
        super(Residual, self).__init__()
        modules = []
        for i in range(num_block):
            c1_tuple = c1[i]
            c2_tuple = c2[i]
            c3_tuple = c3[i]
            modules.append(Depth_Wise(
                c1_tuple, c2_tuple, c3_tuple, residual=True,
                kernel=kernel, padding=padding, stride=stride, groups=groups
            ))
        self.model = Sequential(*modules)

    def forward(self, x):
        return self.model(x)


class SEModule(Module):
    def __init__(self, channels, reduction):
        super(SEModule, self).__init__()
        self.avg_pool = AdaptiveAvgPool2d(1)
        self.fc1 = Conv2d(channels, channels // reduction, kernel_size=1, padding=0, bias=False)
        self.bn1 = BatchNorm2d(channels // reduction)
        self.relu = ReLU(inplace=True)
        self.fc2 = Conv2d(channels // reduction, channels, kernel_size=1, padding=0, bias=False)
        self.bn2 = BatchNorm2d(channels)
        self.sigmoid = Sigmoid()

    def forward(self, x):
        module_input = x
        x = self.avg_pool(x)
        x = self.fc1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.fc2(x)
        x = self.bn2(x)
        x = self.sigmoid(x)
        return module_input * x


class Depth_Wise_SE(Module):
    def __init__(self, c1, c2, c3, residual=False, kernel=(3, 3), stride=(2, 2), padding=(1, 1), groups=1, se_reduct=8):
        super(Depth_Wise_SE, self).__init__()
        c1_in, c1_out = c1
        c2_in, c2_out = c2
        c3_in, c3_out = c3
        self.conv = Conv_block(c1_in, out_c=c1_out, kernel=(1, 1), padding=(0, 0), stride=(1, 1))
        self.conv_dw = Conv_block(c2_in, c2_out, groups=c2_in, kernel=kernel, padding=padding, stride=stride)
        self.project = Linear_block(c3_in, c3_out, kernel=(1, 1), padding=(0, 0), stride=(1, 1))
        self.residual = residual
        self.se_module = SEModule(c3_out, se_reduct)

    def forward(self, x):
        short_cut = x
        x = self.conv(x)
        x = self.conv_dw(x)
        x = self.project(x)
        if self.residual:
            x = self.se_module(x)
            return short_cut + x
        return x


class ResidualSE(Module):
    def __init__(self, c1, c2, c3, num_block, groups, kernel=(3, 3), stride=(1, 1), padding=(1, 1), se_reduct=4):
        super(ResidualSE, self).__init__()
        modules = []
        for i in range(num_block):
            c1_tuple = c1[i]
            c2_tuple = c2[i]
            c3_tuple = c3[i]
            if i == num_block - 1:
                modules.append(Depth_Wise_SE(
                    c1_tuple, c2_tuple, c3_tuple, residual=True,
                    kernel=kernel, padding=padding, stride=stride,
                    groups=groups, se_reduct=se_reduct
                ))
            else:
                modules.append(Depth_Wise(
                    c1_tuple, c2_tuple, c3_tuple, residual=True,
                    kernel=kernel, padding=padding, stride=stride, groups=groups
                ))
        self.model = Sequential(*modules)

    def forward(self, x):
        return self.model(x)


# =============================================================================
# 2. FULL NETWORK ARCHITECTURE
# =============================================================================
class MiniFASNet(Module):
    def __init__(self, keep, embedding_size=128, conv6_kernel=(5, 5),
                 drop_p=0.0, num_classes=3, img_channel=3):
        super(MiniFASNet, self).__init__()
        self.embedding_size = embedding_size

        self.conv1 = Conv_block(img_channel, keep[0], kernel=(3, 3), stride=(2, 2), padding=(1, 1))
        self.conv2_dw = Conv_block(keep[0], keep[1], kernel=(3, 3), stride=(1, 1), padding=(1, 1), groups=keep[1])

        c1 = [(keep[1], keep[2])]
        c2 = [(keep[2], keep[3])]
        c3 = [(keep[3], keep[4])]
        self.conv_23 = Depth_Wise(c1[0], c2[0], c3[0], kernel=(3, 3), stride=(2, 2), padding=(1, 1), groups=keep[3])

        c1 = [(keep[4], keep[5]), (keep[7], keep[8]), (keep[10], keep[11]), (keep[13], keep[14])]
        c2 = [(keep[5], keep[6]), (keep[8], keep[9]), (keep[11], keep[12]), (keep[14], keep[15])]
        c3 = [(keep[6], keep[7]), (keep[9], keep[10]), (keep[12], keep[13]), (keep[15], keep[16])]
        self.conv_3 = Residual(c1, c2, c3, num_block=4, groups=keep[4], kernel=(3, 3), stride=(1, 1), padding=(1, 1))

        c1 = [(keep[16], keep[17])]
        c2 = [(keep[17], keep[18])]
        c3 = [(keep[18], keep[19])]
        self.conv_34 = Depth_Wise(c1[0], c2[0], c3[0], kernel=(3, 3), stride=(2, 2), padding=(1, 1), groups=keep[19])

        c1 = [(keep[19], keep[20]), (keep[22], keep[23]), (keep[25], keep[26]), (keep[28], keep[29]),
              (keep[31], keep[32]), (keep[34], keep[35])]
        c2 = [(keep[20], keep[21]), (keep[23], keep[24]), (keep[26], keep[27]), (keep[29], keep[30]),
              (keep[32], keep[33]), (keep[35], keep[36])]
        c3 = [(keep[21], keep[22]), (keep[24], keep[25]), (keep[27], keep[28]), (keep[30], keep[31]),
              (keep[33], keep[34]), (keep[36], keep[37])]
        self.conv_4 = Residual(c1, c2, c3, num_block=6, groups=keep[19], kernel=(3, 3), stride=(1, 1), padding=(1, 1))

        c1 = [(keep[37], keep[38])]
        c2 = [(keep[38], keep[39])]
        c3 = [(keep[39], keep[40])]
        self.conv_45 = Depth_Wise(c1[0], c2[0], c3[0], kernel=(3, 3), stride=(2, 2), padding=(1, 1), groups=keep[40])

        c1 = [(keep[40], keep[41]), (keep[43], keep[44])]
        c2 = [(keep[41], keep[42]), (keep[44], keep[45])]
        c3 = [(keep[42], keep[43]), (keep[45], keep[46])]
        self.conv_5 = Residual(c1, c2, c3, num_block=2, groups=keep[40], kernel=(3, 3), stride=(1, 1), padding=(1, 1))

        self.conv_6_sep = Conv_block(keep[46], keep[47], kernel=(1, 1), stride=(1, 1), padding=(0, 0))
        self.conv_6_dw = Linear_block(keep[47], keep[48], groups=keep[48], kernel=conv6_kernel, stride=(1, 1), padding=(0, 0))
        self.conv_6_flatten = Flatten()
        self.linear = Linear(512, embedding_size, bias=False)
        self.bn = BatchNorm1d(embedding_size)
        self.drop = nn.Dropout(p=drop_p)
        self.prob = Linear(embedding_size, num_classes, bias=False)

    def forward(self, x):
        out = self.conv1(x)
        out = self.conv2_dw(out)
        out = self.conv_23(out)
        out = self.conv_3(out)
        out = self.conv_34(out)
        out = self.conv_4(out)
        out = self.conv_45(out)
        out = self.conv_5(out)
        out = self.conv_6_sep(out)
        out = self.conv_6_dw(out)
        out = self.conv_6_flatten(out)
        if self.embedding_size != 512:
            out = self.linear(out)
        out = self.bn(out)
        out = self.drop(out)
        out = self.prob(out)
        return out


class MiniFASNetSE(MiniFASNet):
    def __init__(self, keep, embedding_size=128, conv6_kernel=(5, 5),
                 drop_p=0.0, num_classes=3, img_channel=3):
        super(MiniFASNetSE, self).__init__(
            keep=keep, embedding_size=embedding_size, conv6_kernel=conv6_kernel,
            drop_p=drop_p, num_classes=num_classes, img_channel=img_channel
        )
        c1 = [(keep[4], keep[5]), (keep[7], keep[8]), (keep[10], keep[11]), (keep[13], keep[14])]
        c2 = [(keep[5], keep[6]), (keep[8], keep[9]), (keep[11], keep[12]), (keep[14], keep[15])]
        c3 = [(keep[6], keep[7]), (keep[9], keep[10]), (keep[12], keep[13]), (keep[15], keep[16])]
        self.conv_3 = ResidualSE(c1, c2, c3, num_block=4, groups=keep[4], kernel=(3, 3), stride=(1, 1), padding=(1, 1))

        c1 = [(keep[19], keep[20]), (keep[22], keep[23]), (keep[25], keep[26]), (keep[28], keep[29]),
              (keep[31], keep[32]), (keep[34], keep[35])]
        c2 = [(keep[20], keep[21]), (keep[23], keep[24]), (keep[26], keep[27]), (keep[29], keep[30]),
              (keep[32], keep[33]), (keep[35], keep[36])]
        c3 = [(keep[21], keep[22]), (keep[24], keep[25]), (keep[27], keep[28]), (keep[30], keep[31]),
              (keep[33], keep[34]), (keep[36], keep[37])]
        self.conv_4 = ResidualSE(c1, c2, c3, num_block=6, groups=keep[19], kernel=(3, 3), stride=(1, 1), padding=(1, 1))

        c1 = [(keep[40], keep[41]), (keep[43], keep[44])]
        c2 = [(keep[41], keep[42]), (keep[44], keep[45])]
        c3 = [(keep[42], keep[43]), (keep[45], keep[46])]
        self.conv_5 = ResidualSE(c1, c2, c3, num_block=2, groups=keep[40], kernel=(3, 3), stride=(1, 1), padding=(1, 1))


# Configurations
_keep_dict = {
    '1.8M': [32, 32, 103, 103, 64, 13, 13, 64, 26, 26,
             64, 13, 13, 64, 52, 52, 64, 231, 231, 128,
             154, 154, 128, 52, 52, 128, 26, 26, 128, 52,
             52, 128, 26, 26, 128, 26, 26, 128, 308, 308,
             128, 26, 26, 128, 26, 26, 128, 512, 512],

    '1.8M_': [32, 32, 103, 103, 64, 13, 13, 64, 13, 13, 64, 13,
              13, 64, 13, 13, 64, 231, 231, 128, 231, 231, 128, 52,
              52, 128, 26, 26, 128, 77, 77, 128, 26, 26, 128, 26, 26,
              128, 308, 308, 128, 26, 26, 128, 26, 26, 128, 512, 512]
}


def build_minifasnet_v1(conv6_kernel=(5, 5), num_classes=3) -> MiniFASNet:
    return MiniFASNet(_keep_dict['1.8M'], embedding_size=128, conv6_kernel=conv6_kernel, num_classes=num_classes)


def build_minifasnet_v2(conv6_kernel=(5, 5), num_classes=3) -> MiniFASNet:
    """Mô hình MiniFASNetV2 chính thức (khớp với Anti_Spoof_minifasnetv2_(Copy).pth)"""
    return MiniFASNet(_keep_dict['1.8M_'], embedding_size=128, conv6_kernel=conv6_kernel, num_classes=num_classes)


def build_minifasnet_v1_se(conv6_kernel=(5, 5), num_classes=3) -> MiniFASNetSE:
    return MiniFASNetSE(_keep_dict['1.8M'], embedding_size=128, conv6_kernel=conv6_kernel, num_classes=num_classes)


def build_minifasnet_v2_se(conv6_kernel=(5, 5), num_classes=3) -> MiniFASNetSE:
    return MiniFASNetSE(_keep_dict['1.8M_'], embedding_size=128, conv6_kernel=conv6_kernel, num_classes=num_classes)


# =============================================================================
# 3. HIGH-LEVEL OFFICIAL DETECTOR
# =============================================================================
def find_official_minifasnet_model() -> str:
    """Tìm file weights MiniFASNet chính thức trong models/"""
    candidates = [
        "2.7_80x80_MiniFASNetV2.pth",
        "4_0_0_80x80_MiniFASNetV1SE.pth",
        "Anti_Spoof_minifasnet.pth",
        "best_minifasnetv2.pth"
    ]
    for c in candidates:
        p = os.path.join(BASE_DIR, "models", c)
        if os.path.exists(p):
            return p

    # Quét tất cả file .pth
    pth_files = glob.glob(os.path.join(BASE_DIR, "models", "*minifas*.pth"))
    if pth_files:
        # Ưu tiên file lớn nhất (model chính thức ~1.85MB)
        return max(pth_files, key=os.path.getsize)

    return os.path.join(BASE_DIR, "models", "2.7_80x80_MiniFASNetV2.pth")


class AntiSpoofOfficial:
    """
    Bộ nhận diện Anti-Spoofing chính thức (Silent-Face-Anti-Spoofing MiniFASNetV2)
    Độ chính xác cao, nhận diện:
      - REAL  (Mặt người sống / Live)
      - SPOOF 2D (Ảnh in, giấy...)
      - SPOOF 3D (Màn hình điện thoại/máy tính, video phát lại...)
    """
    def __init__(
        self,
        model_path: Optional[str] = None,
        input_size: Tuple[int, int] = (80, 80),
        scale_factor: float = 2.7,  # Mặc định chuẩn của Silent-Face-Anti-Spoofing là 2.7x
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
        self.model_path = model_path if model_path else find_official_minifasnet_model()
        self.model_name = os.path.basename(self.model_path)

        # Nạp mô hình
        self.model = self._load_model()

    def _load_model(self) -> nn.Module:
        if not os.path.exists(self.model_path):
            raise FileNotFoundError(f"Không tìm thấy file weights tại: {self.model_path}")

        file_size_mb = os.path.getsize(self.model_path) / (1024.0 * 1024.0)
        print(f"\n[INFO] 📦 Nạp Official MiniFASNetV2: {self.model_name} ({file_size_mb:.2f} MB)")
        print(f"[INFO] 📍 Đường dẫn: {self.model_path}")
        print(f"[INFO] ⚙️ Thiết bị: {self.device}")

        raw_sd = torch.load(self.model_path, map_location=self.device)
        if isinstance(raw_sd, dict):
            if "model_state_dict" in raw_sd:
                raw_sd = raw_sd["model_state_dict"]
            elif "state_dict" in raw_sd:
                raw_sd = raw_sd["state_dict"]
            elif "model" in raw_sd:
                raw_sd = raw_sd["model"]
            elif "net" in raw_sd:
                raw_sd = raw_sd["net"]

        cleaned_sd = {}
        for k, v in raw_sd.items():
            name = k[7:] if k.startswith("module.") else k
            cleaned_sd[name] = v

        # Kiểm tra kích thước kernel conv6 dựa trên weights
        k_shape = cleaned_sd.get("conv_6_dw.conv.weight", None)
        conv6_k = (k_shape.shape[2], k_shape.shape[3]) if k_shape is not None else (5, 5)

        # Thử lần lượt các kiến trúc chính thức
        builders = [
            ("MiniFASNetV2", build_minifasnet_v2),
            ("MiniFASNetV1", build_minifasnet_v1),
            ("MiniFASNetV2SE", build_minifasnet_v2_se),
            ("MiniFASNetV1SE", build_minifasnet_v1_se),
        ]

        loaded_model = None
        for arch_name, builder_fn in builders:
            try:
                m = builder_fn(conv6_kernel=conv6_k, num_classes=3)
                m.load_state_dict(cleaned_sd, strict=True)
                m.to(self.device)
                m.eval()
                loaded_model = m
                print(f"[INFO] ✅ Nhận diện kiến trúc thành công: {arch_name} (Kernel: {conv6_k})")
                break
            except Exception:
                continue

        if loaded_model is None:
            # Fallback load với strict=False nếu có lệch nhỏ
            m = build_minifasnet_v2(conv6_kernel=conv6_k, num_classes=3)
            m.load_state_dict(cleaned_sd, strict=False)
            m.to(self.device)
            m.eval()
            loaded_model = m
            print(f"[WARN] ⚠️ Nạp với strict=False cho MiniFASNetV2")

        return loaded_model

    def preprocess_crop(self, face_img: np.ndarray) -> torch.Tensor:
        """
        Chuẩn hoá ảnh crop khuôn mặt về (80, 80) theo chuẩn ImageNet
        """
        if face_img is None or face_img.size == 0:
            raise ValueError("Ảnh crop rỗng!")

        resized = cv2.resize(face_img, self.input_size)
        rgb = cv2.cvtColor(resized, cv2.COLOR_BGR2RGB)
        tensor = rgb.astype(np.float32) / 255.0
        tensor = tensor.transpose((2, 0, 1))  # HWC -> CHW

        mean = np.array([0.485, 0.456, 0.406], dtype=np.float32).reshape(3, 1, 1)
        std = np.array([0.229, 0.224, 0.225], dtype=np.float32).reshape(3, 1, 1)
        tensor = (tensor - mean) / std

        tensor = torch.from_numpy(tensor).unsqueeze(0).to(self.device)
        return tensor

    def crop_face(self, frame: np.ndarray, bbox: List[int], scale: Optional[float] = None) -> np.ndarray:
        """
        Cắt khuôn mặt có scale mở rộng (chuẩn Silent-Face dùng scale 2.7x)
        """
        x1, y1, x2, y2 = bbox
        h, w = frame.shape[:2]
        s = scale if scale is not None else self.scale_factor

        bw = x2 - x1
        bh = y2 - y1
        cx = x1 + bw / 2.0
        cy = y1 + bh / 2.0

        new_w = bw * s
        new_h = bh * s

        cx1 = int(cx - new_w / 2.0)
        cy1 = int(cy - new_h / 2.0)
        cx2 = int(cx1 + new_w)
        cy2 = int(cy1 + new_h)

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
                crop, pad_top, pad_bottom, pad_left, pad_right,
                cv2.BORDER_CONSTANT, value=[0, 0, 0]
            )
        return crop

    def predict_crop(self, face_crop: np.ndarray) -> Dict[str, Any]:
        """
        Dự đoán Anti-Spoof trực tiếp từ ảnh cắt khuôn mặt
        """
        tensor = self.preprocess_crop(face_crop)

        with torch.no_grad():
            logits = self.model(tensor)
            probs = torch.softmax(logits, dim=-1).cpu().numpy()[0]

        # 3 Classes trong Silent-Face-Anti-Spoofing:
        # Class 0: 2D attack (Print)
        # Class 1: REAL / LIVE
        # Class 2: 3D attack (Screen/Video/Mask)
        p_spoof_2d = float(probs[0])
        p_real = float(probs[1])
        p_spoof_3d = float(probs[2]) if len(probs) > 2 else 0.0
        total_fake = p_spoof_2d + p_spoof_3d

        # Quyết định: REAL khi class 1 cao nhất hoặc vượt ngưỡng
        is_real = (probs.argmax() == 1) and (p_real >= self.real_threshold)
        label = "REAL" if is_real else "FAKE"
        confidence = p_real if is_real else total_fake

        return {
            "is_real": is_real,
            "label": label,
            "confidence": confidence,
            "real_score": p_real,
            "fake_score": total_fake,
            "class_scores": {
                "spoof_2d": p_spoof_2d,
                "real": p_real,
                "spoof_3d": p_spoof_3d
            },
            "raw_logits": logits.cpu().numpy()[0].tolist()
        }

    def predict_face(self, frame: np.ndarray, bbox: List[int], scale: Optional[float] = None) -> Dict[str, Any]:
        """
        Cắt và dự đoán Anti-Spoofing cho 1 khuôn mặt trong frame
        """
        crop = self.crop_face(frame, bbox, scale=scale)
        if crop.size == 0:
            return {
                "is_real": False,
                "label": "UNKNOWN",
                "confidence": 0.0,
                "real_score": 0.0,
                "fake_score": 1.0,
                "class_scores": {"spoof_2d": 0.5, "real": 0.0, "spoof_3d": 0.5},
                "face_crop": crop,
                "raw_logits": [0.0, 0.0, 0.0]
            }

        res = self.predict_crop(crop)
        res["face_crop"] = crop
        res["bbox"] = bbox
        return res


# =============================================================================
# 4. HIGH-LEVEL DUAL-MODEL ENSEMBLE DETECTOR
# =============================================================================
class OfficialImageCropper:
    """
    Bộ cắt ảnh mở rộng (Crop with scale) chuẩn theo Minivision Silent-Face
    """
    @staticmethod
    def crop(org_img: np.ndarray, bbox: List[int], scale: float, out_w: int = 80, out_h: int = 80) -> np.ndarray:
        x1, y1, x2, y2 = bbox
        box_w = x2 - x1
        box_h = y2 - y1
        src_h, src_w = org_img.shape[:2]

        if box_w <= 0 or box_h <= 0:
            return cv2.resize(org_img, (out_w, out_h))

        s = min((src_h - 1) / float(box_h), min((src_w - 1) / float(box_w), float(scale)))

        new_width = box_w * s
        new_height = box_h * s
        center_x = box_w / 2.0 + x1
        center_y = box_h / 2.0 + y1

        left_top_x = center_x - new_width / 2.0
        left_top_y = center_y - new_height / 2.0
        right_bottom_x = center_x + new_width / 2.0
        right_bottom_y = center_y + new_height / 2.0

        if left_top_x < 0:
            right_bottom_x -= left_top_x
            left_top_x = 0
        if left_top_y < 0:
            right_bottom_y -= left_top_y
            left_top_y = 0
        if right_bottom_x > src_w - 1:
            left_top_x -= (right_bottom_x - src_w + 1)
            right_bottom_x = src_w - 1
        if right_bottom_y > src_h - 1:
            left_top_y -= (right_bottom_y - src_h + 1)
            right_bottom_y = src_h - 1

        lx = max(0, int(round(left_top_x)))
        ly = max(0, int(round(left_top_y)))
        rx = min(src_w - 1, int(round(right_bottom_x)))
        ry = min(src_h - 1, int(round(right_bottom_y)))

        cropped = org_img[ly:ry + 1, lx:rx + 1]
        if cropped.size == 0:
            return cv2.resize(org_img, (out_w, out_h))

        return cv2.resize(cropped, (out_w, out_h), interpolation=cv2.INTER_LINEAR)


def find_official_ensemble_models() -> Tuple[str, str]:
    """Tìm 2 file model chính thức trong models/ hoặc Silent-Face-Anti-Spoofing-master"""
    search_dirs = [
        os.path.join(BASE_DIR, "models"),
        os.path.join(BASE_DIR, "Silent-Face-Anti-Spoofing-master", "resources", "anti_spoof_models"),
        os.path.join(BASE_DIR, "resources", "anti_spoof_models"),
    ]
    m1_name = "2.7_80x80_MiniFASNetV2.pth"
    m2_name = "4_0_0_80x80_MiniFASNetV1SE.pth"

    m1_path = None
    m2_path = None

    for d in search_dirs:
        p1 = os.path.join(d, m1_name)
        if m1_path is None and os.path.exists(p1):
            m1_path = p1
        p2 = os.path.join(d, m2_name)
        if m2_path is None and os.path.exists(p2):
            m2_path = p2

    if m1_path is None or m2_path is None:
        raise FileNotFoundError(
            f"Không tìm thấy đủ 2 file weights chính thức:\n"
            f"  - Model 1: {m1_name} -> {m1_path}\n"
            f"  - Model 2: {m2_name} -> {m2_path}"
        )
    return m1_path, m2_path


class AntiSpoofOfficialEnsemble:
    """
    Bộ nhận diện Anti-Spoofing kết hợp 2 Mô hình MiniFASNet chính thức:
      - Model 1: MiniFASNetV2 (Scale 2.7x)
      - Model 2: MiniFASNetV1SE (Scale 4.0x)
    """
    def __init__(
        self,
        model1_path: Optional[str] = None,
        model2_path: Optional[str] = None,
        real_threshold: float = 0.5,
        device: Optional[Union[str, torch.device]] = None
    ):
        if device is None:
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        elif isinstance(device, str):
            self.device = torch.device(device)
        else:
            self.device = device

        self.real_threshold = real_threshold
        self.cropper = OfficialImageCropper()

        if model1_path is None or model2_path is None:
            p1, p2 = find_official_ensemble_models()
            self.model1_path = model1_path or p1
            self.model2_path = model2_path or p2
        else:
            self.model1_path = model1_path
            self.model2_path = model2_path

        self._load_models()

    def _load_single_weights(self, model: nn.Module, path: str):
        state_dict = torch.load(path, map_location=self.device)
        if isinstance(state_dict, dict):
            if "model_state_dict" in state_dict:
                state_dict = state_dict["model_state_dict"]
            elif "state_dict" in state_dict:
                state_dict = state_dict["state_dict"]
            elif "model" in state_dict:
                state_dict = state_dict["model"]

        cleaned_sd = {}
        for k, v in state_dict.items():
            name = k[7:] if k.startswith("module.") else k
            cleaned_sd[name] = v

        model.load_state_dict(cleaned_sd, strict=True)
        model.to(self.device)
        model.eval()

    def _load_models(self):
        print("\n" + "=" * 80)
        print(" 🚀 ĐANG NẠP 2 MÔ HÌNH CHÍNH THỨC (SILENT-FACE-ANTI-SPOOFING ENSEMBLE)")
        print("=" * 80)
        print(f"[INFO] 📍 Model 1 (2.7x): {os.path.basename(self.model1_path)} ({os.path.getsize(self.model1_path)/(1024*1024):.2f} MB)")
        print(f"[INFO] 📍 Model 2 (4.0x): {os.path.basename(self.model2_path)} ({os.path.getsize(self.model2_path)/(1024*1024):.2f} MB)")
        print(f"[INFO] ⚙️ Thiết bị xử lý : {self.device}")

        # Model 1: MiniFASNetV2, conv6=(5, 5), scale=2.7
        self.model1 = build_minifasnet_v2(conv6_kernel=(5, 5), num_classes=3)
        self._load_single_weights(self.model1, self.model1_path)

        # Model 2: MiniFASNetV1SE, conv6=(5, 5), scale=4.0
        self.model2 = build_minifasnet_v1_se(conv6_kernel=(5, 5), num_classes=3)
        self._load_single_weights(self.model2, self.model2_path)

        print("[INFO] ✅ Đã nạp thành công 2 mô hình vào bộ nhớ!")
        print("=" * 80 + "\n")

    def _to_tensor(self, crop_bgr: np.ndarray) -> torch.Tensor:
        """Chuẩn hóa tensor chuẩn Silent-Face (HWC BGR -> CHW Float32 [0..255])"""
        tensor = torch.from_numpy(crop_bgr.transpose((2, 0, 1))).float()
        return tensor.unsqueeze(0).to(self.device)

    def predict_face(self, frame: np.ndarray, bbox: List[int]) -> Dict[str, Any]:
        """Cắt ảnh theo 2 tỷ lệ (2.7x và 4.0x) và tính xác suất Ensemble trung bình"""
        crop_27 = self.cropper.crop(frame, bbox, scale=2.7, out_w=80, out_h=80)
        crop_40 = self.cropper.crop(frame, bbox, scale=4.0, out_w=80, out_h=80)

        t1 = self._to_tensor(crop_27)
        t2 = self._to_tensor(crop_40)

        with torch.no_grad():
            logits1 = self.model1(t1)
            probs1 = torch.softmax(logits1, dim=-1).cpu().numpy()[0]

            logits2 = self.model2(t2)
            probs2 = torch.softmax(logits2, dim=-1).cpu().numpy()[0]

        ensemble_probs = (probs1 + probs2) / 2.0

        p_spoof_2d = float(ensemble_probs[0])
        p_real = float(ensemble_probs[1])
        p_spoof_3d = float(ensemble_probs[2])
        total_fake = p_spoof_2d + p_spoof_3d

        pred_class = int(np.argmax(ensemble_probs))
        is_real = (pred_class == 1) and (p_real >= self.real_threshold)
        label = "REAL" if is_real else "FAKE"
        confidence = p_real if is_real else (total_fake if pred_class != 1 else p_real)

        return {
            "is_real": is_real,
            "label": label,
            "confidence": confidence,
            "real_score": p_real,
            "fake_score": total_fake,
            "class_scores": {
                "spoof_2d": p_spoof_2d,
                "real": p_real,
                "spoof_3d": p_spoof_3d,
            },
            "model1_scores": {
                "spoof_2d": float(probs1[0]),
                "real": float(probs1[1]),
                "spoof_3d": float(probs1[2]),
            },
            "model2_scores": {
                "spoof_2d": float(probs2[0]),
                "real": float(probs2[1]),
                "spoof_3d": float(probs2[2]),
            },
            "crop_27": crop_27,
            "crop_40": crop_40,
            "face_crop": crop_27,
            "bbox": bbox,
        }
