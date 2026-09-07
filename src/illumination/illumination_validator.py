# -*- coding: utf-8 -*-
"""
=============================================================================
Module: Illumination Quality Validator & Adaptive Low-Light Enhancer
=============================================================================
Đánh giá chất lượng ánh sáng khuôn mặt thời gian thực & Tăng cường sáng thích ứng:
  1. check_illumination_quality:
     - Đo độ rọi sáng (Mean Luminance L trong LAB space).
     - Nếu mean_luminance < 60.0 -> THIẾU SÁNG (chặn chụp ảnh).
  2. enhance_low_light:
     - Tăng cường vi vân bề mặt bằng CLAHE (Contrast Limited Adaptive Histogram Equalization).
=============================================================================
"""

import cv2
import numpy as np
from typing import Tuple, Dict, Any, Optional, List
from enum import Enum


class LightCategory(Enum):
    SEVERELY_DARK = "SEVERELY_DARK"    # Rất tối (< 40)
    LOW_LIGHT = "LOW_LIGHT"            # Thiếu sáng (< 60) -> Chặn chụp theo yêu cầu
    GOOD = "GOOD"                      # Ánh sáng tốt (>= 60 và <= 215) -> Cho chụp
    OVEREXPOSED = "OVEREXPOSED"        # Chói sáng quá mức (> 215)


def check_illumination_quality(
    image: np.ndarray,
    bbox: Optional[List[int]] = None,
    dark_threshold: float = 60.0,
    overexposed_threshold: float = 215.0
) -> Dict[str, Any]:
    """
    Đo lường độ rọi sáng khuôn mặt.
    Khi mean_luminance < dark_threshold (mặc định 60.0) -> is_acceptable = False (KHÔNG CHO CHỤP).
    """
    h, w = image.shape[:2]

    # Cắt vùng khuôn mặt nếu có bbox
    if bbox is not None:
        x1, y1, x2, y2 = bbox
        x1 = max(0, min(w - 1, int(x1)))
        y1 = max(0, min(h - 1, int(y1)))
        x2 = max(0, min(w, int(x2)))
        y2 = max(0, min(h, int(y2)))
        roi = image[y1:y2, x1:x2]
        if roi.size == 0:
            roi = image
    else:
        cy, cx = h // 2, w // 2
        ry, rx = int(h * 0.35), int(w * 0.25)
        y1, y2 = max(0, cy - ry), min(h, cy + ry)
        x1, x2 = max(0, cx - rx), min(w, cx + rx)
        roi = image[y1:y2, x1:x2]

    # Chuyển đổi sang không gian màu LAB (kênh L đo độ sáng thực)
    lab = cv2.cvtColor(roi, cv2.COLOR_BGR2LAB)
    l_channel = lab[:, :, 0]

    mean_l = float(np.mean(l_channel))
    std_l = float(np.std(l_channel))
    under_ratio = float(np.mean(l_channel < 25))

    if mean_l < 40.0:
        cat = LightCategory.SEVERELY_DARK
        is_acceptable = False
        msg = "ANH SANG RAT TOI - VUI LONG BAT DEN HOAC DEN NOI SANG"
    elif mean_l < dark_threshold:
        cat = LightCategory.LOW_LIGHT
        is_acceptable = False
        msg = "ANH SANG YEU (Thieu sang) - VUI LONG BAT DEN HOAC TIEN VE PHIA SANG"
    elif mean_l > overexposed_threshold:
        cat = LightCategory.OVEREXPOSED
        is_acceptable = True
        msg = "ANH SANG QUA MANH (Choi sang) - TRANH DEN ROI TRUC TIEP"
    else:
        cat = LightCategory.GOOD
        is_acceptable = True
        msg = "ANH SANG DAT TIEU CHUAN"

    return {
        "is_acceptable": is_acceptable,
        "is_low_light": (mean_l < dark_threshold),
        "category": cat,
        "mean_luminance": round(mean_l, 1),
        "contrast": round(std_l, 1),
        "under_ratio": round(under_ratio, 3),
        "status_msg": msg
    }


def enhance_low_light(
    image: np.ndarray,
    clip_limit: float = 2.5,
    grid_size: Tuple[int, int] = (8, 8)
) -> np.ndarray:
    """
    Tăng cường vi vân và độ tương phản khuôn mặt thích ứng bằng CLAHE trên kênh L.
    """
    if image is None or image.size == 0:
        return image

    lab = cv2.cvtColor(image, cv2.COLOR_BGR2LAB)
    l, a, b = cv2.split(lab)

    clahe = cv2.createCLAHE(clipLimit=clip_limit, tileGridSize=grid_size)
    cl = clahe.apply(l)

    enhanced_lab = cv2.merge((cl, a, b))
    return cv2.cvtColor(enhanced_lab, cv2.COLOR_LAB2BGR)
