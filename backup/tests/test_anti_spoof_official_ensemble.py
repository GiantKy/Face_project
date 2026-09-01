# -*- coding: utf-8 -*-
"""
=============================================================================
Test Module: Dual-Model Official Ensemble Anti-Spoofing (Silent-Face-Anti-Spoofing)
=============================================================================
Kiểm thử kết hợp đồng thời cả 2 Model MiniFASNet chính thức từ thư mục gốc:
  1. Model 1: 2.7_80x80_MiniFASNetV2.pth     (Scale 2.7x - Vùng mặt gần)
  2. Model 2: 4_0_0_80x80_MiniFASNetV1SE.pth (Scale 4.0x - Vùng ngữ cảnh rộng)

Tổng hợp xác suất (Ensemble Average) để phân loại chính xác 3 Classes:
  - REAL      : Khuôn mặt người thật (Live)
  - SPOOF 2D  : Ảnh in, giấy, poster, sticker...
  - SPOOF 3D  : Màn hình điện thoại, máy tính bảng, video phát lại, mặt nạ...

Tính năng kiểm thử:
  1. Real-time Webcam (Mặc định):
     - Vẽ bounding box & Corner brackets (Xanh lá = REAL, Đỏ = FAKE/SPOOF).
     - Hiển thị bảng điểm Ensemble: [Real %, 2D Spoof %, 3D Spoof %].
     - Hiển thị điểm riêng biệt của từng Model (Model 2.7x vs Model 4.0x).
     - Hiển thị đồng thời 2 Thumbnail crop (2.7x & 4.0x).
     - Đo FPS và thời gian inference (ms).
  2. Kiểm thử ảnh đơn lẻ (--image <path>) hoặc thư mục ảnh (--dir <path>).

Phím tắt trong chế độ Webcam:
  - ESC / Q   : Thoát chương trình
  - S         : Lưu ảnh chụp màn hình vào backup/tests/output/official_ensemble/
  - C         : Bật / Tắt hiển thị thumbnail crops
  - SPACE     : Tạm dừng / Tiếp tục webcam
=============================================================================
"""

import sys
import os

if hasattr(sys.stdout, "reconfigure"):
    try:
        sys.stdout.reconfigure(encoding="utf-8")
        sys.stderr.reconfigure(encoding="utf-8")
    except Exception:
        pass

import time
import argparse
import glob
from pathlib import Path
from typing import Optional, Union, Tuple, Dict, Any, List
import cv2
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

# Thiết lập đường dẫn import tới thư mục gốc Face-Project
CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))  # backup/tests/
BACKUP_DIR = os.path.dirname(CURRENT_DIR)                 # backup/
PROJECT_DIR = os.path.dirname(BACKUP_DIR)                 # Face-Project/

for p in [PROJECT_DIR, BACKUP_DIR]:
    if p not in sys.path:
        sys.path.insert(0, p)

from src.face_detection import FaceDetector
from src.anti_spoof.minifasnet_official import build_minifasnet_v2, build_minifasnet_v1_se

OUTPUT_DIR = os.path.join(CURRENT_DIR, "output", "official_ensemble")
os.makedirs(OUTPUT_DIR, exist_ok=True)


# =============================================================================
# 1. TÌM KIẾM ĐƯỜNG DẪN 2 MODEL CHÍNH THỨC
# =============================================================================
def find_model_paths() -> Tuple[str, str]:
    """
    Tự động tìm kiếm đường dẫn 2 file weights chính thức trong project.
    """
    search_dirs = [
        os.path.join(PROJECT_DIR, "Silent-Face-Anti-Spoofing-master", "resources", "anti_spoof_models"),
        os.path.join(PROJECT_DIR, "models"),
        os.path.join(BACKUP_DIR, "models"),
        os.path.join(PROJECT_DIR, "resources", "anti_spoof_models"),
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
            f"  - Model 2: {m2_name} -> {m2_path}\n"
            f"Vui lòng kiểm tra thư mục 'Silent-Face-Anti-Spoofing-master/resources/anti_spoof_models/'."
        )

    return m1_path, m2_path


# =============================================================================
# 2. CROP IMAGE THEO CHUẨN SILENT-FACE-ANTI-SPOOFING
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


# =============================================================================
# 3. CLASS DUAL-MODEL ENSEMBLE DETECTOR
# =============================================================================
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
            p1, p2 = find_model_paths()
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
        """
        Chuẩn hóa tensor chuẩn Silent-Face (HWC BGR -> CHW Float32 [0..255])
        """
        tensor = torch.from_numpy(crop_bgr.transpose((2, 0, 1))).float()
        return tensor.unsqueeze(0).to(self.device)

    def predict_face(self, frame: np.ndarray, bbox: List[int]) -> Dict[str, Any]:
        """
        Cắt ảnh theo 2 tỷ lệ (2.7x và 4.0x) và tính xác suất Ensemble trung bình
        """
        # Crop 2.7x (Model 1)
        crop_27 = self.cropper.crop(frame, bbox, scale=2.7, out_w=80, out_h=80)
        # Crop 4.0x (Model 2)
        crop_40 = self.cropper.crop(frame, bbox, scale=4.0, out_w=80, out_h=80)

        t1 = self._to_tensor(crop_27)
        t2 = self._to_tensor(crop_40)

        with torch.no_grad():
            logits1 = self.model1(t1)
            probs1 = torch.softmax(logits1, dim=-1).cpu().numpy()[0]

            logits2 = self.model2(t2)
            probs2 = torch.softmax(logits2, dim=-1).cpu().numpy()[0]

        # Trung bình cộng 2 mô hình (Ensemble)
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
            "bbox": bbox,
        }


# =============================================================================
# 4. GIAO DIỆN VẼ OVERLAY TRỰC QUAN (MODERN CARD OVERLAY UI)
# =============================================================================
def draw_card_overlay(img: np.ndarray, x: int, y: int, w: int, h: int, bg_color=(20, 20, 20), alpha=0.72):
    """Vẽ khung card bán trong suốt"""
    sub_img = img[y:y+h, x:x+w]
    if sub_img.size == 0:
        return
    rect = np.full_like(sub_img, bg_color, dtype=np.uint8)
    res = cv2.addWeighted(sub_img, 1.0 - alpha, rect, alpha, 1.0)
    img[y:y+h, x:x+w] = res


def draw_ensemble_face_ui(
    frame: np.ndarray,
    face_idx: int,
    bbox: list,
    res: dict,
    show_crop_thumb: bool = True,
    thumb_x_offset: int = 0
):
    """
    Vẽ Bounding box, Corner Brackets, Thẻ Card thông tin Ensemble và 2 Thumbnail crop
    """
    x1, y1, x2, y2 = bbox
    is_real = res["is_real"]
    label = res["label"]
    conf = res["confidence"]
    real_score = res["real_score"]
    c_scores = res.get("class_scores", {})
    p_2d = c_scores.get("spoof_2d", 0.0)
    p_3d = c_scores.get("spoof_3d", 0.0)
    m1_s = res.get("model1_scores", {})
    m2_s = res.get("model2_scores", {})

    # Màu chủ đạo: Xanh ngọc (REAL) vs Đỏ cam (FAKE)
    main_color = (46, 204, 113) if is_real else (60, 76, 231)  # BGR
    border_color = (0, 255, 127) if is_real else (0, 0, 255)

    # 1. Bounding box & Corner Brackets
    cv2.rectangle(frame, (x1, y1), (x2, y2), main_color, 2)
    line_len = min(22, (x2 - x1) // 4, (y2 - y1) // 4)
    thick = 3
    cv2.line(frame, (x1, y1), (x1 + line_len, y1), border_color, thick)
    cv2.line(frame, (x1, y1), (x1, y1 + line_len), border_color, thick)
    cv2.line(frame, (x2, y1), (x2 - line_len, y1), border_color, thick)
    cv2.line(frame, (x2, y1), (x2, y1 + line_len), border_color, thick)
    cv2.line(frame, (x1, y2), (x1 + line_len, y2), border_color, thick)
    cv2.line(frame, (x1, y2), (x1, y2 - line_len), border_color, thick)
    cv2.line(frame, (x2, y2), (x2 - line_len, y2), border_color, thick)
    cv2.line(frame, (x2, y2), (x2, y2 - line_len), border_color, thick)

    # 2. Pill Badge phía trên Box
    badge_text = f" #{face_idx} {label} {conf*100:.1f}% "
    font = cv2.FONT_HERSHEY_DUPLEX
    font_scale = 0.55
    (tw, th), tb = cv2.getTextSize(badge_text, font, font_scale, 1)

    badge_x = x1
    badge_y = max(th + 10, y1 - 8)
    bx1 = badge_x
    by1 = badge_y - th - 6
    bx2 = badge_x + tw + 10
    by2 = badge_y + 4

    cv2.rectangle(frame, (bx1, by1), (bx2, by2), main_color, cv2.FILLED)
    cv2.putText(frame, badge_text, (bx1 + 3, badge_y - 2), font, font_scale, (255, 255, 255), 1, cv2.LINE_AA)

    # 3. Card chi tiết Ensemble bên dưới Box
    card_w = max(230, x2 - x1)
    card_h = 105
    card_x = x1
    card_y = y2 + 6
    h_frame, w_frame = frame.shape[:2]

    if card_y + card_h > h_frame:
        card_y = max(10, y1 - card_h - 25)
    if card_x + card_w > w_frame:
        card_x = max(10, w_frame - card_w - 10)

    draw_card_overlay(frame, card_x, card_y, card_w, card_h, bg_color=(15, 15, 15), alpha=0.75)
    cv2.rectangle(frame, (card_x, card_y), (card_x + card_w, card_y + card_h), (80, 80, 80), 1)

    # Thanh đo xác suất Real %
    bar_x = card_x + 10
    bar_y = card_y + 20
    bar_w = card_w - 20
    bar_h = 8
    cv2.rectangle(frame, (bar_x, bar_y), (bar_x + bar_w, bar_y + bar_h), (50, 50, 50), cv2.FILLED)
    real_bar_len = int(bar_w * real_score)
    if real_bar_len > 0:
        cv2.rectangle(frame, (bar_x, bar_y), (bar_x + real_bar_len, bar_y + bar_h), (46, 204, 113), cv2.FILLED)

    # Văn bản thông số Ensemble
    cv2.putText(frame, f"REAL (Live): {real_score*100:.1f}%", (card_x + 10, card_y + 14), font, 0.38, (46, 204, 113), 1, cv2.LINE_AA)
    cv2.putText(frame, f"2D Paper: {p_2d*100:.1f}%  |  3D Screen: {p_3d*100:.1f}%", (card_x + 10, card_y + 42), font, 0.38, (200, 200, 200), 1, cv2.LINE_AA)

    # Điểm riêng của từng model
    m1_real = m1_s.get("real", 0.0) * 100.0
    m2_real = m2_s.get("real", 0.0) * 100.0
    cv2.putText(frame, f"M1 (2.7x): {m1_real:.1f}%  |  M2 (4.0x): {m2_real:.1f}%", (card_x + 10, card_y + 64), font, 0.36, (140, 200, 255), 1, cv2.LINE_AA)

    status_tag = "[ ENSEMBLE: AUTHENTIC ]" if is_real else "[ ENSEMBLE: SPOOF DETECTED ]"
    tag_color = (46, 204, 113) if is_real else (60, 76, 231)
    cv2.putText(frame, status_tag, (card_x + 10, card_y + 88), font, 0.38, tag_color, 1, cv2.LINE_AA)

    # 4. Hiển thị 2 Thumbnail crop ở góc trên bên phải
    if show_crop_thumb:
        crop27 = res.get("crop_27", None)
        crop40 = res.get("crop_40", None)
        thumb_size = 72
        pad = 8
        ty = 10 + thumb_x_offset
        tx2 = w_frame - pad - thumb_size
        tx1 = tx2 - pad - thumb_size

        if tx1 > 0 and ty + thumb_size + 24 < h_frame:
            # Thumbnail 1 (2.7x)
            if crop27 is not None:
                t_img1 = cv2.resize(crop27, (thumb_size, thumb_size))
                frame[ty:ty+thumb_size, tx1:tx1+thumb_size] = t_img1
                cv2.rectangle(frame, (tx1, ty), (tx1+thumb_size, ty+thumb_size), (0, 255, 127), 1)
                draw_card_overlay(frame, tx1, ty+thumb_size, thumb_size, 16, bg_color=(0, 0, 0), alpha=0.8)
                cv2.putText(frame, "Crop 2.7x", (tx1+4, ty+thumb_size+12), font, 0.32, (255, 255, 255), 1, cv2.LINE_AA)

            # Thumbnail 2 (4.0x)
            if crop40 is not None:
                t_img2 = cv2.resize(crop40, (thumb_size, thumb_size))
                frame[ty:ty+thumb_size, tx2:tx2+thumb_size] = t_img2
                cv2.rectangle(frame, (tx2, ty), (tx2+thumb_size, ty+thumb_size), (255, 200, 0), 1)
                draw_card_overlay(frame, tx2, ty+thumb_size, thumb_size, 16, bg_color=(0, 0, 0), alpha=0.8)
                cv2.putText(frame, "Crop 4.0x", (tx2+4, ty+thumb_size+12), font, 0.32, (255, 255, 255), 1, cv2.LINE_AA)


# =============================================================================
# 5. RUN IMAGE / DATASET TEST
# =============================================================================
def run_image_test(image_paths: List[str], detector: FaceDetector, ensemble: AntiSpoofOfficialEnsemble, save_output: bool = True):
    print("\n" + "=" * 105)
    print(" 📊 KIỂM THỬ ANTI-SPOOF DUAL-MODEL ENSEMBLE TRÊN DANH SÁCH ẢNH")
    print("=" * 105)

    total_images = len(image_paths)
    total_faces = 0
    real_count = 0
    fake_count = 0

    header = f"{'STT':<4} | {'TÊN ẢNH':<22} | {'MẶT':<4} | {'KẾT QUẢ':<8} | {'REAL %':<8} | {'2D SPOOF %':<11} | {'3D SPOOF %':<11} | {'M1(2.7x)':<9} | {'M2(4.0x)':<9} | {'TIME(ms)':<8}"
    print(header)
    print("-" * 115)

    for img_idx, img_p in enumerate(image_paths, 1):
        filename = os.path.basename(img_p)
        frame = cv2.imread(img_p)
        if frame is None:
            print(f"{img_idx:<4} | {filename:<22} | [LỖI: Không thể đọc file ảnh]")
            continue

        faces = detector.detect(frame)
        if not faces:
            print(f"{img_idx:<4} | {filename:<22} | 0    | NO FACE  | -        | -           | -           | -         | -         | -")
            continue

        annotated = frame.copy()
        for f_idx, face in enumerate(faces, 1):
            total_faces += 1
            bbox = face["bbox"]

            t0 = time.perf_counter()
            res = ensemble.predict_face(frame, bbox)
            infer_ms = (time.perf_counter() - t0) * 1000.0

            if res["is_real"]:
                real_count += 1
            else:
                fake_count += 1

            lbl = res["label"]
            r_pct = res["real_score"] * 100.0
            cs = res.get("class_scores", {})
            p2_pct = cs.get("spoof_2d", 0.0) * 100.0
            p3_pct = cs.get("spoof_3d", 0.0) * 100.0
            m1_r = res.get("model1_scores", {}).get("real", 0.0) * 100.0
            m2_r = res.get("model2_scores", {}).get("real", 0.0) * 100.0

            print(f"{img_idx:<4} | {filename:<22} | #{f_idx:<3} | {lbl:<8} | {r_pct:5.1f}%  | {p2_pct:5.1f}%      | {p3_pct:5.1f}%      | {m1_r:5.1f}%   | {m2_r:5.1f}%   | {infer_ms:6.1f}")

            draw_ensemble_face_ui(
                annotated, f_idx, bbox, res,
                show_crop_thumb=True, thumb_x_offset=(f_idx - 1) * 110
            )

        if save_output:
            out_file = os.path.join(OUTPUT_DIR, f"ensemble_{filename}")
            cv2.imwrite(out_file, annotated)

    print("-" * 115)
    print(f"[TỔNG KẾT] Đã kiểm thử: {total_images} ảnh | Tổng số khuôn mặt: {total_faces}")
    print(f"           - REAL (Thật) : {real_count} ({real_count/max(1, total_faces)*100:.1f}%)")
    print(f"           - FAKE (Giả mạo): {fake_count} ({fake_count/max(1, total_faces)*100:.1f}%)")
    if save_output:
        print(f"[INFO] Ảnh kết quả đã được lưu tại: {OUTPUT_DIR}")
    print("=" * 105 + "\n")


# =============================================================================
# 6. RUN REAL-TIME WEBCAM TEST
# =============================================================================
def run_webcam_test(camera_id: int, detector: FaceDetector, ensemble: AntiSpoofOfficialEnsemble):
    print("=" * 80)
    print(f"[INFO] Đang kết nối tới Camera #{camera_id}...")
    cap = cv2.VideoCapture(camera_id)

    if not cap.isOpened():
        print(f"[ERROR] Không thể mở Camera #{camera_id}! Vui lòng kiểm tra cáp webcam hoặc quyền truy cập.")
        return

    print("[INFO] Camera đã sẵn sàng! Đang phát hiện...")
    print("Phím điều khiển:")
    print("  - [ESC] / [Q] : Thoát chương trình")
    print("  - [S]         : Chụp ảnh lưu vào backup/tests/output/official_ensemble/")
    print("  - [C]         : Bật / Tắt hiển thị Thumbnail crops")
    print("  - [SPACE]     : Tạm dừng / Tiếp tục webcam")
    print("=" * 80)

    device_str = "CUDA GPU" if torch.cuda.is_available() else "CPU"
    show_thumb = True
    paused = False
    prev_time = time.time()
    fps = 0.0
    last_frame = None

    while True:
        if not paused:
            ret, frame = cap.read()
            if not ret:
                print("[WARN] Không đọc được frame từ webcam.")
                break
            last_frame = frame.copy()
        else:
            frame = last_frame.copy() if last_frame is not None else np.zeros((480, 640, 3), dtype=np.uint8)

        # Tính FPS
        curr_time = time.time()
        dt = curr_time - prev_time
        prev_time = curr_time
        if dt > 0:
            fps = 0.9 * fps + 0.1 * (1.0 / dt)

        # Phát hiện khuôn mặt
        faces = detector.detect(frame)

        # Xử lý từng khuôn mặt qua Ensemble
        for idx, face in enumerate(faces, 1):
            bbox = face["bbox"]
            res = ensemble.predict_face(frame, bbox)
            draw_ensemble_face_ui(
                frame, idx, bbox, res,
                show_crop_thumb=show_thumb,
                thumb_x_offset=(idx - 1) * 110
            )

        # Thanh Header tổng quan hệ thống
        draw_card_overlay(frame, 0, 0, frame.shape[1], 42, bg_color=(15, 15, 15), alpha=0.85)
        cv2.line(frame, (0, 42), (frame.shape[1], 42), (0, 255, 127), 2)

        font = cv2.FONT_HERSHEY_DUPLEX
        cv2.putText(frame, "DUAL-MODEL OFFICIAL ENSEMBLE (Silent-Face-Anti-Spoofing)", (15, 26), font, 0.52, (255, 255, 255), 1, cv2.LINE_AA)
        status_dev = f"FPS: {fps:4.1f} | {device_str} | Faces: {len(faces)}"
        (tw, _), _ = cv2.getTextSize(status_dev, font, 0.48, 1)
        cv2.putText(frame, status_dev, (frame.shape[1] - tw - 15, 26), font, 0.48, (0, 255, 127), 1, cv2.LINE_AA)

        if paused:
            draw_card_overlay(frame, frame.shape[1]//2 - 120, frame.shape[0]//2 - 30, 240, 60, bg_color=(0, 0, 100), alpha=0.8)
            cv2.putText(frame, "PAUSED (SPACE to resume)", (frame.shape[1]//2 - 105, frame.shape[0]//2 + 6), font, 0.5, (255, 255, 255), 1, cv2.LINE_AA)

        cv2.imshow("Dual-Model Anti-Spoofing Ensemble [Silent-Face Official]", frame)
        key = cv2.waitKey(1) & 0xFF

        if key in (27, ord('q'), ord('Q')):
            break
        elif key in (ord('s'), ord('S')):
            save_name = f"webcam_ensemble_{int(time.time())}.jpg"
            save_path = os.path.join(OUTPUT_DIR, save_name)
            cv2.imwrite(save_path, frame)
            print(f"[INFO] 📸 Đã lưu ảnh chụp màn hình tại: {save_path}")
        elif key in (ord('c'), ord('C')):
            show_thumb = not show_thumb
        elif key == 32:  # SPACE
            paused = not paused

    cap.release()
    cv2.destroyAllWindows()
    print("[INFO] Đã kết thúc chương trình kiểm thử webcam.")


# =============================================================================
# 7. MAIN ENTRY POINT
# =============================================================================
def main():
    parser = argparse.ArgumentParser(
        description="Kiểm thử Anti-Spoofing Dual-Model Ensemble chính thức (MiniFASNetV2 2.7x + MiniFASNetV1SE 4.0x)"
    )
    parser.add_argument("--camera", type=int, default=0, help="Chỉ số Camera Webcam (mặc định 0)")
    parser.add_argument("--image", type=str, default=None, help="Đường dẫn tới 1 file ảnh cụ thể để test")
    parser.add_argument("--dir", type=str, default=None, help="Đường dẫn tới thư mục chứa ảnh (ví dụ: data_raw hoặc sample)")
    parser.add_argument("--model1", type=str, default=None, help="Đường dẫn tới file model 2.7_80x80_MiniFASNetV2.pth")
    parser.add_argument("--model2", type=str, default=None, help="Đường dẫn tới file model 4_0_0_80x80_MiniFASNetV1SE.pth")
    parser.add_argument("--thresh", type=float, default=0.5, help="Ngưỡng phân loại REAL (mặc định 0.5)")
    args = parser.parse_args()

    print("=" * 80)
    print(" KHỞI TẠO BỘ KIỂM THỬ ANTI-SPOOFING DUAL-MODEL ENSEMBLE CHÍNH THỨC ")
    print("=" * 80)

    # 1. Nạp Face Detection
    print("[INFO] Đang tải mô hình Face Detection (Face_Detection.pt)...")
    detector = FaceDetector()
    print("[INFO] Tải Face Detection thành công!")

    # 2. Nạp Ensemble Anti-Spoofing (2 Models)
    ensemble = AntiSpoofOfficialEnsemble(
        model1_path=args.model1,
        model2_path=args.model2,
        real_threshold=args.thresh
    )

    # 3. Điều hướng kiểm thử
    if args.image:
        if not os.path.exists(args.image):
            print(f"[ERROR] File ảnh không tồn tại: {args.image}")
            return
        run_image_test([args.image], detector, ensemble, save_output=True)

    elif args.dir:
        if not os.path.isdir(args.dir):
            print(f"[ERROR] Thư mục không tồn tại: {args.dir}")
            return
        extensions = ("*.jpg", "*.jpeg", "*.png", "*.bmp", "*.webp")
        img_paths = []
        for ext in extensions:
            img_paths.extend(glob.glob(os.path.join(args.dir, ext)))
            img_paths.extend(glob.glob(os.path.join(args.dir, "**", ext), recursive=True))
        img_paths = sorted(list(set(img_paths)))

        if not img_paths:
            print(f"[WARN] Không tìm thấy ảnh nào trong thư mục: {args.dir}")
            return
        print(f"[INFO] Tìm thấy {len(img_paths)} ảnh trong thư mục: {args.dir}")
        run_image_test(img_paths, detector, ensemble, save_output=True)

    else:
        # Chạy mẫu trên các ảnh sample có sẵn nếu có
        sample_dir = os.path.join(PROJECT_DIR, "Silent-Face-Anti-Spoofing-master", "images", "sample")
        sample_imgs = glob.glob(os.path.join(sample_dir, "*.jpg"))
        if sample_imgs:
            print(f"[INFO] Tự động kiểm thử mẫu trên {len(sample_imgs)} ảnh mẫu trong {sample_dir}...")
            run_image_test(sample_imgs, detector, ensemble, save_output=True)

        # Mặc định mở Webcam
        run_webcam_test(args.camera, detector, ensemble)


if __name__ == "__main__":
    main()
