# -*- coding: utf-8 -*-
"""
=============================================================================
Test Module: Official Silent-Face MiniFASNetV2 Anti-Spoofing (3-Class Model)
=============================================================================
Kiểm thử mô hình Anti-Spoofing MiniFASNetV2 chính thức (Weights 1.85MB)
Hỗ trợ phân biệt:
  - REAL      : Khuôn mặt người thật (Live)
  - SPOOF 2D  : Ảnh in, giấy, poster...
  - SPOOF 3D  : Màn hình điện thoại, máy tính bảng, video phát lại, mặt nạ...

Tính năng kiểm thử:
  1. Real-time Webcam (Mặc định):
     - Vẽ bounding box nổi bật (Xanh lá = REAL, Đỏ = FAKE/SPOOF).
     - Hiển thị chi tiết xác suất 3 classes: [Real %, Spoof 2D %, Spoof 3D %].
     - Hiển thị ảnh thu nhỏ vùng mặt crop (80x80) đưa vào mô hình.
     - Hiển thị FPS thực tế & thiết bị phần cứng (CUDA GPU / CPU).
  2. Kiểm thử ảnh đơn lẻ (--image <path>) hoặc thư mục ảnh (--dir <path>).

Phím tắt trong chế độ Webcam:
  - ESC / Q   : Thoát chương trình
  - S         : Lưu ảnh chụp màn hình vào output/official_minifasnet/
  - C         : Bật / Tắt ảnh thumbnail crop 80x80
  - M         : Đổi hệ số scale crop (2.7x chuẩn -> 4.0x mở rộng -> 1.2x gọn)
  - SPACE     : Tạm dừng / Tiếp tục
=============================================================================
"""

import sys
import os
import time
import argparse
from pathlib import Path
import cv2
import numpy as np
import torch

# Thiết lập đường dẫn import tới thư mục gốc Face-Project
CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_DIR = os.path.dirname(CURRENT_DIR)

if PROJECT_DIR not in sys.path:
    sys.path.insert(0, PROJECT_DIR)

from src.face_detection import FaceDetector
from src.anti_spoof import AntiSpoofOfficial

OUTPUT_DIR = os.path.join(CURRENT_DIR, "output", "official_minifasnet")
os.makedirs(OUTPUT_DIR, exist_ok=True)


# =============================================================================
# UI / DRAWING HELPERS
# =============================================================================
def draw_card_overlay(img, x, y, w, h, bg_color=(20, 20, 20), alpha=0.7):
    """Vẽ nền bán trong suốt làm card thông tin"""
    sub_img = img[y:y+h, x:x+w]
    if sub_img.size == 0:
        return
    rect = np.full_like(sub_img, bg_color, dtype=np.uint8)
    res = cv2.addWeighted(sub_img, 1.0 - alpha, rect, alpha, 1.0)
    img[y:y+h, x:x+w] = res


def draw_official_face_ui(
    frame: np.ndarray,
    face_idx: int,
    bbox: list,
    res: dict,
    show_crop_thumb: bool = True,
    thumb_x_offset: int = 0
):
    """
    Vẽ bounding box, nhãn dự đoán, thanh đo 3 classes và thumbnail crop
    """
    x1, y1, x2, y2 = bbox
    is_real = res["is_real"]
    label = res["label"]
    conf = res["confidence"]
    real_score = res["real_score"]
    c_scores = res.get("class_scores", {})
    p_2d = c_scores.get("spoof_2d", 0.0)
    p_3d = c_scores.get("spoof_3d", 0.0)
    face_crop = res.get("face_crop", None)

    # Màu sắc chủ đạo: Xanh ngọc (REAL) / Đỏ cam (FAKE)
    main_color = (46, 204, 113) if is_real else (60, 76, 231)  # BGR
    border_color = (0, 255, 127) if is_real else (0, 0, 255)

    # 1. Bounding box & Góc bo tròn (Corner brackets)
    cv2.rectangle(frame, (x1, y1), (x2, y2), main_color, 2)
    line_len = min(22, (x2 - x1) // 4, (y2 - y1) // 4)
    thick = 3
    # Top-Left
    cv2.line(frame, (x1, y1), (x1 + line_len, y1), border_color, thick)
    cv2.line(frame, (x1, y1), (x1, y1 + line_len), border_color, thick)
    # Top-Right
    cv2.line(frame, (x2, y1), (x2 - line_len, y1), border_color, thick)
    cv2.line(frame, (x2, y1), (x2, y1 + line_len), border_color, thick)
    # Bottom-Left
    cv2.line(frame, (x1, y2), (x1 + line_len, y2), border_color, thick)
    cv2.line(frame, (x1, y2), (x1, y2 - line_len), border_color, thick)
    # Bottom-Right
    cv2.line(frame, (x2, y2), (x2 - line_len, y2), border_color, thick)
    cv2.line(frame, (x2, y2), (x2, y2 - line_len), border_color, thick)

    # 2. Tag Header label
    tag_text = f"FACE #{face_idx}: {label} ({conf * 100:.1f}%)"
    font = cv2.FONT_HERSHEY_SIMPLEX
    font_scale = 0.6
    font_thick = 2
    (tw, th), _ = cv2.getTextSize(tag_text, font, font_scale, font_thick)

    tag_y1 = max(0, y1 - th - 12)
    tag_y2 = max(th + 12, y1)
    tag_x2 = min(frame.shape[1], x1 + tw + 16)
    cv2.rectangle(frame, (x1, tag_y1), (tag_x2, tag_y2), main_color, -1)
    cv2.putText(
        frame, tag_text, (x1 + 8, tag_y2 - 6),
        font, font_scale, (255, 255, 255), font_thick, cv2.LINE_AA
    )

    # 3. Thanh đo 3-Class Breakdown dưới box
    card_w = max(200, x2 - x1)
    card_h = 32
    card_y = min(frame.shape[0] - card_h - 4, y2 + 6)
    if card_y + card_h < frame.shape[0]:
        draw_card_overlay(frame, x1, card_y, card_w, card_h, bg_color=(20, 20, 20), alpha=0.8)
        cv2.rectangle(frame, (x1, card_y), (x1 + card_w, card_y + card_h), (80, 80, 80), 1)

        # Thanh màu phân bố [2D Spoof | REAL | 3D Spoof]
        bar_x = x1 + 6
        bar_y = card_y + 6
        bar_w = card_w - 12
        bar_h = 8
        cv2.rectangle(frame, (bar_x, bar_y), (bar_x + bar_w, bar_y + bar_h), (50, 50, 50), -1)

        w_real = int(bar_w * real_score)
        w_2d = int(bar_w * p_2d)
        w_3d = bar_w - w_real - w_2d

        cur_x = bar_x
        if w_2d > 0:
            cv2.rectangle(frame, (cur_x, bar_y), (cur_x + w_2d, bar_y + bar_h), (255, 140, 0), -1)  # Cam (2D)
            cur_x += w_2d
        if w_real > 0:
            cv2.rectangle(frame, (cur_x, bar_y), (cur_x + w_real, bar_y + bar_h), (46, 204, 113), -1)  # Xanh (Real)
            cur_x += w_real
        if w_3d > 0:
            cv2.rectangle(frame, (cur_x, bar_y), (cur_x + w_3d, bar_y + bar_h), (60, 76, 231), -1)  # Đỏ (3D)

        # Chú thích chi tiết tỷ lệ %
        info_str = f"Real:{real_score*100:4.1f}% | 2D:{p_2d*100:4.1f}% | 3D:{p_3d*100:4.1f}%"
        cv2.putText(
            frame, info_str, (x1 + 6, card_y + 24),
            font, 0.38, (220, 220, 220), 1, cv2.LINE_AA
        )

    # 4. Thumbnail Crop khuôn mặt (góc dưới màn hình)
    if show_crop_thumb and face_crop is not None and face_crop.size > 0:
        thumb_size = 90
        pad = 15
        thumb_y = frame.shape[0] - thumb_size - pad
        thumb_x = pad + thumb_x_offset

        if thumb_x + thumb_size < frame.shape[1] and thumb_y >= 0:
            try:
                resized_thumb = cv2.resize(face_crop, (thumb_size, thumb_size))
                cv2.rectangle(
                    frame,
                    (thumb_x - 3, thumb_y - 3),
                    (thumb_x + thumb_size + 3, thumb_y + thumb_size + 3),
                    main_color, -1
                )
                frame[thumb_y:thumb_y+thumb_size, thumb_x:thumb_x+thumb_size] = resized_thumb
                cv2.putText(
                    frame, f"Face #{face_idx} (80x80)", (thumb_x, thumb_y - 6),
                    font, 0.4, (220, 220, 220), 1, cv2.LINE_AA
                )
            except Exception:
                pass


def draw_hud_header(frame: np.ndarray, fps: float, device_name: str, num_faces: int, scale_factor: float, model_name: str):
    """Vẽ thanh HUD thông số trạng thái ở góc trên màn hình"""
    h_overlay = 48
    w_overlay = 520
    draw_card_overlay(frame, 10, 10, w_overlay, h_overlay, bg_color=(15, 15, 20), alpha=0.8)
    cv2.rectangle(frame, (10, 10), (10 + w_overlay, 10 + h_overlay), (80, 80, 80), 1)

    font = cv2.FONT_HERSHEY_SIMPLEX
    cv2.putText(
        frame, "Official MiniFASNetV2 Anti-Spoof (Silent-Face)", (20, 30),
        font, 0.55, (0, 220, 255), 2, cv2.LINE_AA
    )
    info_text = f"FPS: {fps:4.1f} | Dev: {device_name} | Faces: {num_faces} | Scale: {scale_factor:.1f}x | {model_name[:20]}"
    cv2.putText(
        frame, info_text, (20, 48),
        font, 0.38, (200, 200, 200), 1, cv2.LINE_AA
    )


# ==============================================================
# BATCH / IMAGE TESTING
# ==============================================================
def run_image_test(image_paths: list, detector: FaceDetector, anti_spoof: AntiSpoofOfficial, save_output: bool = True):
    print("=" * 85)
    print(" KIỂM THỬ ANTI-SPOOF MINIFASNETV2 CHÍNH THỨC TRÊN DANH SÁCH ẢNH")
    print("=" * 85)

    total_images = len(image_paths)
    total_faces = 0
    real_count = 0
    fake_count = 0

    print(f"\n{'STT':<4} | {'TÊN ẢNH':<22} | {'MẶT':<4} | {'KẾT QUẢ':<8} | {'REAL %':<8} | {'2D SPOOF %':<11} | {'3D SPOOF %':<11} | {'TIME(ms)':<8}")
    print("-" * 100)

    for img_idx, img_p in enumerate(image_paths, 1):
        filename = os.path.basename(img_p)
        frame = cv2.imread(img_p)
        if frame is None:
            print(f"{img_idx:<4} | {filename:<22} | [LỖI: Không thể đọc file ảnh]")
            continue

        faces = detector.detect(frame)
        if not faces:
            print(f"{img_idx:<4} | {filename:<22} | 0    | NO FACE  | -        | -           | -           | -")
            continue

        annotated = frame.copy()
        for f_idx, face in enumerate(faces, 1):
            total_faces += 1
            bbox = face["bbox"]

            t0 = time.perf_counter()
            res = anti_spoof.predict_face(frame, bbox)
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

            print(f"{img_idx:<4} | {filename:<22} | #{f_idx:<3} | {lbl:<8} | {r_pct:5.1f}%  | {p2_pct:5.1f}%      | {p3_pct:5.1f}%      | {infer_ms:6.1f}")

            draw_official_face_ui(
                annotated, f_idx, bbox, res,
                show_crop_thumb=True, thumb_x_offset=(f_idx - 1) * 110
            )

        if save_output:
            out_file = os.path.join(OUTPUT_DIR, f"result_{filename}")
            cv2.imwrite(out_file, annotated)

    print("-" * 100)
    print(f"[TỔNG KẾT] Đã kiểm thử: {total_images} ảnh | Tổng số khuôn mặt: {total_faces}")
    print(f"           - REAL (Thật): {real_count}")
    print(f"           - FAKE (Giả) : {fake_count}")
    if save_output:
        print(f"[INFO] Ảnh kết quả đã được lưu tại: {OUTPUT_DIR}")
    print("=" * 85)


# ==============================================================
# REAL-TIME WEBCAM LOOP
# ==============================================================
def run_webcam_test(camera_id: int, detector: FaceDetector, anti_spoof: AntiSpoofOfficial):
    print("=" * 80)
    print(f"[INFO] Đang mở Camera #{camera_id}...")
    cap = cv2.VideoCapture(camera_id)

    if not cap.isOpened():
        print(f"[ERROR] Không thể mở Camera #{camera_id}!")
        return

    print("[INFO] Camera đã sẵn sàng!")
    print("Phím điều khiển:")
    print("  - [ESC] / [Q] : Thoát chương trình")
    print("  - [S]         : Chụp ảnh lưu vào output/official_minifasnet/")
    print("  - [C]         : Bật / Tắt ảnh Crop 80x80")
    print("  - [M]         : Đổi Scale crop (2.7x -> 4.0x -> 1.2x)")
    print("  - [SPACE]     : Tạm dừng / Tiếp tục frame")
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
                break

            frame = cv2.flip(frame, 1)

            curr_time = time.time()
            fps = 1.0 / max(1e-5, (curr_time - prev_time))
            prev_time = curr_time

            # Phát hiện khuôn mặt
            faces = detector.detect(frame)

            # Dự đoán Anti-Spoof
            for idx, face in enumerate(faces, 1):
                bbox = face["bbox"]
                res = anti_spoof.predict_face(frame, bbox)

                draw_official_face_ui(
                    frame,
                    face_idx=idx,
                    bbox=bbox,
                    res=res,
                    show_crop_thumb=show_thumb,
                    thumb_x_offset=(idx - 1) * 110
                )

            # HUD
            draw_hud_header(
                frame, fps, device_str, len(faces),
                anti_spoof.scale_factor, anti_spoof.model_name
            )
            last_frame = frame.copy()

        display_frame = last_frame if paused and last_frame is not None else frame

        if paused:
            cv2.putText(
                display_frame, "[PAUSED - Press SPACE to Resume]",
                (display_frame.shape[1] // 2 - 160, 80),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 165, 255), 2, cv2.LINE_AA
            )

        cv2.imshow("Official MiniFASNetV2 Anti-Spoofing", display_frame)

        key = cv2.waitKey(1) & 0xFF
        if key in (27, ord('q'), ord('Q')):
            break
        elif key in (ord('s'), ord('S')):
            timestamp = time.strftime("%Y%m%d_%H%M%S")
            sp = os.path.join(OUTPUT_DIR, f"capture_{timestamp}.jpg")
            cv2.imwrite(sp, display_frame)
            print(f"[INFO] Đã lưu ảnh: {sp}")
        elif key in (ord('c'), ord('C')):
            show_thumb = not show_thumb
            print(f"[INFO] Crop thumbnail: {'BẬT' if show_thumb else 'TẮT'}")
        elif key in (ord('m'), ord('M')):
            if abs(anti_spoof.scale_factor - 2.7) < 0.1:
                anti_spoof.scale_factor = 4.0
            elif abs(anti_spoof.scale_factor - 4.0) < 0.1:
                anti_spoof.scale_factor = 1.2
            else:
                anti_spoof.scale_factor = 2.7
            print(f"[INFO] Scale Factor chuyển sang: {anti_spoof.scale_factor:.1f}x")
        elif key == 32:
            paused = not paused

    cap.release()
    cv2.destroyAllWindows()


# ==============================================================
# MAIN ENTRYPOINT
# ==============================================================
def main():
    parser = argparse.ArgumentParser(description="Kiểm thử Official MiniFASNetV2 Anti-Spoofing")
    parser.add_argument("--camera", type=int, default=0, help="Camera ID (mặc định 0)")
    parser.add_argument("--image", type=str, default=None, help="Đường dẫn file ảnh")
    parser.add_argument("--dir", type=str, default=None, help="Đường dẫn thư mục ảnh")
    parser.add_argument("--model", type=str, default=None, help="Đường dẫn tới file weights .pth")
    parser.add_argument("--scale", type=float, default=2.7, help="Tỷ lệ crop khuôn mặt (mặc định 2.7x)")
    parser.add_argument("--thresh", type=float, default=0.5, help="Ngưỡng phân loại REAL (mặc định 0.5)")

    args = parser.parse_args()

    print("\n" + "=" * 80)
    print(" KHỞI TẠO BỘ KIỂM THỬ OFFICIAL MINIFASNETV2 ANTI-SPOOFING ")
    print("=" * 80)

    # 1. Face Detector
    print("[INFO] Đang nạp YOLO Face Detector...")
    detector = FaceDetector()
    print("[INFO] ✅ Nạp Face Detector thành công!")

    # 2. Official Anti-Spoof
    anti_spoof = AntiSpoofOfficial(
        model_path=args.model,
        scale_factor=args.scale,
        real_threshold=args.thresh
    )
    print("[INFO] ✅ Nạp Official MiniFASNetV2 thành công!")

    # 3. Chọn chế độ kiểm thử
    if args.image:
        if not os.path.exists(args.image):
            print(f"[ERROR] Không tìm thấy ảnh: {args.image}")
            sys.exit(1)
        run_image_test([args.image], detector, anti_spoof)
    elif args.dir:
        if not os.path.exists(args.dir):
            print(f"[ERROR] Không tìm thấy thư mục: {args.dir}")
            sys.exit(1)
        valid_exts = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
        files = [os.path.join(args.dir, f) for f in os.listdir(args.dir) if os.path.splitext(f.lower())[1] in valid_exts]
        if not files:
            print("[WARN] Không có ảnh hợp lệ trong thư mục!")
            sys.exit(1)
        run_image_test(files, detector, anti_spoof)
    else:
        run_webcam_test(args.camera, detector, anti_spoof)


if __name__ == "__main__":
    main()
