"""
=============================================================================
Test Module: MobileNetV2 Anti-Spoofing (Face Liveness / Spoof Detection)
=============================================================================
Sử dụng mô hình MobileNetV2 (Hugging Face Transformers / Safetensors):
  - LIVE / REAL (0) : Khuôn mặt thật trực tiếp trước camera
  - SPOOF / FAKE (1): Giả mạo (ảnh in giấy, màn hình điện thoại, máy tính, video phát lại...)

Hỗ trợ:
  1. Real-time Webcam (Mặc định):
     - Bounding box góc viền hiện đại (Xanh lá = REAL, Đỏ = FAKE).
     - Hiển thị thanh đo xác suất Real % vs Fake %.
     - Hiển thị thumbnail Face Crop (224x224) đưa vào model.
     - Hiển thị FPS thực tế & thiết bị phần cứng (CUDA GPU / CPU).
  2. Test ảnh đơn (--image <path>) hoặc thư mục ảnh (--dir <path>):
     - Duyệt qua từng ảnh và xuất bảng báo cáo chi tiết.

Phím tắt trong chế độ Webcam:
  - ESC hoặc Q : Thoát
  - S          : Lưu ảnh chụp màn hình vào output/mobilenetv2/
  - C          : Bật / Tắt thumbnail Face Crop (224x224)
  - M          : Đổi chế độ crop scale (1.0x -> 1.2x -> 1.5x -> 2.0x)
  - + hoặc =   : Tăng ngưỡng Real threshold (+0.05)
  - - hoặc _   : Giảm ngưỡng Real threshold (-0.05)
  - SPACE      : Tạm dừng / Tiếp tục
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
import cv2
import numpy as np
import torch

# Thiết lập đường dẫn import
CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
BASE_DIR = os.path.dirname(CURRENT_DIR)  # Face-Project/

if BASE_DIR not in sys.path:
    sys.path.insert(0, BASE_DIR)

from src.face_detection import FaceDetector
from src.anti_spoof import AntiSpoofMobileNetV2, find_default_mobilenetv2_model

OUTPUT_DIR = os.path.join(CURRENT_DIR, "output", "mobilenetv2")
os.makedirs(OUTPUT_DIR, exist_ok=True)


# =============================================================================
# UI / DRAWING HELPERS
# =============================================================================
def draw_card_overlay(img: np.ndarray, x: int, y: int, w: int, h: int, bg_color=(20, 20, 25), alpha=0.7):
    """Vẽ nền bán trong suốt làm card thông tin HUD"""
    sub_img = img[y:y+h, x:x+w]
    if sub_img.size == 0:
        return
    rect = np.full_like(sub_img, bg_color, dtype=np.uint8)
    res = cv2.addWeighted(sub_img, 1.0 - alpha, rect, alpha, 1.0)
    img[y:y+h, x:x+w] = res


def draw_face_anti_spoof_ui(
    frame: np.ndarray,
    face_idx: int,
    bbox: list,
    res: dict,
    show_crop_thumb: bool = True,
    thumb_x_offset: int = 0
):
    """
    Vẽ bounding box, nhãn dự đoán, thanh xác suất và thumbnail cho từng khuôn mặt
    """
    x1, y1, x2, y2 = bbox
    is_real = res["is_real"]
    label = res["label"]
    conf = res["confidence"]
    real_score = res["real_score"]
    fake_score = res["fake_score"]
    face_crop = res.get("face_crop", None)

    # Màu sắc chủ đạo: Xanh lá sáng (REAL) / Đỏ tươi (FAKE)
    main_color = (46, 204, 113) if is_real else (60, 76, 231)  # BGR
    border_color = (0, 255, 127) if is_real else (0, 0, 255)

    # 1. Bounding box & 4 góc viền nổi bật (Corner brackets)
    cv2.rectangle(frame, (x1, y1), (x2, y2), main_color, 2)
    line_len = max(15, min(25, (x2 - x1) // 4, (y2 - y1) // 4))
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

    # 2. Tag / Header label
    tag_text = f"FACE #{face_idx}: {label} ({conf * 100:.1f}%)"
    font = cv2.FONT_HERSHEY_SIMPLEX
    font_scale = 0.58
    font_thick = 2
    (tw, th), _ = cv2.getTextSize(tag_text, font, font_scale, font_thick)

    tag_y1 = max(0, y1 - th - 12)
    tag_y2 = max(th + 12, y1)
    tag_x2 = min(frame.shape[1], x1 + tw + 16)
    cv2.rectangle(frame, (x1, tag_y1), (tag_x2, tag_y2), main_color, -1)
    cv2.putText(
        frame,
        tag_text,
        (x1 + 8, tag_y2 - 6),
        font,
        font_scale,
        (255, 255, 255),
        font_thick,
        cv2.LINE_AA
    )

    # 3. Mini bar trực quan Real vs Fake ngay dưới bbox
    bar_w = min(170, max(120, x2 - x1))
    bar_h = 10
    bar_y = min(frame.shape[0] - bar_h - 4, y2 + 6)
    if bar_y + bar_h < frame.shape[0]:
        # Background bar
        cv2.rectangle(frame, (x1, bar_y), (x1 + bar_w, bar_y + bar_h), (45, 45, 50), -1)
        real_fill = int(bar_w * real_score)
        if real_fill > 0:
            cv2.rectangle(frame, (x1, bar_y), (x1 + real_fill, bar_y + bar_h), (46, 204, 113), -1)
        if bar_w - real_fill > 0:
            cv2.rectangle(frame, (x1 + real_fill, bar_y), (x1 + bar_w, bar_y + bar_h), (60, 76, 231), -1)

        # Chữ tỉ lệ %
        score_str = f"R:{real_score*100:.0f}% F:{fake_score*100:.0f}%"
        cv2.putText(
            frame,
            score_str,
            (x1, bar_y + bar_h + 12),
            font,
            0.38,
            (230, 230, 230),
            1,
            cv2.LINE_AA
        )

    # 4. Hiển thị thumbnail crop 224x224 góc dưới màn hình
    if show_crop_thumb and face_crop is not None and face_crop.size > 0:
        thumb_size = 100
        pad = 15
        thumb_y = frame.shape[0] - thumb_size - pad - 25  # Chừa chỗ cho thanh footer
        thumb_x = pad + thumb_x_offset

        if thumb_x + thumb_size < frame.shape[1] and thumb_y >= 0:
            try:
                resized_thumb = cv2.resize(face_crop, (thumb_size, thumb_size))
                # Viền màu theo kết quả
                cv2.rectangle(
                    frame,
                    (thumb_x - 3, thumb_y - 3),
                    (thumb_x + thumb_size + 3, thumb_y + thumb_size + 3),
                    main_color,
                    -1
                )
                frame[thumb_y:thumb_y+thumb_size, thumb_x:thumb_x+thumb_size] = resized_thumb
                cv2.putText(
                    frame,
                    f"Crop 224 #{face_idx}",
                    (thumb_x, thumb_y - 6),
                    font,
                    0.38,
                    (230, 230, 230),
                    1,
                    cv2.LINE_AA
                )
            except Exception:
                pass


def draw_hud_header(
    frame: np.ndarray,
    fps: float,
    device_name: str,
    num_faces: int,
    scale_factor: float,
    threshold: float,
    model_name: str = "MobileNetV2"
):
    """Vẽ thanh thông tin trạng thái HUD ở góc trên màn hình"""
    h_overlay = 48
    w_overlay = 550
    draw_card_overlay(frame, 10, 10, w_overlay, h_overlay, bg_color=(15, 17, 24), alpha=0.8)
    cv2.rectangle(frame, (10, 10), (10 + w_overlay, 10 + h_overlay), (70, 75, 90), 1)

    font = cv2.FONT_HERSHEY_SIMPLEX
    # Tiêu đề kèm tên file model weights
    cv2.putText(
        frame,
        f"MobileNetV2 Anti-Spoof [{model_name}]",
        (20, 32),
        font,
        0.52,
        (0, 220, 255),
        2,
        cv2.LINE_AA
    )
    # Thông số FPS, Thiết bị, Faces, Scale, Threshold
    info_text = f"FPS: {fps:4.1f} | Dev: {device_name} | Faces: {num_faces} | Scale: {scale_factor:.1f}x | Thresh: {threshold:.2f}"
    cv2.putText(
        frame,
        info_text,
        (20, 48),
        font,
        0.38,
        (200, 205, 215),
        1,
        cv2.LINE_AA
    )



def draw_hotkeys_footer(frame: np.ndarray):
    """Vẽ thanh chỉ dẫn phím tắt ở góc dưới màn hình"""
    h, w = frame.shape[:2]
    footer_h = 24
    draw_card_overlay(frame, 0, h - footer_h, w, footer_h, bg_color=(10, 10, 15), alpha=0.85)
    font = cv2.FONT_HERSHEY_SIMPLEX
    text = "[ESC/Q] Thoat | [S] Chup anh | [C] Thumbnail | [M] Doi Scale | [+/-] Nguong | [SPACE] Pause"
    cv2.putText(
        frame,
        text,
        (15, h - 7),
        font,
        0.38,
        (180, 185, 195),
        1,
        cv2.LINE_AA
    )


# =============================================================================
# TEST ON IMAGES / DIRECTORY
# =============================================================================
def run_image_test(
    image_paths: list,
    detector: FaceDetector,
    anti_spoof: AntiSpoofMobileNetV2,
    save_output: bool = True
):
    """Chạy kiểm thử MobileNetV2 Anti-Spoof trên danh sách ảnh"""
    print("=" * 85)
    print(" BẮT ĐẦU KIỂM THỬ MOBILENETV2 ANTI-SPOOF TRÊN DANH SÁCH ẢNH")
    print("=" * 85)

    total_images = len(image_paths)
    total_faces = 0
    real_count = 0
    fake_count = 0

    print(f"\n{'STT':<4} | {'TÊN ẢNH':<22} | {'MẶT':<5} | {'KẾT QUẢ':<8} | {'CONFIDENCE':<12} | {'REAL SCORE':<12} | {'FAKE SCORE':<12} | {'TIME(ms)':<8}")
    print("-" * 102)

    for img_idx, img_p in enumerate(image_paths, 1):
        filename = os.path.basename(img_p)
        frame = cv2.imread(img_p)

        if frame is None:
            print(f"{img_idx:<4} | {filename:<22} | [LỖI: Không thể đọc file ảnh]")
            continue

        start_t = time.perf_counter()
        faces = detector.detect(frame)
        infer_ms = (time.perf_counter() - start_t) * 1000.0

        if not faces:
            print(f"{img_idx:<4} | {filename:<22} | 0     | NO FACE  | -            | -            | -            | {infer_ms:6.1f}")
            continue

        annotated_frame = frame.copy()

        for f_idx, face in enumerate(faces, 1):
            total_faces += 1
            bbox = face["bbox"]

            t0 = time.perf_counter()
            res = anti_spoof.predict_face(frame, bbox)
            face_ms = (time.perf_counter() - t0) * 1000.0

            if res["is_real"]:
                real_count += 1
            else:
                fake_count += 1

            lbl = res["label"]
            conf = res["confidence"] * 100.0
            r_sc = res["real_score"] * 100.0
            f_sc = res["fake_score"] * 100.0

            print(f"{img_idx:<4} | {filename:<22} | #{f_idx:<4} | {lbl:<8} | {conf:5.1f}%      | {r_sc:5.1f}%      | {f_sc:5.1f}%      | {face_ms:6.1f}")

            # Vẽ lên ảnh
            draw_face_anti_spoof_ui(
                annotated_frame,
                f_idx,
                bbox,
                res,
                show_crop_thumb=True,
                thumb_x_offset=(f_idx - 1) * 120
            )

        if save_output:
            out_file = os.path.join(OUTPUT_DIR, f"result_{filename}")
            cv2.imwrite(out_file, annotated_frame)

    print("-" * 102)
    print(f"[TỔNG KẾT] Đã kiểm thử: {total_images} ảnh | Tổng số khuôn mặt: {total_faces}")
    print(f"           - REAL / LIVE (Thật): {real_count}")
    print(f"           - FAKE / SPOOF (Giả): {fake_count}")
    if save_output:
        print(f"[INFO] Ảnh kết quả đã được lưu tại: {OUTPUT_DIR}")
    print("=" * 85)


# =============================================================================
# REAL-TIME WEBCAM LOOP
# =============================================================================
def run_webcam_test(
    camera_id: int,
    detector: FaceDetector,
    anti_spoof: AntiSpoofMobileNetV2
):
    """Mở Webcam và thực hiện kiểm thử Anti-Spoofing MobileNetV2 thời gian thực"""
    print("=" * 80)
    print(f"[INFO] Đang mở Camera #{camera_id}...")
    cap = cv2.VideoCapture(camera_id)

    if not cap.isOpened():
        print(f"[ERROR] Không thể mở Camera #{camera_id}!")
        print("Vui lòng kiểm tra lại thiết bị Webcam của bạn.")
        return

    print("[INFO] Camera đã sẵn sàng!")
    print("Phím điều khiển:")
    print("  - [ESC] / [Q] : Thoát chương trình")
    print("  - [S]         : Chụp ảnh lưu vào output/mobilenetv2/")
    print("  - [C]         : Bật / Tắt thumbnail Face Crop (224x224)")
    print("  - [M]         : Thay đổi crop scale (1.0x -> 1.2x -> 1.5x -> 2.0x)")
    print("  - [+/-]       : Tăng / Giảm Real threshold")
    print("  - [SPACE]     : Tạm dừng / Tiếp tục")
    print("=" * 80)

    # Đặt độ phân giải camera mượt mà (720p nếu hỗ trợ)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)

    show_thumb = True
    scales = [1.0, 1.2, 1.5, 2.0]
    scale_idx = 1  # Mặc định 1.2x
    paused = False

    device_name = anti_spoof.device.type.upper()
    if device_name == "CUDA":
        device_name = f"GPU ({torch.cuda.get_device_name(0)})"

    prev_time = time.perf_counter()
    fps = 0.0

    last_frame = None

    while True:
        if not paused:
            ret, frame = cap.read()
            if not ret:
                print("[WARNING] Mất tín hiệu từ camera!")
                break

            # Lật gương để tương tác tự nhiên
            frame = cv2.flip(frame, 1)

            curr_scale = scales[scale_idx]
            anti_spoof.scale_factor = curr_scale

            # Phát hiện khuôn mặt
            faces = detector.detect(frame)
            num_faces = len(faces)

            # Dự đoán Anti-Spoof MobileNetV2 cho từng mặt
            for f_idx, face in enumerate(faces, 1):
                bbox = face["bbox"]
                res = anti_spoof.predict_face(frame, bbox, scale=curr_scale)

                draw_face_anti_spoof_ui(
                    frame,
                    f_idx,
                    bbox,
                    res,
                    show_crop_thumb=show_thumb,
                    thumb_x_offset=(f_idx - 1) * 125
                )

            # Tính FPS mượt
            curr_time = time.perf_counter()
            dt = curr_time - prev_time
            prev_time = curr_time
            if dt > 0:
                current_fps = 1.0 / dt
                fps = 0.85 * fps + 0.15 * current_fps if fps > 0 else current_fps

            # Vẽ HUD Dashboard & Footer
            draw_hud_header(
                frame,
                fps,
                device_name,
                num_faces,
                curr_scale,
                anti_spoof.real_threshold,
                model_name=anti_spoof.model_name
            )
            draw_hotkeys_footer(frame)
            last_frame = frame.copy()

        else:
            if last_frame is not None:
                frame = last_frame.copy()
                # Hiển thị chữ PAUSED
                h, w = frame.shape[:2]
                cv2.putText(
                    frame,
                    "[PAUSED - PRESS SPACE TO RESUME]",
                    (w // 2 - 220, h // 2),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.75,
                    (0, 255, 255),
                    2,
                    cv2.LINE_AA
                )

        cv2.imshow("MobileNetV2 Face Anti-Spoof Test (eKYC)", frame)
        key = cv2.waitKey(1) & 0xFF

        if key in (27, ord('q'), ord('Q')):
            print("[INFO] Đã nhấn thoát.")
            break
        elif key in (ord('s'), ord('S')):
            timestamp = time.strftime("%Y%m%d_%H%M%S")
            snap_path = os.path.join(OUTPUT_DIR, f"snapshot_mobilenetv2_{timestamp}.jpg")
            cv2.imwrite(snap_path, frame)
            print(f"[INFO] Đã lưu snapshot tại: {snap_path}")
        elif key in (ord('c'), ord('C')):
            show_thumb = not show_thumb
            status = "BẬT" if show_thumb else "TẮT"
            print(f"[INFO] Thumbnail Face Crop: {status}")
        elif key in (ord('m'), ord('M')):
            scale_idx = (scale_idx + 1) % len(scales)
            print(f"[INFO] Đổi Crop Scale sang: {scales[scale_idx]:.1f}x")
        elif key in (ord('+'), ord('=')):
            anti_spoof.real_threshold = min(0.95, round(anti_spoof.real_threshold + 0.05, 2))
            print(f"[INFO] Ngưỡng Real Threshold tăng lên: {anti_spoof.real_threshold:.2f}")
        elif key in (ord('-'), ord('_')):
            anti_spoof.real_threshold = max(0.05, round(anti_spoof.real_threshold - 0.05, 2))
            print(f"[INFO] Ngưỡng Real Threshold giảm xuống: {anti_spoof.real_threshold:.2f}")
        elif key == 32:  # SPACE
            paused = not paused
            status = "TẠM DỪNG" if paused else "TIẾP TỤC"
            print(f"[INFO] Video: {status}")

    cap.release()
    cv2.destroyAllWindows()
    print("[INFO] Chương trình kết thúc.")


# =============================================================================
# ENTRY POINT & CLI PARSER
# =============================================================================
def main():
    parser = argparse.ArgumentParser(
        description="Kiểm thử mô hình MobileNetV2 Face Anti-Spoofing (Liveness Detection)"
    )
    parser.add_argument(
        "--camera",
        type=int,
        default=0,
        help="ID Webcam (mặc định: 0)"
    )
    parser.add_argument(
        "--image",
        type=str,
        default=None,
        help="Đường dẫn đến 1 ảnh để kiểm thử"
    )
    parser.add_argument(
        "--dir",
        type=str,
        default=None,
        help="Đường dẫn đến thư mục chứa ảnh kiểm thử"
    )
    parser.add_argument(
        "--model-path", "--model-dir",
        dest="model_path",
        type=str,
        default=None,
        help="Đường dẫn đến thư mục hoặc file .safetensors cụ thể của MobileNetV2 (mặc định tự động chọn file mới nhất)"
    )
    parser.add_argument(
        "--scale",
        type=float,
        default=1.2,
        help="Tỷ lệ mở rộng crop khuôn mặt (mặc định: 1.2)"
    )
    parser.add_argument(
        "--thresh",
        type=float,
        default=0.5,
        help="Ngưỡng xác suất để phân loại REAL (mặc định: 0.5)"
    )

    args = parser.parse_args()

    print("\n" + "=" * 80)
    print(" KHỞI ĐỘNG HỆ THỐNG KIỂM THỬ MOBILENETV2 FACE ANTI-SPOOFING")
    print("=" * 80)

    # 1. Khởi tạo Face Detector (YOLO)
    print("[1/2] Đang khởi tạo bộ nhận diện khuôn mặt FaceDetector...")
    detector = FaceDetector()
    print("[OK] FaceDetector đã sẵn sàng!")

    # 2. Khởi tạo AntiSpoofMobileNetV2
    print(f"\n[2/2] Đang tải mô hình MobileNetV2 Anti-Spoof...")
    try:
        anti_spoof = AntiSpoofMobileNetV2(
            model_path=args.model_path,
            scale_factor=args.scale,
            real_threshold=args.thresh
        )

    except Exception as e:
        print(f"[ERROR] Không thể khởi tạo MobileNetV2: {e}")
        sys.exit(1)

    # 3. Phân luồng chạy
    if args.image:
        if not os.path.exists(args.image):
            print(f"[ERROR] File không tồn tại: {args.image}")
            sys.exit(1)
        run_image_test([args.image], detector, anti_spoof)
    elif args.dir:
        if not os.path.isdir(args.dir):
            print(f"[ERROR] Thư mục không tồn tại: {args.dir}")
            sys.exit(1)
        valid_exts = ("*.jpg", "*.jpeg", "*.png", "*.bmp", "*.webp")
        img_paths = []
        for ext in valid_exts:
            img_paths.extend(glob.glob(os.path.join(args.dir, ext)))
            img_paths.extend(glob.glob(os.path.join(args.dir, ext.upper())))
        img_paths = sorted(list(set(img_paths)))
        if not img_paths:
            print(f"[WARNING] Không tìm thấy ảnh nào trong thư mục: {args.dir}")
            sys.exit(0)
        run_image_test(img_paths, detector, anti_spoof)
    else:
        run_webcam_test(args.camera, detector, anti_spoof)


if __name__ == "__main__":
    main()
