"""
=============================================================================
Test Module: MiniFASNet Anti-Spoofing (Face Liveness / Spoof Detection)
=============================================================================
Sử dụng mô hình MiniFASNetV2 (Silent-Face-Anti-Spoofing lightweight architecture)
để phân loại khuôn mặt:
  - REAL  (Khuôn mặt thật / Liveness)
  - FAKE  (Ảnh in 2D, màn hình điện thoại, máy tính bảng, video phát lại...)

Hỗ trợ:
  1. Real-time Webcam (Mặc định):
     - Vẽ bounding box theo màu (Xanh lá = REAL, Đỏ = FAKE).
     - Hiển thị thanh đo xác suất (Real % vs Fake %).
     - Hiển thị ảnh thu nhỏ Face Crop (80x80) đưa vào model.
     - Hiển thị FPS thực tế & thiết bị phần cứng (CUDA GPU / CPU).
  2. Test qua ảnh đơn hoặc thư mục ảnh (--image <path> hoặc --dir <path>):
     - Duyệt qua từng ảnh và xuất báo cáo kết quả chi tiết.

Phím tắt trong chế độ Webcam:
  - ESC hoặc Q : Thoát
  - S          : Lưu ảnh chụp màn hình vào output/minifasnet/
  - C          : Bật / Tắt hiển thị ảnh Face Crop thu nhỏ
  - M          : Đổi chế độ crop scale (1.0x chuẩn vs 1.2x mở rộng)
  - SPACE      : Tạm dừng / Tiếp tục
=============================================================================
"""

import sys
import os
import time
import argparse
import glob
from pathlib import Path
import cv2
import numpy as np
import torch

# Thiết lập đường dẫn import tới thư mục backup/
CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
BACKUP_DIR = os.path.dirname(CURRENT_DIR)
BASE_DIR = os.path.dirname(BACKUP_DIR)

if BACKUP_DIR not in sys.path:
    sys.path.insert(0, BACKUP_DIR)

from src.face_detection import FaceDetector
from src.anti_spoof import AntiSpoofMiniFASNet

OUTPUT_DIR = os.path.join(CURRENT_DIR, "output", "minifasnet")
os.makedirs(OUTPUT_DIR, exist_ok=True)


# =============================================================================
# UI / DRAWING HELPERS
# =============================================================================
def draw_card_overlay(img, x, y, w, h, bg_color=(20, 20, 20), alpha=0.65):
    """Vẽ nền bán trong suốt làm card thông tin"""
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

    # 1. Bounding box góc bo tròn hoặc box kép
    cv2.rectangle(frame, (x1, y1), (x2, y2), main_color, 2)
    # Vẽ 4 góc viền nổi bật (Corner brackets)
    line_len = min(20, (x2 - x1) // 4, (y2 - y1) // 4)
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
    font_scale = 0.6
    font_thick = 2
    (tw, th), _ = cv2.getTextSize(tag_text, font, font_scale, font_thick)

    # Tag background
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
    bar_w = min(160, x2 - x1)
    bar_h = 10
    bar_y = min(frame.shape[0] - bar_h - 4, y2 + 6)
    if bar_y + bar_h < frame.shape[0]:
        # Background bar
        cv2.rectangle(frame, (x1, bar_y), (x1 + bar_w, bar_y + bar_h), (50, 50, 50), -1)
        real_fill = int(bar_w * real_score)
        if real_fill > 0:
            cv2.rectangle(frame, (x1, bar_y), (x1 + real_fill, bar_y + bar_h), (46, 204, 113), -1)
        if bar_w - real_fill > 0:
            cv2.rectangle(frame, (x1 + real_fill, bar_y), (x1 + bar_w, bar_y + bar_h), (60, 76, 231), -1)

    # 4. Hiển thị thumbnail crop 80x80 góc dưới màn hình
    if show_crop_thumb and face_crop is not None and face_crop.size > 0:
        thumb_size = 90
        pad = 15
        thumb_y = frame.shape[0] - thumb_size - pad
        thumb_x = pad + thumb_x_offset

        if thumb_x + thumb_size < frame.shape[1] and thumb_y >= 0:
            try:
                resized_thumb = cv2.resize(face_crop, (thumb_size, thumb_size))
                # Viền màu theo trạng thái
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
                    f"Crop 80x80 #{face_idx}",
                    (thumb_x, thumb_y - 6),
                    font,
                    0.4,
                    (220, 220, 220),
                    1,
                    cv2.LINE_AA
                )
            except Exception:
                pass


def draw_hud_header(frame: np.ndarray, fps: float, device_name: str, num_faces: int, scale_factor: float):
    """Vẽ thanh thông tin trạng thái HUD ở góc trên màn hình"""
    h_overlay = 45
    w_overlay = 460
    draw_card_overlay(frame, 10, 10, w_overlay, h_overlay, bg_color=(15, 15, 20), alpha=0.75)
    cv2.rectangle(frame, (10, 10), (10 + w_overlay, 10 + h_overlay), (70, 70, 70), 1)

    font = cv2.FONT_HERSHEY_SIMPLEX
    # Tiêu đề
    cv2.putText(
        frame,
        "MiniFASNetV2 Anti-Spoof",
        (20, 32),
        font,
        0.55,
        (0, 220, 255),
        2,
        cv2.LINE_AA
    )
    # Thông số FPS, Faces, Scale
    info_text = f"FPS: {fps:4.1f} | Dev: {device_name} | Faces: {num_faces} | Scale: {scale_factor:.1f}x"
    cv2.putText(
        frame,
        info_text,
        (20, 48),
        font,
        0.4,
        (200, 200, 200),
        1,
        cv2.LINE_AA
    )


# =============================================================================
# TEST ON IMAGES / DIRECTORY
# =============================================================================
def run_image_test(
    image_paths: list,
    detector: FaceDetector,
    anti_spoof: AntiSpoofMiniFASNet,
    save_output: bool = True
):
    """Chạy kiểm thử Anti-Spoof MiniFASNet trên danh sách ảnh"""
    print("=" * 80)
    print(" BẮT ĐẦU KIỂM THỬ ANTI-SPOOF MINIFASNET TRÊN DANH SÁCH ẢNH")
    print("=" * 80)

    total_images = len(image_paths)
    total_faces = 0
    real_count = 0
    fake_count = 0

    print(f"\n{'STT':<4} | {'TÊN ẢNH':<20} | {'MẶT':<5} | {'KẾT QUẢ':<8} | {'CONFIDENCE':<12} | {'REAL SCORE':<12} | {'FAKE SCORE':<12} | {'TIME(ms)':<8}")
    print("-" * 95)

    for img_idx, img_p in enumerate(image_paths, 1):
        filename = os.path.basename(img_p)
        frame = cv2.imread(img_p)

        if frame is None:
            print(f"{img_idx:<4} | {filename:<20} | [LỖI: Không thể đọc file ảnh]")
            continue

        start_t = time.perf_counter()
        faces = detector.detect(frame)
        infer_ms = (time.perf_counter() - start_t) * 1000.0

        if not faces:
            print(f"{img_idx:<4} | {filename:<20} | 0     | NO FACE  | -            | -            | -            | {infer_ms:6.1f}")
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

            print(f"{img_idx:<4} | {filename:<20} | #{f_idx:<4} | {lbl:<8} | {conf:5.1f}%      | {r_sc:5.1f}%      | {f_sc:5.1f}%      | {face_ms:6.1f}")

            # Vẽ lên ảnh
            draw_face_anti_spoof_ui(
                annotated_frame,
                f_idx,
                bbox,
                res,
                show_crop_thumb=True,
                thumb_x_offset=(f_idx - 1) * 110
            )

        if save_output:
            out_file = os.path.join(OUTPUT_DIR, f"result_{filename}")
            cv2.imwrite(out_file, annotated_frame)

    print("-" * 95)
    print(f"[TỔNG KẾT] Đã kiểm thử: {total_images} ảnh | Tổng số khuôn mặt: {total_faces}")
    print(f"           - REAL (Thật): {real_count}")
    print(f"           - FAKE (Giả) : {fake_count}")
    if save_output:
        print(f"[INFO] Ảnh kết quả đã được lưu tại: {OUTPUT_DIR}")
    print("=" * 80)


# =============================================================================
# REAL-TIME WEBCAM LOOP
# =============================================================================
def run_webcam_test(
    camera_id: int,
    detector: FaceDetector,
    anti_spoof: AntiSpoofMiniFASNet
):
    """Mở Webcam và thực hiện kiểm thử Anti-Spoofing thời gian thực"""
    print("=" * 80)
    print(f"[INFO] Đang mở Camera #{camera_id}...")
    cap = cv2.VideoCapture(camera_id)

    if not cap.isOpened():
        print(f"[ERROR] Không thể mở Camera #{camera_id}!")
        print("Vui lòng kiểm tra lại thiết bị Webcam.")
        return

    print("[INFO] Camera đã sẵn sàng!")
    print("Phím điều khiển:")
    print("  - [ESC] / [Q] : Thoát chương trình")
    print("  - [S]         : Chụp ảnh lưu vào output/minifasnet/")
    print("  - [C]         : Bật / Tắt xem ảnh Face Crop 80x80")
    print("  - [M]         : Chuyển đổi tỷ lệ crop scale (1.0x vs 1.2x)")
    print("  - [SPACE]     : Tạm dừng / Tiếp tục frame")
    print("=" * 80)

    device_str = "CUDA GPU" if torch.cuda.is_available() else "CPU"
    show_thumb = True
    paused = False
    prev_time = time.time()
    fps = 0.0

    last_annotated_frame = None

    while True:
        if not paused:
            ret, frame = cap.read()
            if not ret:
                print("[WARN] Mất tín hiệu từ Camera.")
                break

            # Lật gương để trải nghiệm tự nhiên
            frame = cv2.flip(frame, 1)

            # Tính FPS
            curr_time = time.time()
            fps = 1.0 / max(1e-5, (curr_time - prev_time))
            prev_time = curr_time

            # 1. Phát hiện khuôn mặt
            faces = detector.detect(frame)

            # 2. Dự đoán Anti-Spoof cho từng khuôn mặt
            for idx, face in enumerate(faces, 1):
                bbox = face["bbox"]
                res = anti_spoof.predict_face(frame, bbox)

                # Vẽ giao diện cho từng khuôn mặt
                draw_face_anti_spoof_ui(
                    frame,
                    face_idx=idx,
                    bbox=bbox,
                    res=res,
                    show_crop_thumb=show_thumb,
                    thumb_x_offset=(idx - 1) * 110
                )

            # 3. Vẽ thanh HUD trạng thái
            draw_hud_header(
                frame,
                fps=fps,
                device_name=device_str,
                num_faces=len(faces),
                scale_factor=anti_spoof.scale_factor
            )

            last_annotated_frame = frame.copy()

        display_frame = last_annotated_frame if paused and last_annotated_frame is not None else frame

        if paused:
            cv2.putText(
                display_frame,
                "[PAUSED - Press SPACE to Resume]",
                (display_frame.shape[1] // 2 - 160, 80),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.7,
                (0, 165, 255),
                2,
                cv2.LINE_AA
            )

        cv2.imshow("MiniFASNet Anti-Spoofing Test (E-KYC)", display_frame)

        key = cv2.waitKey(1) & 0xFF
        if key in (27, ord('q'), ord('Q')):
            print("[INFO] Đang đóng Camera...")
            break
        elif key in (ord('s'), ord('S')):
            timestamp = time.strftime("%Y%m%d_%H%M%S")
            save_path = os.path.join(OUTPUT_DIR, f"capture_{timestamp}.jpg")
            cv2.imwrite(save_path, display_frame)
            print(f"[INFO] Đã lưu ảnh chụp: {save_path}")
        elif key in (ord('c'), ord('C')):
            show_thumb = not show_thumb
            print(f"[INFO] Crop thumbnail: {'BẬT' if show_thumb else 'TẮT'}")
        elif key in (ord('m'), ord('M')):
            if anti_spoof.scale_factor == 1.0:
                anti_spoof.scale_factor = 1.2
            elif anti_spoof.scale_factor == 1.2:
                anti_spoof.scale_factor = 1.5
            else:
                anti_spoof.scale_factor = 1.0
            print(f"[INFO] Crop Scale Factor chuyển sang: {anti_spoof.scale_factor:.1f}x")
        elif key == 32:  # SPACE
            paused = not paused
            print(f"[INFO] Trạng thái: {'TẠM DỪNG' if paused else 'TIẾP TỤC'}")

    cap.release()
    cv2.destroyAllWindows()
    print("[INFO] Đã hoàn tất.")


# =============================================================================
# MAIN ENTRYPOINT
# =============================================================================
def main():
    parser = argparse.ArgumentParser(
        description="Kiểm thử mô hình Anti-Spoofing MiniFASNetV2"
    )
    parser.add_argument(
        "--camera", type=int, default=0,
        help="Chỉ số Camera Webcam (mặc định 0)"
    )
    parser.add_argument(
        "--image", type=str, default=None,
        help="Đường dẫn tới 1 file ảnh cụ thể để test"
    )
    parser.add_argument(
        "--dir", type=str, default=None,
        help="Đường dẫn tới thư mục chứa ảnh (ví dụ: data_raw)"
    )
    parser.add_argument(
        "--model", type=str, default=None,
        help="Đường dẫn tuỳ chỉnh tới file model Anti_Spoof_minifasnetv2.pth"
    )
    parser.add_argument(
        "--scale", type=float, default=1.0,
        help="Tỷ lệ phóng to vùng crop khuôn mặt (mặc định 1.0)"
    )
    parser.add_argument(
        "--thresh", type=float, default=0.5,
        help="Ngưỡng phân loại REAL (mặc định 0.5)"
    )
    parser.add_argument(
        "--check-model", action="store_true",
        help="Chỉ kiểm tra chất lượng model rồi thoát (không chạy inference)"
    )

    args = parser.parse_args()

    print("\n" + "=" * 80)
    print(" KHỞI TẠO BỘ KIỂM THỬ ANTI-SPOOF MINIFASNETV2 ")
    print("=" * 80)

    # 1. Tải Face Detection
    print("[INFO] Đang tải mô hình Face Detection (Face_Detection.pt)...")
    detector = FaceDetector()
    print("[INFO] Tải Face Detection thành công!")

    # 2. Tải MiniFASNet Anti-Spoof
    model_file = args.model
    if model_file is None:
        from src.anti_spoof.minifasnet import find_default_minifasnet_model
        model_file = find_default_minifasnet_model()

    print(f"[INFO] Đang tải mô hình MiniFASNet từ: {model_file}")
    anti_spoof = AntiSpoofMiniFASNet(
        model_path=model_file,
        scale_factor=args.scale,
        real_threshold=args.thresh
    )

    # 3. Hiển thị trạng thái model
    qi = anti_spoof.quality_info
    print(f"[INFO] Tải mô hình MiniFASNetV2 thành công!")
    print(f"[INFO] Model quality: {'✓ SẴN SÀNG' if qi['is_usable'] else '✗ CHƯA SẴN SÀNG (cần train thêm)'}")
    print(f"[INFO] Batches trained: {qi['num_batches_trained']}")

    if args.check_model:
        print("\n" + "=" * 80)
        print(" KẾT QUẢ CHẨN ĐOÁN MODEL ")
        print("=" * 80)
        print(f"  Model path     : {model_file}")
        print(f"  Batches trained: {qi['num_batches_trained']}")
        print(f"  Classifier bias: {qi['classifier_bias']}")
        print(f"  Usable         : {qi['is_usable']}")
        if qi["warnings"]:
            print(f"  Cảnh báo:")
            for w in qi["warnings"]:
                print(f"    ⚠ {w}")
        else:
            print(f"  Không có cảnh báo. Model đã sẵn sàng sử dụng.")
        print("=" * 80)
        return

    # 4. Chọn chế độ kiểm thử
    if args.image:
        if not os.path.exists(args.image):
            print(f"[ERROR] Không tìm thấy file ảnh: {args.image}")
            sys.exit(1)
        run_image_test([args.image], detector, anti_spoof)

    elif args.dir:
        if not os.path.exists(args.dir):
            print(f"[ERROR] Không tìm thấy thư mục: {args.dir}")
            sys.exit(1)
        valid_exts = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
        image_files = [
            os.path.join(args.dir, f) for f in os.listdir(args.dir)
            if os.path.splitext(f.lower())[1] in valid_exts
        ]
        if not image_files:
            print(f"[WARN] Không tìm thấy ảnh nào trong thư mục: {args.dir}")
            sys.exit(1)
        run_image_test(image_files, detector, anti_spoof)

    else:
        # Nếu không truyền tham số ảnh/thư mục, mặc định mở Webcam
        run_webcam_test(args.camera, detector, anti_spoof)


if __name__ == "__main__":
    main()

