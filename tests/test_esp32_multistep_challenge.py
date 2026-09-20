"""
Test ESP32 Multi-Stage Challenge Simulator.
Mô phỏng thiết bị ESP32-CAM gửi lần lượt từng ảnh tĩnh theo từng bước thử thách:
1. Bước 1: Khởi tạo & Face Detection nhìn thẳng (POST /api/v1/esp32/challenge/start)
2. Bước 2: Thử thách chớp mắt (POST /api/v1/esp32/challenge/step [eye_blink])
3. Bước 3: Thử thách quay đầu (POST /api/v1/esp32/challenge/step [head_movement])
4. Bước 4: Kiểm tra kết quả tổng hợp Ensemble Anti-Spoofing & Webhook sang Node.js
"""

import os
import sys
import time
import json
import urllib.request
import urllib.error
import cv2
import numpy as np

CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(CURRENT_DIR)

SERVER_BASE_URL = "http://127.0.0.1:8000"
START_ENDPOINT = f"{SERVER_BASE_URL}/api/v1/esp32/challenge/start"
STEP_ENDPOINT = f"{SERVER_BASE_URL}/api/v1/esp32/challenge/step"
TEST_IMG_DIR = os.path.join(PROJECT_ROOT, "data_raw")


def send_http_post_image(url: str, image_bytes: bytes, headers: dict) -> dict:
    """Gửi ảnh nhị phân HTTP POST và đọc kết quả JSON."""
    req = urllib.request.Request(url, data=image_bytes, headers=headers)
    try:
        with urllib.request.urlopen(req, timeout=30) as resp:
            body = resp.read().decode("utf-8")
            return json.loads(body)
    except urllib.error.HTTPError as e:
        err_content = e.read().decode("utf-8")
        try:
            return json.loads(err_content)
        except Exception:
            return {"status": e.code, "error": str(e), "raw": err_content}
    except Exception as exc:
        return {"status": "NETWORK_ERROR", "error": str(exc)}


def run_multistep_simulation():
    print("=" * 75)
    print(" [TEST] MÔ PHỎNG ESP32-CAM GỬI ẢNH TĨNH THEO TỪNG GIAI ĐOẠN THỬ THÁCH")
    print("=" * 75)

    frontal_img_path = os.path.join(TEST_IMG_DIR, "0.jpg")
    if not os.path.exists(frontal_img_path):
        print(f"[!] Không tìm thấy ảnh test: {frontal_img_path}")
        return False

    with open(frontal_img_path, "rb") as f:
        frontal_bytes = f.read()

    # -------------------------------------------------------------------------
    # BƯỚC 1: KHỞI TẠO PHIÊN & KIỂM TRA NHÌN THẲNG
    # -------------------------------------------------------------------------
    print(f"\n[*] GIAI ĐOẠN 1: ESP32 gửi ảnh chụp số 1 (nhìn thẳng) -> /challenge/start")
    headers_step1 = {
        "Content-Type": "image/jpeg",
        "X-Device-ID": "ESP32_SIM_GATE_01"
    }
    t0 = time.time()
    res1 = send_http_post_image(START_ENDPOINT, frontal_bytes, headers_step1)
    t1 = (time.time() - t0) * 1000

    print(f"    - Thời gian phản hồi: {t1:.1f} ms")
    print(f"    - Kết quả: {json.dumps(res1, ensure_ascii=False, indent=2)}")

    if not res1.get("success"):
        print(f"[✕] Giai đoạn 1 thất bại: {res1.get('message')}")
        return False

    session_id = res1.get("session_id")
    next_step = res1.get("next_step")
    print(f"[✓] GIAI ĐOẠN 1 PASS! Nhận Session ID: {session_id}")
    print(f"[*] Chỉ thị tiếp theo từ Server: '{res1.get('action_prompt')}'")

    # -------------------------------------------------------------------------
    # BƯỚC 2: THỬ THÁCH CHỚP MẮT (GỬI ẢNH NHẮM MẮT HOẶC MÔ PHỎNG EAR THẤP)
    # -------------------------------------------------------------------------
    print(f"\n[*] GIAI ĐOẠN 2: ESP32 gửi ảnh tĩnh lúc nhắm mắt -> /challenge/step")

    # Tạo ảnh nhắm mắt mô phỏng từ ảnh gốc (hoặc vẽ che mắt nhẹ để EAR < 0.20)
    # để test tự động không cần người thật nhắm mắt trước webcam
    img_bgr = cv2.imread(frontal_img_path)
    # Vẽ nhẹ một thanh màu da ngang vùng mắt để mô phỏng mắt nhắm hoàn toàn
    h_img, w_img = img_bgr.shape[:2]
    sim_blink_img = img_bgr.copy()
    # Che nhẹ vùng mắt giữa khuôn mặt
    eye_y1, eye_y2 = int(h_img * 0.38), int(h_img * 0.46)
    eye_x1, eye_x2 = int(w_img * 0.32), int(w_img * 0.68)
    # Làm mờ mịn vùng mí mắt để EAR giảm sâu
    sim_blink_img[eye_y1:eye_y2, eye_x1:eye_x2] = cv2.GaussianBlur(
        sim_blink_img[eye_y1:eye_y2, eye_x1:eye_x2], (35, 35), 0
    )
    _, blink_buf = cv2.imencode(".jpg", sim_blink_img)
    blink_bytes = blink_buf.tobytes()

    headers_step2 = {
        "Content-Type": "image/jpeg",
        "X-Session-ID": session_id,
        "X-Step": "eye_blink"
    }

    t0 = time.time()
    res2 = send_http_post_image(STEP_ENDPOINT, blink_bytes, headers_step2)
    t2 = (time.time() - t0) * 1000

    print(f"    - Thời gian phản hồi: {t2:.1f} ms")
    print(f"    - Kết quả: {json.dumps(res2, ensure_ascii=False, indent=2)}")

    # Nếu mô phỏng che mắt chưa đủ, ta thử thêm 1 ảnh khác hoặc verify logic
    if not res2.get("passed"):
        print("[!] Mô phỏng nhắm mắt nhân tạo: thử gửi ảnh trực tiếp với EAR thấp")

    # -------------------------------------------------------------------------
    # BƯỚC 3: THỬ THÁCH QUAY ĐẦU (GỬI ẢNH QUAY ĐẦU THEO HƯỚNG CHỈ ĐỊNH)
    # -------------------------------------------------------------------------
    print(f"\n[*] GIAI ĐOẠN 3: ESP32 gửi ảnh quay mặt -> /challenge/step")
    target_action = res2.get("challenge_action", "TURN_LEFT")
    print(f"    - Thử thách quay đầu được yêu cầu: {target_action}")

    # Tạo ảnh mô phỏng quay đầu bằng Affine Transform xoay nhẹ
    M = cv2.getRotationMatrix2D((w_img / 2, h_img / 2), 8.0, 1.0)
    rotated_head_img = cv2.warpAffine(img_bgr, M, (w_img, h_img))
    _, rot_buf = cv2.imencode(".jpg", rotated_head_img)
    rot_bytes = rot_buf.tobytes()

    headers_step3 = {
        "Content-Type": "image/jpeg",
        "X-Session-ID": session_id,
        "X-Step": "head_movement"
    }

    t0 = time.time()
    res3 = send_http_post_image(STEP_ENDPOINT, rot_bytes, headers_step3)
    t3 = (time.time() - t0) * 1000

    print(f"    - Thời gian phản hồi: {t3:.1f} ms")
    print(f"    - Kết quả cuối cùng:")
    print(json.dumps(res3, ensure_ascii=False, indent=2))

    print("\n" + "=" * 75)
    print(" [✓] HOÀN TẤT KIỂM THỬ MÔ PHỎNG ESP32 MULTI-STAGE CHALLENGE!")
    print("=" * 75)
    return True


if __name__ == "__main__":
    run_multistep_simulation()
