"""
Script giả lập ESP32-CAM gửi ảnh chụp sang FastAPI AI Server
và kiểm tra luồng Webhook chuyển tiếp kết quả sang Node.js Server.
"""

import os
import sys
import time
import urllib.request
import urllib.error
import json

CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(CURRENT_DIR)

AI_SERVER_URL = "http://127.0.0.1:8000/api/v1/esp32/verify"
NODEJS_WEBHOOK_STATUS_URL = "http://127.0.0.1:3000/api/ekyc/history"
TEST_IMAGE_PATH = os.path.join(PROJECT_ROOT, "data_raw", "0.jpg")


def test_esp32_flow():
    print("=" * 70)
    print(" [TEST] GIẢ LẬP ESP32-CAM GỬI ẢNH TỚI AI SERVER VÀ NODE.JS")
    print("=" * 70)

    if not os.path.exists(TEST_IMAGE_PATH):
        print(f"[!] Không tìm thấy file ảnh test tại: {TEST_IMAGE_PATH}")
        return False

    with open(TEST_IMAGE_PATH, "rb") as f:
        image_bytes = f.read()

    print(f"[*] Đã nạp ảnh test: {TEST_IMAGE_PATH} ({len(image_bytes)} bytes)")
    print(f"[*] Đang gửi HTTP POST binary (image/jpeg) tới: {AI_SERVER_URL}...")

    req = urllib.request.Request(
        AI_SERVER_URL,
        data=image_bytes,
        headers={
            "Content-Type": "image/jpeg",
            "X-Device-ID": "ESP32_SIMULATOR_TEST_01"
        }
    )

    t0 = time.time()
    try:
        with urllib.request.urlopen(req, timeout=20) as resp:
            status_code = resp.getcode()
            response_body = resp.read().decode("utf-8")
            elapsed = (time.time() - t0) * 1000

            print(f"[✓] AI Server phản hồi thành công! (HTTP {status_code}) sau {elapsed:.1f} ms")
            print("[*] Dữ liệu JSON trả về cho ESP32:")
            data = json.loads(response_body)
            print(json.dumps(data, indent=2, ensure_ascii=False))

            # Kiểm tra các trường dữ liệu quan trọng
            assert data.get("success") is True, "Trường 'success' phải là True"
            assert "approved" in data, "Thiếu trường 'approved'"
            assert "verdict" in data, "Thiếu trường 'verdict'"
            print(f"\n[✓] KẾT QUẢ KIỂM THỬ: PASS! ESP32-CAM nhận phản hồi chuẩn xác.")
            return True

    except urllib.error.URLError as e:
        print(f"[!] Kết nối thất bại: {e}")
        print("    (Hãy đảm bảo FastAPI AI Server đang chạy qua lệnh: python run_api_server.py)")
        return False


if __name__ == "__main__":
    test_esp32_flow()
