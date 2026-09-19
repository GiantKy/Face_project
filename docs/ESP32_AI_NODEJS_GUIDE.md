# HƯỚNG DẪN TÍCH HỢP ESP32-CAM ➔ FASTAPI AI SERVER ➔ NODE.JS BACKEND

Tài liệu này hướng dẫn chi tiết cách vận hành toàn bộ chuỗi hệ thống: **ESP32-CAM** chụp ảnh thực tế, gửi sang **FastAPI AI Server** để nhận diện khuôn mặt và phát hiện giả mạo (Anti-Spoofing), sau đó AI Server tự động đẩy kết quả sang **Node.js Backend**.

---

## 1. Sơ đồ Kiến trúc & Luồng Dữ liệu

```
┌─────────────────┐       HTTP POST (Raw JPEG)       ┌────────────────────────┐
│    ESP32-CAM    │ ───────────────────────────────► │   FastAPI AI Server    │
│ (AI-Thinker/S3) │ ◄─────────────────────────────── │      (Port 8000)       │
└─────────────────┘      JSON phản hồi gọn nhẹ       └───────────┬────────────┘
         ▲            (approved: true/false, ...)                 │
         │                                                        │ Webhook POST
         │ Đèn LED / Relay Mở Cửa                                 │ (JSON + Crop Base64)
         │                                                        ▼
         └──────────────────────────────────────────  ┌────────────────────────┐
                                                      │     Node.js Server     │
                                                      │      (Port 3000)       │
                                                      └────────────────────────┘
```

### Ưu điểm vượt trội:
1. **Không gây tràn RAM ESP32**: ESP32 gửi trực tiếp dữ liệu nhị phân `image/jpeg` từ Frame Buffer (`fb->buf`), không tốn RAM mã hóa Base64. Phản hồi trả về ESP32 là JSON siêu nhẹ (chỉ khoảng 100 bytes).
2. **AI Server xử lý trọn gói**: Nhận diện khuôn mặt (YOLO), kiểm tra tư thế 3D Pose, chống giả mạo Ensemble (YOLO + RF-DETR).
3. **Webhook tự động tới Node.js**: Sau khi phân tích xong, AI Server tự động đẩy kết quả chi tiết kèm ảnh crop khuôn mặt về Node.js để lưu Database hoặc hiển thị Realtime Web.

---

## 2. Các Bước Khởi Động Hệ Thống

### BƯỚC 1: Khởi động Server Node.js Receiver
Mở Terminal 1 và chạy máy chủ Node.js (dùng Node.js tiêu chuẩn, không cần cài đặt thêm thư viện):

```bash
node server_module/nodejs_server_receiver.js
```
- Máy chủ sẽ lắng nghe Webhook tại: `http://127.0.0.1:3000/api/ekyc/result`
- Web Dashboard giám sát trực tiếp: `http://127.0.0.1:3000/`
- Thư mục tự động lưu ảnh khuôn mặt: `server_module/captured_faces/`

---

### BƯỚC 2: Khởi động FastAPI AI Server
Mở Terminal 2 và chạy máy chủ AI:

```bash
python run_api_server.py
```
*(Nếu dùng môi trường ảo: `venv\Scripts\activate` trước khi chạy).*

- Địa chỉ API: `http://127.0.0.1:8000`
- Tài liệu tương tác Swagger UI: `http://127.0.0.1:8000/docs`
- Endpoint chuyên dụng cho ESP32: `POST http://127.0.0.1:8000/api/v1/esp32/verify`

> [!TIP]
> Để đổi địa chỉ URL của Node.js mà không cần khởi động lại AI Server, bạn có thể gọi API:
> `POST http://127.0.0.1:8000/api/v1/webhook/config` với Body `{ "webhook_url": "http://<IP_NODEJS>:3000/api/ekyc/result" }`

---

### BƯỚC 3: Cấu hình và Nạp Code cho ESP32-CAM

1. Mở phần mềm **Arduino IDE**.
2. Mở file: `CameraWebServer/CameraToAIServer/CameraToAIServer.ino`.
3. Kiểm tra file `board_config.h`:
   - Nếu dùng **ESP32-CAM AI-Thinker** (board màu đen/xanh phổ biến có đèn Flash to): giữ nguyên `#define CAMERA_MODEL_AI_THINKER`.
   - Nếu dùng **ESP32-S3-Eye**: mở comment `#define CAMERA_MODEL_ESP32S3_EYE` và đóng comment các dòng khác.
4. Chỉnh sửa thông số mạng trong file `CameraToAIServer.ino`:
   ```cpp
   // Tên và mật khẩu WiFi (Cùng mạng WiFi với máy tính chạy AI Server)
   const char *ssid = "TÊN_WIFI_CỦA_BẠN";
   const char *password = "MẬT_KHẨU_WIFI";

   // IP máy tính chạy FastAPI AI Server trong mạng LAN
   const char *ai_server_ip = "192.168.1.50"; // Xem bằng lệnh `ipconfig` trên Windows
   const int ai_server_port = 8000;
   ```
5. Chọn thiết lập nạp trong Arduino IDE:
   - **Board**: `AI Thinker ESP32-CAM` (hoặc `ESP32S3 Dev Module`).
   - **CPU Frequency**: `240MHz`.
   - **Flash Frequency**: `80MHz`.
   - **Flash Mode**: `QIO`.
   - **Partition Scheme**: `Huge APP (3MB No OTA/1MB SPIFFS)` hoặc `Minimal SPIFFS (1.9MB APP with OTA/190KB SPIFFS)`.
   - **PSRAM**: `Enabled`.
6. Nhấn nút **Upload** nạp code vào ESP32-CAM.
7. Sau khi nạp xong, mở **Serial Monitor** (tốc độ `115200 baud`) và nhấn nút Reset trên ESP32 để xem địa chỉ IP được cấp (ví dụ: `http://192.168.1.85`).

---

## 3. Cách Thử Nghiệm

### Cách 1: Thử nghiệm qua Giao diện Web tích hợp trên ESP32
1. Mở trình duyệt trên máy tính hoặc điện thoại cùng mạng WiFi, gõ địa chỉ IP của ESP32:
   `http://<IP_ESP32_CAM>` (ví dụ `http://192.168.1.85`).
2. Giao diện xuất hiện hình ảnh camera.
3. Bấm nút: **"🚀 CHỤP & GỬI NHẬN DIỆN AI"**.
4. Quan sát:
   - Trên web ESP32: Hiển thị kết quả duyệt hay từ chối và độ tin cậy.
   - Trên Terminal AI Server: Hiển thị log xử lý khuôn mặt.
   - Trên Terminal Node.js: Hiển thị bảng Dashboard thông tin sự kiện và lưu ảnh khuôn mặt.
   - Mở `http://localhost:3000/` để xem ảnh vừa nhận trên Node.js Web Dashboard!

### Cách 2: Tự động chụp định kỳ
Trong file `CameraToAIServer.ino`, đặt:
```cpp
#define ENABLE_AUTO_CAPTURE true
#define AUTO_CAPTURE_INTERVAL_SEC 5 // Cứ mỗi 5 giây tự chụp và gửi AI 1 lần
```

### Cách 3: Dùng Nút bấm hoặc Cảm biến PIR
- Nối nút bấm vào chân `GPIO 13` và `GND`.
- Trong code đặt `#define ENABLE_HARDWARE_BUTTON true`. Khi có người bấm nút, ESP32 sẽ chụp và gửi AI ngay lập tức.

---

## 4. Cấu trúc Gói Tin Webhook Gửi Sang Node.js

Node.js sẽ nhận được payload JSON tại `POST /api/ekyc/result`:
```json
{
  "event": "EKYC_ESP32_VERIFICATION",
  "device_id": "ESP32_CAM_GATE_01",
  "timestamp": "2026-09-18 15:30:00",
  "approved": true,
  "verdict": "VERIFIED_REAL",
  "is_real": true,
  "confidence": 0.985,
  "reasons": ["Dual-model agreement: REAL"],
  "face_detection": {
    "detected": true,
    "face_count": 1,
    "box": [120, 80, 240, 250]
  },
  "pose_3d": {
    "euler_angles": { "yaw": 2.1, "pitch": -1.5, "roll": 0.8 },
    "pose_valid": true
  },
  "crop_face_base64": "data:image/jpeg;base64,...",
  "annotated_image_base64": "data:image/jpeg;base64,...",
  "processing_time_ms": 145.8
}
```

---

## 5. Tùy biến Mở Rộng cho Node.js

Trong file `server_module/nodejs_server_receiver.js`, bạn có thể dễ dàng chèn thêm logic nghiệp vụ:

```javascript
// Gợi ý chèn trong hàm processEKYCResult(data):
if (isApproved) {
  // 1. Lưu vào Database (MongoDB / PostgreSQL)
  // await db.collection('checkins').insertOne({ ... });

  // 2. Phát thông báo Realtime lên Web Frontend qua Socket.IO
  // io.emit('door_unlocked', { user: device_id, time: timestamp });

  // 3. Gửi lệnh mở chốt cửa thông minh qua MQTT / Webhook
} else {
  // Cảnh báo người lạ hoặc giả mạo (Fake/Spoof) qua Bot Telegram / Zalo
  // sendTelegramAlert(`Cảnh báo giả mạo tại ${device_id}!`);
}
```
