# 🛡️ Hệ Thống eKYC Sinh Trắc Học Khuôn Mặt & Chống Giả Mạo (AI Server + ESP32-CAM)

Hệ thống xác thực danh tính điện tử (**eKYC**) toàn diện tích hợp giữa phần cứng nhúng **ESP32-CAM / ESP32-S3** và máy chủ phân tích **AI Ensemble (YOLO_4 + RF-DETR Small + MediaPipe Face Mesh)** theo tiêu chuẩn bảo mật ngân hàng.

---

## 📁 Cấu Trúc Dự Án

```text
ekyc_release/
│
├── esp32_firmware/            # Mã nguồn PlatformIO cho vi điều khiển ESP32-CAM / ESP32-S3
│   ├── src/                   # Logic chương trình chính, WiFi, Camera, LED WS2812
│   ├── include/               # Header cấu hình (HardwareController.h, EKYCService.h, web_ui.h)
│   ├── platformio.ini         # Cấu hình nạp firmware PlatformIO
│   └── README.md              # Hướng dẫn chi tiết nạp code phần cứng
│
├── server_module/             # Máy chủ AI Backend & Giao diện Web Điều Khiển
│   ├── app.py                 # FastAPI REST API Hub (:8000)
│   ├── pipeline_server.py     # Lõi Pipeline thẩm định 8 tiêu chí eKYC & Fail-Fast Early Rejection
│   ├── nodejs_server_receiver.js # Node.js Relay Hub nhận stream MJPEG & điều phối LED RGB (:3000)
│   ├── components/            # Các module thị giác máy tính: Face Detection, Pose 3D, Liveness, Occlusion...
│   ├── models/                # Trọng số mô hình AI (.pt, .task, .onnx)
│   └── static/index.html      # Giao diện Web UI chuyên nghiệp (HUD, Telemetry, Oval Guide, Fail-Fast)
│
├── requirements.txt           # Danh sách thư viện Python cần thiết
├── start_all.bat              # Kịch bản khởi động 1-click cả Node.js và AI Server
├── .gitattributes             # Cấu hình Git LFS cho file trọng số mô hình lớn
└── .gitignore                 # Loại trừ file build, cache và dữ liệu tạm thời
```

---

## ⚡ Các Tính Năng Nổi Bật

1. **Active Liveness Multi-Stage Challenge:**
   - **Bước 1:** Canh chỉnh khuôn mặt trong khung Oval tỷ lệ vàng & kiểm tra kính râm / vật che mặt.
   - **Bước 2:** Thử thách chớp mắt tự nhiên (đo chỉ số EAR qua MediaPipe 468 Landmarks).
   - **Bước 3:** Thử thách quay đầu ngẫu nhiên (Turn Left / Right) tính toán qua thuật toán SolvePnP 3D Pose.
2. **Cơ Chế Fail-Fast Early Rejection:**
   - Nếu người dùng thất bại ở bất kỳ bước nào (hết giờ, tráo người, che mặt), hệ thống lập tức **từ chối ngay (REJECT)**, bật LED đỏ và hủy phiên, không lãng phí tài nguyên chạy full pipeline mô hình nặng.
3. **Dual-Model Ensemble Anti-Spoofing:**
   - Kết hợp giữa **YOLO_4** và **RF-DETR Small** với cơ chế Veto Spoofing nghiêm ngặt (chống in ấn 2D, màn hình điện thoại/laptop, mặt nạ).
4. **Đồng Bộ LED RGB WS2812 Theo Stage Real-Time:**
   - Màu sắc LED phản ánh chính xác từng giai đoạn thử thách và kết quả Approved (Xanh lá) / Rejected (Đỏ).

---

## 🚀 Hướng Dẫn Cài Đặt & Khởi Chạy

### 1. Yêu Cầu Môi Trường
- **Python**: 3.9 - 3.11
- **Node.js**: v16+
- **VS Code** với tiện ích mở rộng **PlatformIO IDE** (dành cho ESP32)

### 2. Cài Đặt Thư Viện Python
Mở Terminal trong thư mục `ekyc_release` và chạy:
```bash
pip install -r requirements.txt
```

### 3. Cài Đặt Thư Viện Node.js
```bash
cd server_module
npm install express cors multer
cd ..
```

### 4. Khởi Chạy Hệ Thống

**Cách 1: Khởi động 1-Click (Khuyên dùng)**
Nhấp đúp chuột vào tệp:
```bash
start_all.bat
```

**Cách 2: Khởi động thủ công**
- **Cửa sổ 1 (Node.js Relay Hub :3000):**
  ```bash
  cd server_module
  node nodejs_server_receiver.js
  ```
- **Cửa sổ 2 (AI Pipeline Server :8000):**
  ```bash
  uvicorn server_module.app:app --host 0.0.0.0 --port 8000 --reload
  ```

Sau khi chạy, mở trình duyệt truy cập:
👉 **Web Dashboard:** [http://localhost:3000/](http://localhost:3000/)  
👉 **API Swagger Docs:** [http://localhost:8000/docs](http://localhost:8000/docs)

---

## 📡 Nạp Firmware Cho ESP32-CAM / ESP32-S3

1. Mở thư mục `esp32_firmware` bằng **VS Code** (đã cài PlatformIO).
2. Mở tệp `include/HardwareController.h` hoặc `src/main.cpp`, cấu hình thông tin WiFi:
   ```cpp
   const char* ssid = "YOUR_WIFI_NAME";
   const char* password = "YOUR_WIFI_PASSWORD";
   ```
3. Cắm cáp USB nối ESP32 vào máy tính và nhấn nút **Upload** (Mũi tên sang phải) trên thanh công cụ PlatformIO để nạp code.

---

## 📤 Hướng Dẫn Đẩy Lên GitHub

> [!IMPORTANT]
> Trong thư mục `server_module/models/` có tệp mô hình `weights.onnx` nặng khoảng **108 MB**. GitHub giới hạn tệp tải lên tối đa là 100 MB. Do đó, bạn nên sử dụng **Git LFS** khi đẩy lên:

```bash
# 1. Khởi tạo Git repository trong thư mục ekyc_release
cd ekyc_release
git init

# 2. Cài đặt Git LFS
git lfs install
git lfs track "*.onnx" "*.pt" "*.task"

# 3. Thêm file và commit
git add .
git commit -m "feat: initial commit for eKYC AI Server and ESP32 Firmware"

# 4. Đẩy lên GitHub repo mới của bạn
git remote add origin https://github.com/YOUR_USERNAME/YOUR_REPO_NAME.git
git branch -M main
git push -u origin main
```
