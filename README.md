# 🛡️ Face-Project: Hệ Thống eKYC Face ID - Anti-Spoofing & Liveness Detection Pipeline

<p align="center">
  <img src="https://img.shields.io/badge/Python-3.10%20%7C%203.11-blue?logo=python" alt="Python Version" />
  <img src="https://img.shields.io/badge/Node.js-v18+-green?logo=node.js" alt="Node.js" />
  <img src="https://img.shields.io/badge/FastAPI-v0.110+-009688?logo=fastapi" alt="FastAPI" />
  <img src="https://img.shields.io/badge/PyTorch-%3E%3D2.0-orange?logo=pytorch" alt="PyTorch" />
  <img src="https://img.shields.io/badge/OpenCV-%3E%3D4.8-green?logo=opencv" alt="OpenCV" />
  <img src="https://img.shields.io/badge/ESP32--S3-CAM%20WROOM%20N16R8-red?logo=espressif" alt="ESP32-S3" />
  <img src="https://img.shields.io/badge/Model%201-YOLO__4%20Anti--Spoof%20(Latest)-yellow" alt="YOLO 4" />
  <img src="https://img.shields.io/badge/Model%202-RF--DETR%20Small%20Transformer%20(Latest)-red" alt="RF-DETR" />
  <img src="https://img.shields.io/badge/MediaPipe-Face%20Landmarker%20478-blueviolet" alt="MediaPipe" />
</p>

Hệ thống xác thực danh tính sinh trắc học khuôn mặt chuẩn FinTech / Ngân hàng (**eKYC Face Verification & Access Control**). Dự án kết hợp khép kín giữa **Thiết bị nhúng ESP32-S3 CAM (hoặc Webcam)**, **Node.js Stream Relay & Web Controller (:3000)** và **AI Server Microservice (FastAPI + PyTorch + ONNX Runtime :8000)**.

> [!IMPORTANT]
> **Hệ thống đang sử dụng kết hợp đồng thời CẢ 2 MÔ HÌNH CHỐNG GIẢ MẠO MỚI NHẤT (Ensemble Dual-Model Architecture):**
> 1. ⚡ **YOLO_4 (`Anti_Spoof_YOLO_4.pt`)**: Mô hình CNN YOLOv8 mới nhất chuyên biệt cho Face Anti-Spoofing, phản hồi siêu tốc, cực nhạy với vân in và viền màn hình.
> 2. 🧠 **RF-DETR Small Transformer (`weights.onnx`)**: Mô hình Real-Time DEtection TRansformer mới nhất từ Roboflow, sử dụng cơ chế Self-Attention đa tầng để soi cấu trúc ánh sáng, chiều sâu và các chiêu trò giả mạo tinh vi (Replay Attack, Silicone Mask, Màn hình OLED).

---

## 📑 Mục Lục
1. [Link Tải Tất Cả Mô Hình AI (Google Drive)](#-link-tải-tất-cả-mô-hình-ai-google-drive)
2. [Chi Tiết 2 Mô Hình AI Mới Nhất (YOLO_4 & RF-DETR)](#-chi-tiết-2-mô-hình-ai-mới-nhất-yolo_4--rf-detr)
3. [Kiến Trúc Xác Thực 3 Bước eKYC (Pipeline Overview)](#-kiến-trúc-xác-thực-3-bước-ekyc-pipeline-overview)
4. [Tính Năng Nổi Bật & Giao Diện Web UI](#-tính-năng-nổi-bật--giao-diện-web-ui)
5. [Cấu Trúc Thư Mục Dự Án](#-cấu-trúc-thư-mục-dự-án)
6. [Cài Đặt & Khởi Chạy Máy Chủ AI & Node.js](#-cài-đặt--khởi-chạy-máy-chủ-ai--nodejs)
7. [Cài Đặt & Nạp Code Firmware ESP32-S3 CAM](#-cài-đặt--nạp-code-firmware-esp32-s3-cam)
8. [Tài Liệu API Endpoints (FastAPI)](#-tài-liệu-api-endpoints-fastapi)
9. [Khắc Phục Sự Cố Thường Gặp (Troubleshooting)](#-khắc-phục-sự-cố-thường-gặp-troubleshooting)

---

## 📦 Link Tải Tất Cả Mô Hình AI (Google Drive)

Toàn bộ trọng số mô hình đã huấn luyện được lưu trữ tại Google Drive:
* 🔗 **Google Drive Repository:** [Google Drive - Face Project Models Folder](https://drive.google.com/drive/folders/1O7lqzhpJ8DE9x2AFzMyrd3M2-8sNdYBn)

### Bảng đối chiếu model sử dụng trong hệ thống:
| Phân loại | Tên File Model | Vị trí trong Project | Kích thước | Chức năng chính & Trạng thái |
|:---|:---|:---|:---:|:---|
| **Lõi AI Chính 1** | `Anti_Spoof_YOLO_4.pt` | `server_module/models/` | ~6.2 MB | **Model Mới Nhất 1:** YOLO_4 Face Anti-Spoofing CNN chuyên biệt |
| **Lõi AI Chính 2** | `roboflow/.../weights.onnx` | `server_module/models/` | ~108.9 MB | **Model Mới Nhất 2:** RF-DETR Small Vision Transformer đa tầng |
| **Phát Hiện Mặt** | `Face_Detection.pt` | `server_module/models/` | ~19 MB | YOLO Face Detection độ nhạy cao (IoU $\ge 0.40$) |
| **3D Landmarks** | `face_landmarker.task` | `server_module/models/` | ~3.7 MB | MediaPipe 478 Landmark 3D đo EAR chớp mắt & Yaw quay đầu |
| *Tham chiếu cũ* | `Anti_Spoof_minifasnet.pth` | `models/` | ~240 KB | CNN MiniFASNetV2 (bản benchmark đối chứng) |
| *Tham chiếu cũ* | `Model_MobilenetV2/` | `models/` | ~9 MB | MobileNetV2 Safetensors (bản benchmark đối chứng) |

---

## 🔬 Chi Tiết 2 Mô Hình AI Mới Nhất (YOLO_4 & RF-DETR)

Hệ thống kết hợp sức mạnh cộng hưởng giữa **CNN (Đặc trưng cục bộ)** và **Transformer (Ngữ cảnh toàn cục)**:

```
                                  ┌───────────────────────────────┐
                                  │   Ảnh Khuôn Mặt (224x224)    │
                                  └──────────────┬────────────────┘
                                                 │
                        ┌────────────────────────┴────────────────────────┐
                        │                                                 │
                        ▼                                                 ▼
        ┌───────────────────────────────┐                 ┌───────────────────────────────┐
        │  MODEL 1: YOLO_4 (v8 CNN)     │                 │  MODEL 2: RF-DETR TRANSFORMER │
        │  File: Anti_Spoof_YOLO_4.pt   │                 │  File: weights.onnx (ONNX RT) │
        ├───────────────────────────────┤                 ├───────────────────────────────┤
        │ • Trích xuất đặc trưng vi mô  │                 │ • Multi-Head Self-Attention   │
        │ • Vân giấy in, viền cắt ảnh   │                 │ • Bối cảnh toàn cục (Global)  │
        │ • Chói lóa mép kính điện thoại│                 │ • Màn hình OLED, video replay │
        │ • Tốc độ: ~15-25ms            │                 │ • Tốc độ: ~45-60ms            │
        └──────────────┬────────────────┘                 └───────────────┬───────────────┘
                       │ (Conf_YOLO, Label_YOLO)                          │ (Conf_RF, Label_RF)
                       └────────────────────────┬─────────────────────────┘
                                                │
                                                ▼
                         ┌──────────────────────────────────────────────┐
                         │   ENSEMBLE ENGINE & STRICT VETO ARBITRATION  │
                         │   • IoU Matching khuôn mặt >= 0.40           │
                         │   • Soft-Voting Confidence Fusion            │
                         │   • Phủ quyết an ninh (Strict Veto >= 68%)   │
                         │   • Đạt cả 2 model mới duyệt APPROVED        │
                         └──────────────────────────────────────────────┘
```

1. **Model 1: YOLO_4 (`Anti_Spoof_YOLO_4.pt`):**
   - Được huấn luyện tối ưu hóa trên nền YOLOv8 mới nhất dành riêng cho khuôn mặt.
   - Tập trung vào các sai khác tần số cao (High-frequency details): hạt mực in, độ thô ráp của bề mặt da thật so với bề mặt nhẵn bóng của màn hình, viền cắt mép ảnh giả mạo.
2. **Model 2: RF-DETR Small Transformer (`weights.onnx`):**
   - Ứng dụng kiến trúc Real-Time Detection Transformer hiện đại từ Roboflow.
   - Nhờ cơ chế Self-Attention, mô hình phân tích mối tương quan ánh sáng giữa khuôn mặt và môi trường xung quanh, phát hiện các trường hợp giả mạo tinh vi mà CNN dễ bỏ sót (ảnh chiếu lại trên màn hình iPad/TV, mặt nạ 3D silicone, ảnh in xuyên sáng).
3. **Cơ chế Phủ Quyết An Ninh (Strict Spoof Veto):**
   - Nếu **bất kỳ model nào** đưa ra cảnh báo giả mạo với độ tin cậy $\ge 68\%$, hệ thống lập tức phủ quyết thành **SPOOF**, ngăn chặn triệt để tình trạng lọt gian lận.
   - Bảng kết quả trả về hiển thị chi tiết độ tin cậy của từng model: `both_detected` (cả 2 model đều phát hiện mặt) và `agreement` (cả 2 đồng thuận kết luận).

---

## 🔄 Kiến Trúc Xác Thực 3 Bước eKYC (Pipeline Overview)

Hệ thống hoạt động theo quy trình xác thực khép kín:

```
                    [ KHỞI ĐẦU: NGƯỜI DÙNG TIẾP CẬN ]
                                   │
                                   ▼
 ┌───────────────────────────────────────────────────────────────────┐
 │ BƯỚC 1: CANH OVAL & ENSEMBLE ANTI-SPOOF (YOLO_4 + RF-DETR)        │
 │ • Chụp ảnh khuôn mặt sắc nét trong khung Oval phát sáng neon      │
 │ • Phát hiện mặt & Đánh giá 3D Pose nhìn thẳng (|Yaw|<=25°, Pitch) │
 │ • Đồng thuận 2 model mới nhất: YOLO_4 + RF-DETR (Strict Veto)     │
 └─────────────────────────────────┬─────────────────────────────────┘
                                   │
                    ┌──────────────┴──────────────┐
                    │                             │
             [SPOOF / LỖI MẶT]              [REAL CHUẨN]
                    │                             │
                    ▼                             ▼
         [FAIL-FAST: DỪNG NGAY]       [TẠO SESSION_ID + GIAO THỬ THÁCH]
         • Hiển thị viền đỏ cảnh báo              │
         • KHÔNG chạy tiếp các bước sau           ▼
                                ┌────────────────────────────────────┐
                                │ BƯỚC 2: THỬ THÁCH CHỚP MẮT (BLINK) │
                                │ • Đo lường chỉ số mở mắt EAR       │
                                │ • Baseline thích ứng từng cá nhân  │
                                │ • Chu kỳ: Mở -> Nhắm -> Mở         │
                                └─────────────────┬──────────────────┘
                                                  │ (Chớp mắt ĐẠT)
                                                  ▼
                                ┌────────────────────────────────────┐
                                │ BƯỚC 3: THỬ THÁCH QUAY ĐẦU (HEAD)  │
                                │ • Thử thách ngẫu nhiên TRÁI / PHẢI │
                                │ • Đo góc Euler Yaw qua 3D PnP      │
                                │ • Nhích nhẹ ~8 độ là vượt qua      │
                                └─────────────────┬──────────────────┘
                                                  │ (Quay đầu ĐẠT)
                                                  ▼
                                ┌────────────────────────────────────┐
                                │ BƯỚC 4: KẾT QUẢ DUAL WINDOW        │
                                │ 1. Kích hoạt Relay mở cửa ESP32    │
                                │ 2. Xuất bảng Side-by-Side 5 phần   │
                                │ 3. Bắn Webhook sang Node.js Backend│
                                └────────────────────────────────────┘
```

---

## 🌟 Tính Năng Nổi Bật & Giao Diện Web UI

### 1. Web UI Hiện Đại Với Nút Chuyển Đổi Camera (Webcam ↔ ESP32-CAM)
* Giao diện điều khiển tại `http://localhost:3000/` được tích hợp sẵn 3 chế độ nguồn hình ảnh:
  - **[💻 Webcam]:** Sử dụng webcam để kiểm thử nhanh.
  - **[📡 ESP32-CAM Stream]:** Chuyển sang thu nhận luồng video thời gian thực từ ESP32-CAM truyền lên Node.js.
  - **[📁 Tải File]:** Thẩm định file ảnh có sẵn trên máy.
  - **Phím tắt nhanh `C`:** Bấm phím `C` trên bàn phím để chuyển đổi camera tức thì.

### 2. Bảng Kết Quả Dual Window (Side-by-Side) Chuẩn `test_pipeline_ensemble_full.py`
* Kết quả hiển thị song song gồm 2 cửa sổ:
  - **Cửa sổ trái:** Ảnh khuôn mặt gốc với khung Bbox và hiệu ứng Bokeh mask làm mờ bối cảnh ngoại vi.
  - **Cửa sổ phải:** Dashboard chẩn đoán AI 5 phần: Tiêu đề duyệt, Khuôn mặt crop 224x224, Góc Euler 3D Pose, Chi tiết đồng thuận 2 model YOLO_4 & RF-DETR, và Trạng thái Active Liveness (EAR + Yaw).
  - Hỗ trợ nút **[🔍 Xem To]** mở Modal phóng to toàn màn hình và **[💾 Tải Ảnh]** để lưu trữ bằng chứng xác thực.          │ 2. Nháy đèn LED flash xác nhận     │
                                │ 3. Bắn Webhook sang Node.js Backend│
                                └────────────────────────────────────┘
```

---

## 🌟 Công Nghệ & Tính Năng Nổi Bật

### 1. Cơ Chế Chuyển Đổi Độ Phân Giải Động (Hybrid Resolution)
* **Khởi tạo (`initCamera`)**: Cấp phát frame buffer trong PSRAM ở mức trần `FRAMESIZE_VGA` (640x480).
* **Bước 1 (Face Detect & Anti-Spoofing)**: Chuyển sang `FRAMESIZE_VGA` (`jpeg_quality = 10`). Khuôn mặt có độ phân giải lớn gấp 2.5 lần, giúp trích xuất chi tiết viền da, con ngươi và soi vân in/vân màn hình.
* **Bước 2 & 3 (Eye Blink & Head Movement)**: Tự động hạ về `FRAMESIZE_240X240` (`jpeg_quality = 15`). File ảnh chỉ nặng 6–8KB, tốc độ bắn ảnh lên server đạt 15–20 FPS, bắt trọn từng khoảnh khắc chớp mắt và quay đầu.

### 2. Lõi Ensemble 2 Model Độc Lập & Veto Rule
* **Khớp Bounding Box IoU ($\ge 0.40$):** Đồng bộ vị trí mặt giữa CNN (YOLO) và Vision Transformer (RF-DETR).
* **Quy tắc Phủ Quyết An Ninh (Strict Spoof Veto $\ge 68\%$):** Khi 1 trong 2 mô hình phát hiện dấu hiệu giả mạo với độ tin cậy $\ge 68\%$, hệ thống lập tức phủ quyết thành **SPOOF** để chặn gian lận ảnh in hoặc màn hình điện thoại/iPad.
* **Fail-Fast Policy:** Nếu Bước 1 phát hiện giả mạo hoặc không có mặt, phiên kết thúc ngay lập tức mà không tốn tài nguyên chạy các bước sau.

### 3. Tự Động Cứu Sáng Thích Nghi (Adaptive Gamma Correction)
* Module `preprocess_esp32_image` tự động chuyển đổi sang không gian màu LAB. Nếu giá trị kênh độ sáng trung bình $L < 85$, thuật toán tự động áp dụng bảng tra cứu **Gamma LUT ($1.45$)** và **CLAHE** để kích sáng khuôn mặt trước khi đưa vào mô hình nhận diện.

### 4. Quy Ước Góc PnP Head Pose Chuẩn
* Sử dụng thuật toán `cv2.solvePnP` kết hợp 468 điểm MediaPipe:
  - **Quay TRÁI**: $\Delta\text{Yaw} \le -5.0^\circ$ (hoặc $\text{Yaw} \le -7.5^\circ$).
  - **Quay PHẢI**: $\Delta\text{Yaw} \ge +5.0^\circ$ (hoặc $\text{Yaw} \ge +7.5^\circ$).
  - Nới lỏng ngưỡng góc và cho phép vượt qua chỉ sau 1 frame đạt chuẩn rõ rệt.

---

## 📁 Cấu Trúc Thư Mục Dự Án

```
Face-Project/
├── start_ai_server.bat               # ⚡ Script chạy nhanh AI Server (:8000)
├── start_nodejs_receiver.bat         # ⚡ Script chạy nhanh Node.js Receiver (:3000)
├── requirements.txt                  # Danh sách dependencies đã chuẩn hóa
├── README.md                         # Tài liệu hướng dẫn dự án
│
├── CameraWebServer/                  # 📷 MÃ NGUỒN FIRMWARE ESP32-S3 CAM
│   └── CameraToAIServer_PIO/         # Dự án PlatformIO (Khuyên dùng)
│       ├── src/main.cpp              # Logic nạp camera, WiFi, HTTP client & Relay
│       ├── include/web_ui.h          # Giao diện Web điều khiển trực tiếp trên ESP32
│       └── platformio.ini           # Cấu hình board esp32cam_s3_wroom_n16r8
│
├── server_module/                    # 🚀 MODULE AI SERVER & MICROSERVICE
│   ├── app.py                        # FastAPI Server & Endpoints (Entry point)
│   ├── config.py                     # Cấu hình ngưỡng AI & paths nội bộ
│   ├── esp32_challenge.py            # Quản lý phiên eKYC 3 bước (ChallengeSession)
│   ├── ensemble_anti_spoof.py        # Cụm Ensemble (YOLO_4 + RF-DETR Small)
│   ├── pipeline_server.py            # Core Pipeline Headless (EKYCPipelineServer)
│   ├── utils.py                      # Tiền xử lý ảnh, CLAHE, IoU, Base64 & HUD
│   └── models/                       # Trọng số model tự chứa (self-contained)
│       ├── Anti_Spoof_YOLO_4.pt
│       ├── Face_Detection.pt
│       ├── face_landmarker.task
│       └── roboflow/.../weights.onnx
│
├── tests/                            # Bộ kiểm thử tích hợp
│   ├── test_fastapi_server.py        # Kiểm thử toàn diện API
│   └── test_pipeline_ensemble_full.py# Thử nghiệm webcam máy tính OpenCV
└── output/                           # Thư mục xuất artifacts & báo cáo
```

---

## ⚙️ Cài Đặt & Khởi Chạy Máy Chủ AI

### 1. Cài đặt môi trường Python
Khuyên dùng **Python 3.10 hoặc 3.11** trên Windows/Linux:

```bash
# 1. Tạo môi trường ảo
python -m venv venv

# 2. Kích hoạt môi trường
# Trên Windows PowerShell:
venv\Scripts\Activate.ps1
# Trên Linux/macOS:
source venv/bin/activate

# 3. Cài đặt dependencies
pip install -r requirements.txt
```

### 2. Khởi chạy Hệ Thống (AI Server + Node.js Web Relay)

Hệ thống hoạt động theo mô hình 2 máy chủ phối hợp nhịp nhàng:

#### A. Khởi chạy FastAPI AI Server (:8000)
Chạy suy luận cụm mô hình chống giả mạo mới nhất **YOLO_4 + RF-DETR + MediaPipe 3D**:
* **Cách 1 (Khuyên dùng):** Nhấp đúp file [start_ai_server.bat](file:///c:/Users/HP/Desktop/Face-Project/start_ai_server.bat)
* **Cách 2:** Chạy lệnh terminal:
  ```bash
  python server_module/app.py
  ```

#### B. Khởi chạy Node.js Stream Relay & Web Controller (:3000)
Chịu trách nhiệm nhận luồng ESP32, broadcast MJPEG stream, reverse proxy API và phục vụ giao diện Web UI:
* **Cách 1 (Khuyên dùng):** Nhấp đúp file [start_nodejs_receiver.bat](file:///c:/Users/HP/Desktop/Face-Project/start_nodejs_receiver.bat)
* **Cách 2:** Chạy lệnh terminal (100% thư viện chuẩn Node.js, không cần npm install):
  ```bash
  node server_module/nodejs_server_receiver.js
  ```

#### Các cổng dịch vụ sau khi khởi động:
* 🌐 **Web eKYC Controller (Giao diện chính):** [http://localhost:3000/](http://localhost:3000/) *(Hỗ trợ nút chuyển Webcam / ESP32-CAM và xem kết quả Dual-Window)*
* 📡 **Luồng Video MJPEG ESP32:** [http://localhost:3000/stream](http://localhost:3000/stream)
* 🚀 **FastAPI AI Server:** [http://127.0.0.1:8000/](http://127.0.0.1:8000/)
* 📚 **Tài liệu API Swagger UI:** [http://127.0.0.1:8000/docs](http://127.0.0.1:8000/docs)

---

## 📷 Cài Đặt & Nạp Code Firmware ESP32-S3 CAM

### Phần cứng hỗ trợ:
* Board **ESP32-S3 CAM** (Chip ESP32-S3 WROOM N16R8, 16MB Flash, 8MB Octal PSRAM).
* Module Camera: **OV2640**.

### Cách 1: Nạp qua PlatformIO (Khuyên dùng)
1. Mở thư mục `CameraWebServer/CameraToAIServer_PIO` bằng VS Code.
2. Kiểm tra thông tin WiFi trong file `src/main.cpp`:
   ```cpp
   const char *ssid = "TÊN_WIFI";
   const char *password = "MAT_KHAU";
   String ai_server_ip = "192.168.1.X"; // IP máy tính chạy AI Server
   ```
3. Cắm cáp USB vào cổng COM ESP32 và chạy lệnh:
   ```bash
   pio run -t upload
   ```
4. Mở Monitor kiểm tra:
   ```bash
   pio device monitor -b 115200
   ```

### Cách 2: Nạp qua Arduino IDE
1. Mở file `CameraWebServer/CameraToAIServer/CameraToAIServer.ino`.
2. Chọn Board: **ESP32S3 Dev Module**.
3. Cấu hình:
   - **PSRAM:** OPI PSRAM
   - **Flash Size:** 16MB
   - **Partition Scheme:** 16M Flash (3MB APP/9.9MB FATFS)
4. Nhấn **Upload**.

---

## 📡 Tài Liệu API Endpoints (FastAPI)

### 1. `POST /api/v1/esp32/challenge/start`
* **Chức năng:** Bước 1 - Nhận ảnh snapshot chất lượng cao, kiểm tra Face Detect và duyệt Ensemble Anti-Spoofing.
* **Payload:** Binary JPEG (Content-Type: `image/jpeg`) hoặc Form-Data.
* **Response chính:**
  ```json
  {
    "success": true,
    "session_id": "8f3b2a1c-...",
    "step": "eye_blink",
    "is_real": true,
    "confidence": 0.895,
    "target_head_action": "TURN_LEFT",
    "head_prompt": "Hãy quay mặt sang bên TRÁI",
    "captured_image_base64": "..."
  }
  ```

### 2. `POST /api/v1/esp32/challenge/step`
* **Chức năng:** Bước 2 & 3 - Nhận chuỗi frame liên tục cho thử thách Chớp mắt (`eye_blink`) và Quay đầu (`head_movement`).
* **Header / Query Params:** `session_id`, `step`.
* **Giới hạn thời gian:** **10 giây** cho mỗi thử thách.
* **Response khi hoàn thành toàn diện:**
  ```json
  {
    "success": true,
    "passed": true,
    "approved": true,
    "step": "completed",
    "confidence": 0.912,
    "message": "Xác thực thành công! Người thật (REAL)."
  }
  ```

### 3. `POST /api/v1/esp32/challenge/reset`
* **Chức năng:** Hủy hoặc làm mới phiên xác thực.

### 4. `GET /api/v1/health`
* **Chức năng:** Kiểm tra tình trạng nạp model (`models_loaded: true`) và GPU/CPU.

---

## 💻 Tích Hợp Webhook & Node.js Receiver Hub

Hệ thống đã tích hợp sẵn máy chủ **Node.js Receiver Hub** tại [server_module/nodejs_server_receiver.js](file:///c:/Users/HP/Desktop/Face-Project/server_module/nodejs_server_receiver.js) (chạy thuần 100% Node.js tiêu chuẩn, không cần cài thư viện ngoài).

Khi người dùng hoàn tất quá trình xác thực, AI Server tự động gửi toàn bộ kết quả chẩn đoán qua Webhook:
* **Endpoint tiếp nhận:** `POST http://localhost:3000/api/ekyc/result`
* **Xử lý tự động của Node.js:**
  - Lưu ảnh khuôn mặt Crop vào `server_module/captured_faces/face_<timestamp>_<verdict>.jpg`.
  - Lưu ảnh **Dual-Window Side-by-Side** vào `server_module/captured_faces/dual_<timestamp>_<verdict>.jpg`.
  - Cập nhật danh sách lịch sử xác thực thời gian thực tại `GET /api/ekyc/history`.
  - In thông báo màu trực quan lên Terminal máy chủ Node.js.

```bash
# Cấu trúc payload gửi từ AI Server:
{
  "session_id": "SES_1727170000",
  "approved": true,
  "verdict": "APPROVED",
  "is_real": true,
  "confidence": 0.942,
  "dual_window_image_base64": "data:image/jpeg;base64,...",
  "crop_face_base64": "data:image/jpeg;base64,...",
  "processing_time_ms": 78.5
}
```

---

## 🛠️ Khắc Phục Sự Cố Thường Gặp (Troubleshooting)

1. **Camera bị tối ở Bước 1:**
   * Hệ thống đã tích hợp `capturePhotoSafe()` xả 5 frame đệm trước khi chụp để thuật toán AEC/AGC kịp đo sáng. Đồng thời chế độ Hybrid tự động kích hoạt `FRAMESIZE_VGA` để thu lượng photon tối đa.
2. **Hiện tượng dội sáng (Flare / Chói đèn) do bóng đèn gần camera:**
   * **Nguyên nhân:** Ống kính nhựa của ESP32 không có lớp phủ quang học chống lóa.
   * **Khắc phục phần mềm:** Firmware đã hạ `contrast = 0`, kích hoạt `aec2 = 1` và bộ lọc `wpc/bpc` khử điểm cháy trắng.
   * **Khắc phục vật lý:** Quấn một đoạn băng dính đen nhô ra khỏi thấu kính 5–8mm làm loa che nắng (Lens hood), và di chuyển đèn chếch 45 độ, tránh chiếu thẳng vào mắt kính.
3. **Quá thời gian thử thách (10s Timeout):**
   * Đảm bảo đứng cách camera 35–50cm, thực hiện dứt khoát: nhắm mắt giữ 0.5s rồi mở ra, hoặc quay nhẹ mặt sang hướng được chỉ định trên màn hình.

---

<p align="center">
  <sub>Phát triển & Tối ưu bởi <strong>GiantKy</strong> &bull; 2026</sub>
</p>
