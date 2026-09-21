# 🛡️ Face-Project: Hệ Thống eKYC Face ID - Anti-Spoofing & Liveness Detection Pipeline

<p align="center">
  <img src="https://img.shields.io/badge/Python-3.10%20%7C%203.11-blue?logo=python" alt="Python Version" />
  <img src="https://img.shields.io/badge/FastAPI-v0.110+-009688?logo=fastapi" alt="FastAPI" />
  <img src="https://img.shields.io/badge/PyTorch-%3E%3D2.0-orange?logo=pytorch" alt="PyTorch" />
  <img src="https://img.shields.io/badge/OpenCV-%3E%3D4.8-green?logo=opencv" alt="OpenCV" />
  <img src="https://img.shields.io/badge/ESP32--S3-CAM%20WROOM%20N16R8-red?logo=espressif" alt="ESP32-S3" />
  <img src="https://img.shields.io/badge/YOLO-v8%20Face%20%26%20Anti--Spoof%204-yellow" alt="YOLO" />
  <img src="https://img.shields.io/badge/Roboflow-RF--DETR%20Small%20Transformer-red" alt="RF-DETR" />
  <img src="https://img.shields.io/badge/MediaPipe-Face%20Landmarker%20478-blueviolet" alt="MediaPipe" />
</p>

Hệ thống xác thực danh tính sinh trắc học khuôn mặt chuẩn FinTech / Ngân hàng (**eKYC Face Verification & Access Control**). Dự án kết hợp khép kín giữa **Thiết bị nhúng ESP32-S3 CAM (Điều khiển cửa Relay)** và **AI Server Microservice (FastAPI + PyTorch + Tensor/ONNX)**, tích hợp: phát hiện khuôn mặt, trích xuất 478 điểm mốc 3D, **chống giả mạo đa tầng kết hợp Ensemble (YOLO_4 + RF-DETR Small Transformer)** và xác thực cử động sống chủ động (Active Liveness: Chớp mắt & Quay đầu).

---

## 📑 Mục Lục
1. [Link Tải Tất Cả Mô Hình AI (Google Drive)](#-link-tải-tất-cả-mô-hình-ai-google-drive)
2. [Kiến Trúc Xác Thực 3 Bước eKYC (Pipeline Overview)](#-kiến-trúc-xác-thực-3-bước-ekyc-pipeline-overview)
3. [Công Nghệ & Tính Năng Nổi Bật](#-công-nghệ--tính-năng-nổi-bật)
4. [Cấu Trúc Thư Mục Dự Án](#-cấu-trúc-thư-mục-dự-án)
5. [Cài Đặt & Khởi Chạy Máy Chủ AI](#-cài-đặt--khởi-chạy-máy-chủ-ai)
6. [Cài Đặt & Nạp Code Firmware ESP32-S3 CAM](#-cài-đặt--nạp-code-firmware-esp32-s3-cam)
7. [Tài Liệu API Endpoints (FastAPI)](#-tài-liệu-api-endpoints-fastapi)
8. [Tích Hợp Webhook Node.js Backend](#-tích-hợp-webhook-nodejs-backend)
9. [Khắc Phục Sự Cố Thường Gặp (Troubleshooting)](#-khắc-phục-sự-cố-thường-gặp-troubleshooting)

---

## 📦 Link Tải Tất Cả Mô Hình AI (Google Drive)

Toàn bộ trọng số mô hình đã huấn luyện được lưu trữ tại Google Drive:
* 🔗 **Google Drive Repository:** [Google Drive - Face Project Models Folder](https://drive.google.com/drive/folders/1O7lqzhpJ8DE9x2AFzMyrd3M2-8sNdYBn)

### Bảng đối chiếu model sử dụng:
| Tên File Model | Vị trí trong Project | Kích thước | Chức năng chính |
|:---|:---|:---:|:---|
| `Face_Detection.pt` | `server_module/models/` & `models/` | ~19 MB | YOLO Face Detection độ nhạy cao |
| `face_landmarker.task` | `server_module/models/` & `models/` | ~3.7 MB | Google MediaPipe 478 Landmarks 3D |
| `Anti_Spoof_YOLO_4.pt` | `server_module/models/` & `models/` | ~6.2 MB | Model 1: YOLOv8 Face Anti-Spoofing |
| `roboflow/**/weights.onnx` | `server_module/models/` & `models/` | ~108.9 MB | Model 2: RF-DETR Small Transformer |
| `Anti_Spoof_minifasnet.pth` | `models/` | ~240 KB | CNN MiniFASNetV2 PyTorch |
| `Model_MobilenetV2/` | `models/` | ~9 MB | MobileNetV2 Safetensors |

---

## 🔄 Kiến Trúc Xác Thực 3 Bước eKYC (Pipeline Overview)

Hệ thống hoạt động theo mô hình **Push-Image linh hoạt từ ESP32-S3 lên AI Server**:

```
                    [ KHỞI ĐẦU: NGƯỜI DÙNG TIẾP CẬN ]
                                   │
                                   ▼
 ┌───────────────────────────────────────────────────────────────────┐
 │ BƯỚC 1: SNAPSHOT VGA 640x480 - FACE DETECT & ENSEMBLE ANTI-SPOOF  │
 │ • Chụp 1 ảnh tĩnh VGA sắc nét (Pixel size lớn, chống bết dính)    │
 │ • Phát hiện mặt 5 tầng (YOLO 0.28 -> CLAHE -> MediaPipe BlazeFace)│
 │ • Đánh giá tư thế nhìn thẳng (|Yaw| <= 25°, |Pitch| <= 22°)       │
 │ • Cụm Ensemble Anti-Spoof: YOLO_4 + RF-DETR Small (Strict Veto)   │
 └─────────────────────────────────┬─────────────────────────────────┘
                                   │
                    ┌──────────────┴──────────────┐
                    │                             │
             [SPOOF / LỖI MẶT]              [REAL CHUẨN]
                    │                             │
                    ▼                             ▼
         [FAIL-FAST: DỪNG NGAY]       [TẠO SESSION_ID + GIAO THỬ THÁCH]
         • Hiển thị ảnh chụp viền đỏ              │
         • KHÔNG chạy tiếp các bước sau           ▼
                                ┌────────────────────────────────────┐
                                │ BƯỚC 2: PUSH-IMAGE 240x240 - BLINK │
                                │ • Tự động hạ về 240x240 (FPS cao)  │
                                │ • Giới hạn thời gian: 10 giây      │
                                │ • EAR giảm > 15% so với baseline   │
                                │ • Chu kỳ: Mở -> Nhắm -> Mở         │
                                └─────────────────┬──────────────────┘
                                                  │ (Chớp mắt ĐẠT)
                                                  ▼
                                ┌────────────────────────────────────┐
                                │ BƯỚC 3: PUSH-IMAGE 240x240 - HEAD  │
                                │ • Thử thách ngẫu nhiên TRÁI / PHẢI │
                                │ • Giới hạn thời gian: 10 giây      │
                                │ • PnP Pose chuẩn: Quay Trái: Yaw-  │
                                │                   Quay Phải: Yaw+  │
                                │ • Biên độ >= 5.0° (pass sau 1 shot)│
                                └─────────────────┬──────────────────┘
                                                  │ (Quay đầu ĐẠT)
                                                  ▼
                                ┌────────────────────────────────────┐
                                │ KẾT QUẢ CUỐI CÙNG (APPROVED REAL)  │
                                │ 1. Kích hoạt Relay mở cửa ESP32    │
                                │ 2. Nháy đèn LED flash xác nhận     │
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

### 2. Khởi chạy AI Server
Cách 1: Chạy file script tự động:
```cmd
start_ai_server.bat
```
Cách 2: Chạy trực tiếp qua Python:
```bash
python server_module/app.py
```
Sau khi khởi động thành công:
* **API Server:** Chạy tại [http://0.0.0.0:8000](http://0.0.0.0:8000)
* **Tài liệu API Swagger UI:** [http://127.0.0.1:8000/docs](http://127.0.0.1:8000/docs)
* **Web Scanner Client:** [http://127.0.0.1:8000/](http://127.0.0.1:8000/)

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

## 💻 Tích Hợp Webhook Node.js Backend

Khi người dùng hoàn tất toàn bộ 3 bước, AI Server tự động bắn kết quả tới Backend Node.js qua Webhook:

```javascript
// Example Express.js Webhook Handler
const express = require('express');
const app = express();
app.use(express.json({ limit: '10mb' }));

app.post('/api/ekyc/result', (req, res) => {
  const { session_id, approved, is_real, confidence, crop_face_base64 } = req.body;

  if (approved) {
    console.log(`[PASS] Phiên ${session_id} thành công! Độ tin cậy: ${(confidence * 100).toFixed(1)}%`);
    // Lưu ảnh khuôn mặt và mở khóa tài khoản/ghi log chấm công
  } else {
    console.log(`[REJECT] Phiên ${session_id} bị từ chối xác thực!`);
  }

  res.json({ received: true });
});

app.listen(3000, () => console.log('Node.js Webhook Server chạy tại port 3000'));
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
