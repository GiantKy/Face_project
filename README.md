# 🛡️ Face-Project: Hệ Thống eKYC Face ID - Ensemble Anti-Spoofing & Liveness Pipeline

<p align="center">
  <img src="https://img.shields.io/badge/Python-3.10%20%7C%203.11-blue?logo=python" alt="Python Version" />
  <img src="https://img.shields.io/badge/FastAPI-v0.110+-009688?logo=fastapi" alt="FastAPI" />
  <img src="https://img.shields.io/badge/Node.js-v18+-green?logo=node.js" alt="Node.js" />
  <img src="https://img.shields.io/badge/PyTorch-%3E%3D2.0-orange?logo=pytorch" alt="PyTorch" />
  <img src="https://img.shields.io/badge/ESP32--S3-CAM%20WROOM%20N16R8-red?logo=espressif" alt="ESP32-S3" />
  <img src="https://img.shields.io/badge/Ensemble-YOLOv8%20%2B%20RF--DETR-yellow" alt="Ensemble Anti-Spoof" />
</p>

Hệ thống xác thực danh tính sinh trắc học khuôn mặt chuẩn ngân hàng / FinTech (**eKYC Face Verification & Anti-Spoofing**). Dự án hỗ trợ linh hoạt cả **Webcam máy tính** và **Camera nhúng ESP32-S3**, kết hợp cùng **Node.js Web Controller (:3000)** và **FastAPI AI Microservice (:8000)**.

---

## 📦 Link Tải Mô Hình AI (Google Drive)

Toàn bộ trọng số mô hình AI đã huấn luyện được phân loại chuẩn hóa và lưu trữ tại Google Drive. Vui lòng lưu ý tên thư mục trên Drive khi tải về:

* 🌐 **Link tải Toàn Bộ Mô Hình Dự Án (`models/`):**  
  - Tên thư mục trên Google Drive: **`models (2)`**  
  - 👉 [**Google Drive - Toàn Bộ Models Project [models (2)]**](https://drive.google.com/drive/folders/1zEIGdw3krTHwO8w5BmkCxSP0v6Wdw0LU?usp=drive_link)  
  - 📝 *Cách bố trí:* Sau khi tải thư mục **`models (2)`** về máy, đổi tên thư mục thành **`models`** (hoặc giải nén nội dung) và đặt tại thư mục gốc dự án: `Face-Project/models/`.

* 🚀 **Link tải Mô Hình Server Module (`server_module/models/`):**  
  - Tên thư mục trên Google Drive: **`models (3)`**  
  - 👉 [**Google Drive - Server Module Models [models (3)]**](https://drive.google.com/drive/folders/1w-Xcl0irJzlXPCzJuVRYWPepxnH-3oiX?usp=drive_link)  
  - 📝 *Cách bố trí:* Sau khi tải thư mục **`models (3)`** về máy, đổi tên thư mục thành **`models`** (hoặc giải nén nội dung) và đặt tại thư mục server: `Face-Project/server_module/models/`.

### 📊 Bảng Danh Mục Các Mô Hình Cốt Lõi:
| Tác vụ | File / Thư mục lưu trữ | Định dạng | Dung lượng | Vai trò trong hệ thống |
| :--- | :--- | :---: | :---: | :---|
| **1. Face Detection** | `models/face_detection/yolo_face_detection_official.pt` | PyTorch | 18.3 MB | Phát hiện khuôn mặt trong khung hình, canh vị trí oval |
| **2. 3D Landmarks** | `models/landmarks/mediapipe_face_landmarker_official.task` | MediaPipe | 3.6 MB | 478 tọa độ 3D đo EAR chớp mắt & Yaw/Pitch quay đầu |
| **3. Anti-Spoof 1** | `models/anti_spoof/yolo/yolo_anti_spoof_v4_official.pt` | PyTorch | 6.0 MB | CNN chuyên biệt bắt vân in, mép màn hình, độ chói lóa |
| **4. Anti-Spoof 2** | `models/roboflow/k-thi-gia-s-workspace/.../weights.onnx` | ONNX | 108.9 MB | RF-DETR Small Transformer phân tích chiều sâu đa tầng |
| **5. Vật che mặt** | `models/roboflow/glass-and-mask-q5de1/2/weights.onnx` | ONNX | 9.3 MB | Phát hiện Kính mắt & Khẩu trang (YOLO26n) |

---

## 🔄 Quy Trình Xác Thực eKYC 4 Bước (Pipeline Architecture)

```
[Camera Input] ──► [Bước 1: Oval & Occlusion] ──► [Bước 2: Chớp Mắt] ──► [Bước 3: Quay Đầu] ──► [Bước 4: Chụp & Ensemble AI] ──► [APPROVED / REJECTED]
```

1. **Bước 1 - Canh chỉnh Oval & Kiểm tra che mặt:**
   - Hướng dẫn khuôn mặt vào giữa khung Oval trung tâm.
   - AI YOLO26n phát hiện nếu người dùng đang đeo kính mắt hoặc khẩu trang, yêu cầu tháo ra trước khi thực hiện.
2. **Bước 2 - Thử thách chớp mắt tự nhiên (Active Liveness: Blink):**
   - Đo tỷ lệ mở mắt EAR (*Eye Aspect Ratio*) qua MediaPipe 3D Landmarks.
   - Giới hạn thời gian thử thách: **10 giây**.
3. **Bước 3 - Thử thách cử động đầu (Active Liveness: Head Movement):**
   - Hệ thống tự động sinh ngẫu nhiên hướng quay đầu (*Quay Trái / Quay Phải*).
   - Đo góc Euler Yaw ($\ge 16^\circ$), thời gian giới hạn: **10 giây**. Đồng bộ trực tiếp lệnh sang ESP32 và Web UI.
4. **Bước 4 - Chụp ảnh & Thẩm định Ensemble Anti-Spoofing (Passive Liveness):**
   - Chụp ảnh chất lượng cao độ phân giải gốc khi người dùng nhìn thẳng.
   - Thẩm định kép song song qua **Dual-Model Ensemble** (YOLO_4 CNN + RF-DETR Transformer) với cơ chế Soft-Voting và Strict Spoof Veto.

---

## 📁 Cấu Trúc Thư Mục Dự Án (Project Structure)

```text
Face-Project/
├── server_module/                 # [Microservice] FastAPI AI Server (:8000)
│   ├── app.py                     # Entry point FastAPI, WebSocket & REST APIs
│   ├── config.py                  # Toàn bộ cấu hình ngưỡng, timeout & đường dẫn
│   ├── pipeline_server.py         # Quản trị luồng eKYC 4 bước phía server
│   ├── ensemble_anti_spoof.py     # Lõi kết hợp YOLO_4 + RF-DETR Small
│   ├── esp32_challenge.py         # Quản trị phiên tương tác giữa ESP32 & Web UI
│   ├── static/                    # Giao diện Web UI HTML5/CSS/JS thời gian thực
│   └── models/                    # Thư mục chứa model phục vụ server_module
├── src/                           # [Core Modules] Các thư viện xử lý thị giác máy tính
│   ├── face_detection/            # Module YOLOv8 phát hiện khuôn mặt
│   ├── landmark_detection/        # Module MediaPipe 478 điểm 3D
│   ├── pose_validation/           # Module đo góc quay đầu Yaw/Pitch/Roll
│   ├── head_movement/             # Module quản trị thử thách cử động đầu
│   ├── face_alignment_crop/       # Module căn chỉnh xoay thẳng và crop 224x224
│   └── anti_spoof/                # Module phụ trợ Anti-Spoofing
├── models/                        # [Models Catalog] Quản lý toàn bộ trọng số AI
│   ├── face_detection/            # Trọng số YOLO Face Detection
│   ├── landmarks/                 # Trọng số MediaPipe Face Landmarker
│   ├── anti_spoof/                # Trọng số Anti-Spoofing (YOLO, RF-DETR, MiniFAS)
│   ├── face_occlusion/            # Trọng số Kính mắt & Khẩu trang
│   ├── roboflow/                  # Cache Offline cho Roboflow Inference SDK
│   └── README.md                  # Sổ tay chi tiết về từng phiên bản model
├── tests/                         # [Test Suites] Các kịch bản chạy thử nghiệm
│   └── test_pipeline_ensemble_full.py # Kịch bản kiểm thử toàn diện kèm GUI Webcam
├── CameraWebServer/               # Firmware C++ cho bo mạch ESP32-S3 CAM
├── start_ai_server.bat            # Script 1-click khởi chạy FastAPI Server
├── start_nodejs_receiver.bat      # Script 1-click khởi chạy Node.js Receiver
├── requirements.txt               # Danh sách thư viện Python cần thiết
└── .gitignore                     # Cấu hình bỏ qua file nhị phân & dữ liệu nhạy cảm
```

---

## 🚀 Hướng Dẫn Cài Đặt & Khởi Chạy (Quick Start)

### 1. Cài đặt môi trường
Yêu cầu: **Python 3.10 hoặc 3.11** và **Node.js 18+**.
```bash
# Clone repo về máy
git clone https://github.com/GiantKy/Face_project.git
cd Face_project

# Cài đặt các thư viện Python
pip install -r requirements.txt
```

### 2. Tải trọng số Mô hình AI
Tải mô hình từ Google Drive theo link ở đầu tài liệu:
1. Thư mục Drive **`models (2)`** ➔ Tải về, đổi tên thành **`models`** và đặt tại thư mục gốc: `Face-Project/models/` (dành cho toàn bộ dự án & test standalone).
2. Thư mục Drive **`models (3)`** ➔ Tải về, đổi tên thành **`models`** và đặt tại: `Face-Project/server_module/models/` (dành riêng cho server microservice).

### 3. Khởi chạy hệ thống

* **Cách 1: Chạy kiểm thử trực tiếp trên Webcam (Standalone Test):**
  ```bash
  python tests/test_pipeline_ensemble_full.py
  ```
  *Phím tắt: `[SPACE]` hoặc `[c]` để chụp và chạy eKYC, `[s]` để chụp lưu nhanh, `[q]` để thoát.*

* **Cách 2: Chạy hệ thống Full Server Microservice (:8000 & :3000):**
  - **Khởi động AI Server:** Nhấp đúp file `start_ai_server.bat` (hoặc chạy `python server_module/app.py`).
  - **Khởi động Node.js Gateway:** Nhấp đúp file `start_nodejs_receiver.bat`.
  - **Truy cập Giao diện Web UI:** Mở trình duyệt tại `http://localhost:8000` hoặc `http://localhost:3000`.

---

## ⚙️ Các Cấu Hình Tham Số Cốt Lõi (`server_module/config.py`)

| Tham số | Giá trị mặc định | Giải thích |
| :--- | :---: | :---|
| `ACTIVE_LIVENESS_TIMEOUT_SEC` | `10.0` (giây) | Thời gian tối đa hoàn thành thử thách Chớp mắt / Quay đầu |
| `HEAD_MOVEMENT_YAW_DEG` | `16.0` (độ) | Ngưỡng góc quay đầu sang trái / phải để ghi nhận PASS |
| `HEAD_MOVEMENT_PITCH_DEG` | `12.0` (độ) | Ngưỡng góc gật đầu ngửa / cúi |
| `STRICT_GLASSES_POLICY` | `True` | Bắt buộc tháo kính mắt và khẩu trang trước khi chụp |
| `ENSEMBLE_REAL_THRESHOLD` | `0.50` | Ngưỡng xác suất trung bình kết luận REAL |
| `STRICT_SPOOF_VETO` | `True` | Nếu bất kỳ model nào phát hiện SPOOF $\ge 70\%$ thì lập tức VETO |
| `CAMERA_WIDTH` / `HEIGHT` | `640` x `480` | Độ phân giải chuẩn xử lý video stream |

---

## 📄 Bản Quyền & Giấy Phép
Dự án được xây dựng và phát triển phục vụ mục đích nghiên cứu và triển khai giải pháp xác thực sinh trắc học eKYC. Mọi đóng góp xin vui lòng tạo Issue hoặc Pull Request.
