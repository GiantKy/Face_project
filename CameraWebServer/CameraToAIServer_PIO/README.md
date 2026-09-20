# ESP32-S3 Camera to AI Server (PlatformIO Project)

Dự án firmware PlatformIO cho module **ESP32-S3-CAM** (WROOM N16R8: 16MB Flash + 8MB OPI PSRAM), cung cấp:
- **MJPEG Live Stream** độ trễ cực thấp ở độ phân giải **240x240** (`FRAMESIZE_240X240`) tại cổng `81` (`/stream`).
- **Web UI điều khiển eKYC Liveness đa bước** tại cổng `80` (`/`).
- Tích hợp trực tiếp với FastAPI AI Server (`/api/v1/esp32/challenge/start` & `/api/v1/esp32/challenge/step`).
- Lưu IP AI Server trực tiếp vào bộ nhớ Flash NVS.

---

## 📁 Cấu trúc thư mục

```text
CameraToAIServer_PIO/
├── platformio.ini               # Cấu hình PlatformIO (flash 16MB, OPI PSRAM, baudrate)
├── boards/
│   └── esp32cam_s3_wroom_n16r8.json  # Định nghĩa phần cứng ESP32-S3 N16R8 chuẩn từ esp32cam-rtsp
├── include/
│   ├── board_config.h           # Chọn model camera (mặc định CAMERA_MODEL_ESP32S3_EYE)
│   ├── camera_pins.h            # Sơ đồ chân camera OV2640 / OV3660 / OV5640
│   └── web_ui.h                 # Toàn bộ giao diện HTML5/JS giao tiếp với AI Server
└── src/
    └── main.cpp                 # Mã nguồn C++ chính (Camera init, HTTP endpoints, MJPEG stream)
```

---

## 🚀 Cách nạp code bằng PlatformIO (VS Code)

1. **Mở dự án trong VS Code**:
   - Mở VS Code -> Menu **File** -> **Open Folder...** -> Chọn thư mục:
     `c:\Users\HP\Desktop\Face-Project\CameraWebServer\CameraToAIServer_PIO`
2. **Cài đặt tiện ích PlatformIO**:
   - Nếu chưa có, vào tab Extensions (Ctrl+Shift+X) tìm `PlatformIO IDE` và nhấn Install.
3. **Biên dịch & Nạp code**:
   - Cắm cáp ESP32-S3-CAM vào máy tính.
   - Nhấn icon **Build** (dấu tích `✔`) ở thanh trạng thái dưới cùng để biên dịch.
   - Nhấn icon **Upload** (mũi tên `➔`) để nạp vào ESP32-S3.
   - Nhấn icon **Serial Monitor** (hình phích cắm hoặc biểu tượng màn hình) với baud rate `115200` để xem địa chỉ IP được cấp.
