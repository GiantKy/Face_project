# 🧠 DANH MỤC QUẢN LÝ MÔ HÌNH AI (AI MODELS CATALOG)

Tài liệu này tổng hợp toàn bộ các mô hình Trí tuệ Nhân tạo (AI Models) trong dự án **Face-Project**, được phân loại và đặt tên chuẩn hóa theo chức năng nghiệp vụ, phân định rõ giữa **Bản chính thức (Official / Production)** và **Bản thử nghiệm (Experimental / Legacy)**.

## 📦 Link Tải Toàn Bộ Model (Google Drive)

* 🌐 **Link tải Toàn Bộ Mô Hình (`models/`):**  
  👉 [**Google Drive - Toàn Bộ Models Project**](https://drive.google.com/drive/folders/1zEIGdw3krTHwO8w5BmkCxSP0v6Wdw0LU?usp=drive_link)
* 🚀 **Link tải Mô Hình Server Module (`server_module/models/`):**  
  👉 [**Google Drive - Server Module Models**](https://drive.google.com/drive/folders/1w-Xcl0irJzlXPCzJuVRYWPepxnH-3oiX?usp=drive_link)

---

## 📁 1. Cấu Trúc Cây Thư Mục Quản Lý (Directory Structure)

```text
models/
├── face_detection/                                 # [TASK 1] Phát hiện khuôn mặt trong ảnh
│   └── yolo_face_detection_official.pt             # ⭐ Bản chính thức: YOLOv8 Face Detection (18.3 MB)
│
├── landmarks/                                      # [TASK 2] Định vị 478 tọa độ 3D Landmarks khuôn mặt
│   └── mediapipe_face_landmarker_official.task     # ⭐ Bản chính thức: MediaPipe 3D Landmarker (3.6 MB)
│
├── anti_spoof/                                     # [TASK 3] Chống giả mạo sinh trắc học (Anti-Spoofing)
│   ├── yolo/                                       # Nhánh mô hình CNN YOLO Object Detection
│   │   ├── yolo_anti_spoof_v4_official.pt          # ⭐ Bản chính thức: YOLO_4 CNN Liveness (6.0 MB)
│   │   ├── yolo_anti_spoof_v0_legacy.pt            # Bản gốc ban đầu (Anti_Spoof_YOLO.pt - 6.0 MB)
│   │   ├── yolo_anti_spoof_v1_exp.pt               # Bản thử nghiệm v1 (6.0 MB)
│   │   ├── yolo_anti_spoof_v2_medium_exp.pt        # Bản thử nghiệm v2 cỡ Medium (23.4 MB)
│   │   └── yolo_anti_spoof_v3_medium_exp.pt        # Bản thử nghiệm v3 cỡ Medium (23.4 MB)
│   │
│   ├── rf_detr/                                    # Nhánh mô hình Vision Transformer (RF-DETR)
│   │   ├── rfdetr_small_official/                  # ⭐ Bản chính thức: RF-DETR Small Ensemble (108.9 MB)
│   │   ├── rfdetr_nano_v1_exp/                     # Bản thử nghiệm Nano v1 (104.2 MB)
│   │   ├── rfdetr_nano_v2_exp/                     # Bản thử nghiệm Nano v2 (104.2 MB)
│   │   ├── rfdetr_nano_v3_exp/                     # Bản thử nghiệm Nano v3 (108.9 MB)
│   │   └── rfdetr_nano_v0_legacy/                  # Bản thử nghiệm Nano v0 (10.1 MB)
│   │
│   ├── minifasnet/                                 # Nhánh mô hình phụ trợ Silent-Face MiniFASNet
│   │   ├── minifasnet_v1se_exp.pth                 # Bản MiniFASNetV1SE 80x80 (1.8 MB)
│   │   ├── minifasnet_v2_exp.pth                   # Bản MiniFASNetV2 80x80 (1.8 MB)
│   │   └── minifasnet_legacy.pth                   # Bản MiniFASNet gốc (239 KB)
│   │
│   └── mobilenet/                                  # Nhánh mô hình đối chứng MobileNetV2
│       └── mobilenetv2_exp/                        # Trọng số Safetensors & Config (9.0 MB)
│
├── face_occlusion/                                 # [TASK 4] Kiểm tra vật che mặt (Kính mắt & Khẩu trang)
│   └── yolo26n_glass_and_mask_official/            # ⭐ Bản chính thức: YOLO26n Kính & Khẩu trang (9.3 MB)
│       ├── weights.onnx                            # Trọng số ONNX Runtime
│       ├── model_type.json
│       ├── environment.json
│       └── class_names.txt
│
└── roboflow/                                       # Thư mục Cache Offline Roboflow Inference SDK (Chuẩn Model ID)
    ├── k-thi-gia-s-workspace/                      # ⭐ Thư mục Cache Model 2 RF-DETR Small (108.9 MB)
    │   └── face-spoof-detection-liika-owgrl-1-rfdetr-small-t1/
    ├── glass-and-mask-q5de1/                       # ⭐ Thư mục Cache Kính & Khẩu trang (9.3 MB)
    │   └── 2/
    ├── face-spoof-detection-liika-qopyy/           # Cache RF-DETR Nano v0 gốc (10.1 MB)
    ├── rfdetr_nano_v1_exp/                         # Thử nghiệm RF-DETR Nano v1 (104.2 MB)
    ├── rfdetr_nano_v2_exp/                         # Thử nghiệm RF-DETR Nano v2 (104.2 MB)
    ├── rfdetr_small_v3_exp/                        # Thử nghiệm RF-DETR Small v3 (108.9 MB)
    ├── glass_and_mask_legacy/                      # Bản Mask_Glass cũ (9.3 MB)
    └── yolo_v1_legacy/                             # Bản YOLO_V1 cũ (10.1 MB)
```

---

## 📊 2. Bảng Đối Chiếu & Chi Tiết Các Model (Model Mapping Table)

| STT | Tác vụ (Task) | Tên quản lý mới | Tên file gốc | Định dạng | Dung lượng | Phân loại | Vai trò trong hệ thống |
|:---:|:---|:---|:---|:---:|:---:|:---:|:---|
| **1** | **Face Detection** | `face_detection/yolo_face_detection_official.pt` | `Face_Detection.pt` | PyTorch (`.pt`) | 18.3 MB | ⭐ **Official** | Phát hiện khuôn mặt trong khung hình, canh oval Bước 1. |
| **2** | **Landmarks** | `landmarks/mediapipe_face_landmarker_official.task` | `face_landmarker.task` | MediaPipe (`.task`) | 3.6 MB | ⭐ **Official** | Trích xuất 478 điểm 3D: đo tỷ lệ mở mắt EAR và góc quay đầu Yaw/Pitch/Roll. |
| **3** | **Anti-Spoof (Model 1)** | `anti_spoof/yolo/yolo_anti_spoof_v4_official.pt` | `Anti_Spoof_YOLO_4.pt` | PyTorch (`.pt`) | 6.0 MB | ⭐ **Official** | **Model 1 trong Ensemble**: CNN chuyên biệt phân biệt mặt thật vs màn hình, ảnh in. |
| **4** | **Anti-Spoof (Model 2)** | `anti_spoof/rf_detr/rfdetr_small_official/` | `roboflow/.../rfdetr-small-t1` | ONNX (`.onnx`) | 108.9 MB | ⭐ **Official** | **Model 2 trong Ensemble**: Vision Transformer đa tầng thẩm định chiều sâu. |
| **5** | **Anti-Occlusion** | `face_occlusion/yolo26n_glass_and_mask_official/` | `roboflow/.../glass-and-mask-q5de1/2` | ONNX (`.onnx`) | 9.3 MB | ⭐ **Official** | Phát hiện mắt kính và khẩu trang (Chính sách A - tháo kính trước/sau khi chụp). |
| 6 | Anti-Spoof (YOLO) | `anti_spoof/yolo/yolo_anti_spoof_v0_legacy.pt` | `Anti_Spoof_YOLO.pt` | PyTorch (`.pt`) | 6.0 MB | Legacy | Bản YOLO gốc ban đầu. |
| 7 | Anti-Spoof (YOLO) | `anti_spoof/yolo/yolo_anti_spoof_v1_exp.pt` | `Anti_Spoof_YOLO_1.pt` | PyTorch (`.pt`) | 6.0 MB | Experimental | Bản thử nghiệm v1. |
| 8 | Anti-Spoof (YOLO) | `anti_spoof/yolo/yolo_anti_spoof_v2_medium_exp.pt` | `Anti_Spoof_YOLO_2.pt` | PyTorch (`.pt`) | 23.4 MB | Experimental | Bản thử nghiệm v2 (backbone Medium). |
| 9 | Anti-Spoof (YOLO) | `anti_spoof/yolo/yolo_anti_spoof_v3_medium_exp.pt` | `Anti_Spoof_YOLO_3.pt` | PyTorch (`.pt`) | 23.4 MB | Experimental | Bản thử nghiệm v3 (backbone Medium). |
| 10 | Anti-Spoof (RF-DETR) | `anti_spoof/rf_detr/rfdetr_nano_v1_exp/` | `rfderV1/` | ONNX (`.onnx`) | 104.2 MB | Experimental | Bản thử nghiệm RF-DETR Nano v1. |
| 11 | Anti-Spoof (RF-DETR) | `anti_spoof/rf_detr/rfdetr_nano_v2_exp/` | `rfderV2/` | ONNX (`.onnx`) | 104.2 MB | Experimental | Bản thử nghiệm RF-DETR Nano v2. |
| 12 | Anti-Spoof (RF-DETR) | `anti_spoof/rf_detr/rfdetr_nano_v3_exp/` | `rfderV3/` | ONNX (`.onnx`) | 108.9 MB | Experimental | Bản thử nghiệm RF-DETR Nano v3. |
| 13 | Anti-Spoof (RF-DETR) | `anti_spoof/rf_detr/rfdetr_nano_v0_legacy/` | `face-spoof-detection-liika-qopyy/` | ONNX (`.onnx`) | 10.1 MB | Legacy | Bản thử nghiệm nhẹ 10MB. |
| 14 | Anti-Spoof (MiniFAS) | `anti_spoof/minifasnet/minifasnet_v1se_exp.pth` | `4_0_0_80x80_MiniFASNetV1SE.pth` | PyTorch (`.pth`) | 1.8 MB | Experimental | MiniFASNet V1SE đối chứng. |
| 15 | Anti-Spoof (MiniFAS) | `anti_spoof/minifasnet/minifasnet_v2_exp.pth` | `2.7_80x80_MiniFASNetV2.pth` | PyTorch (`.pth`) | 1.8 MB | Experimental | MiniFASNet V2 đối chứng. |
| 16 | Anti-Spoof (MiniFAS) | `anti_spoof/minifasnet/minifasnet_legacy.pth` | `Anti_Spoof_minifasnet.pth` | PyTorch (`.pth`) | 239 KB | Legacy | Bản MiniFASNet gốc siêu nhẹ. |
| 17 | Anti-Spoof (MobileNet) | `anti_spoof/mobilenet/mobilenetv2_exp/` | `Model_MobilenetV2/` | Safetensors | 9.0 MB | Experimental | Bản MobileNetV2 Transformer / HuggingFace. |

---

## 🔒 3. Cơ Chế Quản Lý & Tương Thích (Clean Architecture & Compatibility)

1. **Phân loại gọn gàng 100% trong thư mục con**:
   - Toàn bộ các file model rời rạc trước đây ở thư mục gốc `models/` và `server_module/models/` đã được chuyển hoàn toàn vào các thư mục con chuyên biệt.
   - Thư mục gốc `models/` hiện chỉ gồm các thư mục chức năng sạch sẽ: `face_detection/`, `landmarks/`, `anti_spoof/`, `face_occlusion/`, `roboflow/` và file `README.md`.
2. **Cấu hình tự động & cập nhật mã nguồn**:
   - Toàn bộ codebase (`server_module/`, `src/`, `tests/`) đã được cập nhật đường dẫn ưu tiên nạp từ các thư mục con mới.
   - `server_module/config.py` và các bộ detector đều được trang bị cơ chế tìm kiếm thông minh và đệ quy, đảm bảo không bao giờ bị lỗi `FileNotFoundError`.
3. **Roboflow Offline Cache**:
   - Thư mục `models/roboflow/` lưu trữ cache trực tiếp dưới dạng thư mục vật lý chuẩn theo định danh `model_id` (`k-thi-gia-s-workspace/...`, `glass-and-mask-q5de1/...`).
   - Roboflow SDK có cơ chế bảo mật nội bộ `_validate_model_type_cache_path` cấm đi qua Symbolic Link / Junction, do đó các thư mục này được giữ nguyên vẹn dạng thư mục thực để SDK tiếp tục vận hành 100% offline mượt mà không bị lỗi `ValueError: traverses a symbolic link`.
