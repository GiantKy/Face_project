# 🧪 Face-Project Tests Directory - Sổ Tay Hướng Dẫn Kiểm Thử

Tài liệu chi tiết, phân tích chuyên sâu và bảng đối chiếu kết quả của toàn bộ hệ thống kiểm thử được lưu trữ tại:
👉 **[TEST_NOTES.md](../TEST_NOTES.md)**

---

## ⚡ Hướng Dẫn Chạy Nhanh (Windows Quickstart)

> [!TIP]
> Luôn sử dụng lệnh **`py`** thay vì `python` để đảm bảo hệ điều hành gọi đúng trình thông dịch Python 3.11 chính thức (nơi đã cài đặt đầy đủ OpenCV `cv2`, PyTorch `torch`, MediaPipe, Transformers).

```powershell
# Chạy Pipeline eKYC hoàn chỉnh với MobileNetV2 (Khuyên dùng)
py tests/test_pipeline_mobilenet.py

# Chạy Pipeline eKYC hoàn chỉnh với YOLOv8
py tests/test_pipeline_full.py

# Chạy Pipeline Ensemble đa mô hình MiniFASNet
py tests/test_pipeline_ensemble.py
```

---

## 📋 Danh Sách 15 Kịch Bản Kiểm Thử

| STT | Tên Script | Nhóm | Công Nghệ / Model | Mô Tả & Lệnh Chạy |
|:---:|:---|:---:|:---|:---|
| 1 | `test_face_detection.py` | Component | YOLO Face (`Face_Detection.pt`) | Bắt khung mặt người thời gian thực, đo FPS.<br>`py tests/test_face_detection.py` |
| 2 | `test_landmark_detection.py` | Component | MediaPipe Tasks (`face_landmarker.task`) | Trích xuất 478 điểm mốc khuôn mặt 3D.<br>`py tests/test_landmark_detection.py` |
| 3 | `test_pose_validation.py` | Component | MediaPipe + PnP Solver | Ước lượng 3 góc Euler: Yaw, Pitch, Roll.<br>`py tests/test_pose_validation.py` |
| 4 | `test_face_alignment_crop.py` | Component | OpenCV Affine Transform | Căn ngang trục 2 mắt và crop mặt chuẩn 224x224 / 80x80.<br>`py tests/test_face_alignment_crop.py` |
| 5 | `test_anti_spoof.py` | Anti-Spoof | YOLO (`Anti_Spoof_YOLO.pt`) | Nhận diện Real/Spoof trực tiếp qua Bounding Box.<br>`py tests/test_anti_spoof.py` |
| 6 | `test_anti_spoof_minifasnet.py` | Anti-Spoof | MiniFASNetV2 (`Anti_Spoof_minifasnet.pth`) | Phân loại Real/Spoof trên Face Crop 80x80.<br>`py tests/test_anti_spoof_minifasnet.py` |
| 7 | `test_anti_spoof_mobilenetv2.py` | Anti-Spoof | MobileNetV2 (`Model_MobilenetV2/`) | Mô hình Hugging Face Safetensors phân loại 224x224.<br>`py tests/test_anti_spoof_mobilenetv2.py` |
| 8 | `test_anti_spoof_official_ensemble.py` | Anti-Spoof | Ensemble MiniFASNet (V2 + V1SE) | Kết hợp 2 model tỷ lệ crop 2.7x và 4.0x.<br>`py tests/test_anti_spoof_official_ensemble.py` |
| 9 | `test_head_movement.py` | Liveness | Pose Angle Tracker | Thử thách tương tác: Quay trái/phải.<br>`py tests/test_head_movement.py` |
| 10 | `test_pipeline.py` | Pipeline v1 | YOLO Face + YOLO Anti-Spoof | Pipeline sơ khởi: Bắt mặt và phát hiện giả mạo.<br>`py tests/test_pipeline.py` |
| 11 | `test_pipeline_2.py` | Pipeline v2 | Detection + Align + Smoothing | Bổ sung xoay thẳng mắt và làm mượt điểm xác suất.<br>`py tests/test_pipeline_2.py` |
| 12 | `test_pipeline_3.py` | Pipeline v3 | Pipeline v2 + Blink Detection | Bổ sung bài kiểm tra chớp mắt tự nhiên qua EAR.<br>`py tests/test_pipeline_3.py` |
| 13 | `test_pipeline_ensemble.py` | Ensemble | Pipeline + MiniFASNet Ensemble | Tích hợp đồng thời 2 model MiniFASNet V2 & V1SE.<br>`py tests/test_pipeline_ensemble.py` |
| 14 | `test_pipeline_mobilenet.py` | **Full eKYC** | MobileNetV2 (Safetensors 224x224) | Quy trình eKYC đầy đủ (Capture -> AI -> Blink -> Head Move -> Report).<br>`py tests/test_pipeline_mobilenet.py` |
| 15 | `test_pipeline_full.py` | **Full eKYC** | YOLO Anti-Spoof (`Anti_Spoof_YOLO.pt`) | Toàn bộ chu trình eKYC chuẩn tích hợp YOLO.<br>`py tests/test_pipeline_full.py` |

---

## ⌨️ Phím Tắt Tương Tác Trong Giao Diện eKYC

Khi đang chạy `test_pipeline_mobilenet.py` hoặc `test_pipeline_full.py`:

| Phím Tắt | Ý Nghĩa |
|:---:|:---|
| <kbd>SPACE</kbd> / <kbd>C</kbd> | Chụp ảnh khuôn mặt và kích hoạt lộ trình eKYC tự động |
| <kbd>A</kbd> | Bật / Tắt chế độ **Auto-Capture** (tự động chụp khi tư thế mặt đạt chuẩn) |
| <kbd>S</kbd> | Chụp nhanh và lưu kết quả ngay lập tức (bỏ qua thử thách Liveness) |
| <kbd>R</kbd> | Đặt lại (Reset) hệ thống để bắt đầu phiên kiểm tra danh tính mới |
| <kbd>Q</kbd> / <kbd>ESC</kbd> | Thoát chương trình |

---

## 📂 Thư Mục Kết Quả Đầu Ra

Các file sau mỗi phiên test được lưu tự động trong:
- MobileNetV2 Pipeline: `output/pipeline_mobilenet/<ID>/`
  - `1_pipeline_result.jpg`: Ảnh chụp hiển thị Dashboard HUD đầy đủ
  - `1_pipeline_result_clean.jpg`: Ảnh chụp sạch chỉ kèm badge kết quả
  - `2_face_crop_224.jpg`: Ảnh khuôn mặt $224 \times 224$ đã chuẩn hóa
  - `3_aligned_full.jpg`: Toàn bộ khung hình sau khi xoay cân bằng mắt
  - `4_report.json`: Báo cáo chỉ số kỹ thuật chi tiết
  - `output/pipeline_mobilenet/batch_summary_mobilenet.csv`: Nhật ký tổng hợp
- Full Pipeline v4: `output/<ID>/` và `output/batch_summary_v4.csv`
