# 📋 SỔ TAY GHI CHÚ KIỂM THỬ (TEST NOTES & GUIDE)
> **Dự án:** Hệ thống eKYC Face ID - Anti-Spoofing & Liveness Detection Pipeline  
> **Cập nhật:** Tháng 09/2026

---

## 📑 MỤC LỤC
1. [Kho Lưu Trữ Models (Google Drive)](#1-kho-lưu-trữ-models-google-drive)
2. [Tổng Quan Luồng Hoạt Động & Cơ Chế Ensemble](#2-tổng-quan-luồng-hoạt-động--cơ-chế-ensemble)
3. [Bảng Tổng Hợp Nhanh Các Kịch Bản Test](#3-bảng-tổng-hợp-nhanh-các-kịch-bản-test)
4. [Chi Tiết Từng Bài Kiểm Thử (Test Cases)](#4-chi-tiết-từng-bài-kiểm-thử-test-cases)
   - [Nhóm A: Pipeline eKYC Hoàn Chỉnh (Khuyên dùng)](#nhóm-a-pipeline-ekyc-hoàn-chỉnh-khuyên-dùng)
   - [Nhóm B: Kiểm Thử Độc Lập Ensemble & Anti-Spoof](#nhóm-b-kiểm-thử-độc-lập-ensemble--anti-spoof)
   - [Nhóm C: Kiểm Thử Từng Thành Phần Đơn Lẻ (Component Tests)](#nhóm-c-kiểm-thử-từng-thành-phần-đơn-lẻ-component-tests)
5. [Quy Tắc Veto & Consensus Filtering Trong Thực Tế](#5-quy-tắc-veto--consensus-filtering-trong-thực-tế)
6. [Bảng Phím Tắt Điều Khiển (Hotkeys)](#6-bảng-phím-tắt-điều-khiển-hotkeys)
7. [Xử Lý Sự Cố Thường Gặp (Troubleshooting)](#7-xử-lý-sự-cố-thường-gặp-troubleshooting)

---

## 1. 📦 Kho Lưu Trữ Models (Google Drive)

Toàn bộ các tệp trọng số huấn luyện và model phục vụ kiểm thử được lưu trữ tại:
* 🔗 **Google Drive Repository:** [Google Drive - Face Project Models Folder](https://drive.google.com/drive/folders/1O7lqzhpJ8DE9x2AFzMyrd3M2-8sNdYBn)

### Bảng đối chiếu model sử dụng trong các bài test:
| File Model | Đường dẫn trong Project | Sử dụng trong file test |
|:---|:---|:---|
| **YOLO Face Detection** | `models/Face_Detection.pt` | `test_face_detection.py`, các pipeline |
| **MediaPipe Landmarker** | `models/face_landmarker.task` | `test_landmark_detection.py`, các pipeline |
| **YOLO_4 Face Anti-Spoof** | `models/Anti_Spoof_YOLO_4.pt` | `test_anti_spoof.py`, `test_pipeline_ensemble_full.py` |
| **RF-DETR Small ONNX** | `models/roboflow/**/weights.onnx` | `test_anti_spoof_rfdetr_small.py`, `test_pipeline_ensemble_full.py` |
| **MiniFASNetV2** | `models/Anti_Spoof_minifasnet.pth` | `test_anti_spoof_minifasnet.py` |
| **MobileNetV2** | `models/Model_MobilenetV2/model.safetensors` | `test_anti_spoof_mobilenetv2.py`, `test_pipeline_mobilenet.py` |

---

## 2. 🌐 Tổng Quan Luồng Hoạt Động & Cơ Chế Ensemble

Pipeline mới nhất (`tests/test_pipeline_ensemble_full.py`) hoạt động theo chu trình khép kín:

```mermaid
flowchart TD
    A[Webcam Preview / Khung Oval Định Vị] -->|Phím SPACE hoặc 's'| B[Chụp Ảnh Gốc -> data_raw/<id>.jpg]
    B --> C[1. Phát Hiện Mặt YOLO & Căn Chỉnh Hình Học]
    C --> D[2. Trích Xuất 478 Điểm Mốc & Pose 3D Yaw/Pitch/Roll]
    D --> E[3. Cắt Chuẩn Hóa BBox 224x224]
    E --> F[4. Chạy Đồng Thời 2 Model Anti-Spoof]
    F --> F1[Model 1: Anti_Spoof_YOLO_4]
    F --> F2[Model 2: RF-DETR Small Transformer]
    F1 & F2 --> G[Khớp Bounding Box: IoU >= 0.40]
    G -->|Chỉ 1 Bên Bắt Được Mặt| H1[⚠️ DISCARD: Lược Bỏ Ảnh (Thiếu Đồng Thuận)]
    G -->|Cả 2 Cùng Bắt Được Mặt| H2{Có Bên Nào Báo SPOOF >= 68%?}
    H2 -->|CÓ| I1[❌ Kích Hoạt Phủ Quyết VETO -> Chốt SPOOF]
    H2 -->|KHÔNG| I2[Soft-Voting Trung Bình Xác Suất Real]
    I2 -->|Real >= 50%| J[5. Active Liveness: Chớp Mắt EAR & Quay Đầu]
    I2 -->|Real < 50%| I1
    J -->|Đạt| K[🎯 eKYC APPROVED]
    J -->|Không Đạt| L[❌ eKYC REJECTED]
    K & I1 & L --> M[💾 Xuất 1_pipeline_result.jpg & 4_report.json]
```

---

## 3. 📊 Bảng Tổng Hợp Nhanh Các Kịch Bản Test

| STT | Tên File Test | Mục Đích | Model Sử Dụng | Thời Gian Ước Tính |
|:---:|:---|:---|:---|:---:|
| **1** | `test_pipeline_ensemble_full.py` | **Full eKYC Ensemble (Đầy đủ tính năng)** | `YOLO_4` + `RF-DETR Small` | Realtime 30 FPS + Snapshot |
| **2** | `test_ensemble_yolo_rfdetr.py` | Kiểm tra riêng bộ lọc OpenCV & Ensemble 2 model | `YOLO_4` + `RF-DETR Small` | ~150 - 250ms/ảnh |
| **3** | `test_anti_spoof_rfdetr_small.py` | Đánh giá độc lập RF-DETR Transformer | `weights.onnx` (Roboflow) | ~100 - 180ms/ảnh |
| **4** | `test_anti_spoof.py` | Đánh giá độc lập YOLO_4 Anti-Spoof | `Anti_Spoof_YOLO_4.pt` | ~15 - 25ms/frame |
| **5** | `test_pipeline_full.py` | Pipeline eKYC chạy riêng với YOLO | `Anti_Spoof_YOLO.pt` | Realtime + Snapshot |
| **6** | `test_pipeline_mobilenet.py` | Pipeline eKYC chạy riêng với MobileNetV2 | `model.safetensors` | Realtime + Snapshot |
| **7** | `test_face_detection.py` | Kiểm tra độ nhạy Bounding Box khuôn mặt | `Face_Detection.pt` | Realtime 30+ FPS |
| **8** | `test_pose_validation.py` | Kiểm tra góc quay 3D Euler (Yaw/Pitch/Roll) | MediaPipe Tasks API | Realtime |

---

## 4. 🔍 Chi Tiết Từng Bài Kiểm Thử (Test Cases)

### Nhóm A: Pipeline eKYC Hoàn Chỉnh (Khuyên Dùng)

#### 1. `test_pipeline_ensemble_full.py` (Kịch bản chính thức)
* **Mục tiêu:** Kiểm thử toàn bộ hệ thống eKYC tích hợp cả **YOLO_4** và **RF-DETR Small** với cơ chế Veto và Consensus Filtering.
* **Lệnh chạy đầy đủ (Interactive Capture + Active Liveness):**
  ```powershell
  py tests/test_pipeline_ensemble_full.py --cam 0
  ```
* **Lệnh chạy chế độ chụp nhanh (Chụp là có kết quả ngay, bỏ qua Liveness):**
  ```powershell
  py tests/test_pipeline_ensemble_full.py --cam 0 --static
  ```
* **Các bước kiểm tra cần quan sát:**
  1. **Giai đoạn Preview:** Khung Oval có làm mờ ngoại vi không? Đèn rọi có đủ sáng ($L \ge 55$) để mở khóa chụp không?
  2. **Bấm phím `SPACE` hoặc `s`:** Ảnh gốc có được lưu vào `data_raw/<id>.jpg` không?
  3. **Kết quả Ensemble:** Báo cáo in ra có hiển thị chi tiết điểm số của cả `YOLO_4` và `RF-DETR` không?
  4. **Thư mục đầu ra:** Kiểm tra `tests/output/<id>/` có đủ 4 tệp: `1_pipeline_result.jpg`, `2_face_crop_224.jpg`, `3_aligned_full.jpg`, `4_report.json`.

---

### Nhóm B: Kiểm Thử Độc Lập Ensemble & Anti-Spoof

#### 2. `test_ensemble_yolo_rfdetr.py`
* **Mục tiêu:** Kiểm thử độc lập logic ghép 2 model, thuật toán tính IoU matching, đo độ sắc nét Laplacian và độ sáng HSV.
* **Lệnh chạy kiểm thử Webcam:**
  ```powershell
  py tests/test_ensemble_yolo_rfdetr.py --cam 0
  ```
* **Lệnh chạy trên 1 ảnh mẫu:**
  ```powershell
  py tests/test_ensemble_yolo_rfdetr.py --image data_raw/0.jpg
  ```

#### 3. `test_anti_spoof_rfdetr_small.py`
* **Mục tiêu:** Kiểm tra độ chính xác và khả năng nhận diện vân màn hình/ảnh in của kiến trúc Transformer RF-DETR Small.
* **Lệnh chạy:**
  ```powershell
  py tests/test_anti_spoof_rfdetr_small.py --cam 0
  ```

#### 4. `test_anti_spoof.py`
* **Mục tiêu:** Kiểm tra tốc độ và độ nhạy của model YOLO_4 trên luồng Webcam liên tục.
* **Lệnh chạy:**
  ```powershell
  py tests/test_anti_spoof.py
  ```

---

### Nhóm C: Kiểm Thử Từng Thành Phần Đơn Lẻ (Component Tests)

#### 5. `test_face_detection.py`
* **Mục tiêu:** Đo đạc FPS và độ chính xác của Bounding Box (`Face_Detection.pt`).
* **Lệnh chạy:**
  ```powershell
  py tests/test_face_detection.py
  ```

#### 6. `test_pose_validation.py`
* **Mục tiêu:** Kiểm tra thuật toán giải PnP tính 3 góc Euler:
  * $|\text{Yaw}| \le 15^\circ$: Quay trái/phải.
  * $|\text{Pitch}| \le 15^\circ$: Ngước lên/cúi xuống.
  * $|\text{Roll}| \le 15^\circ$: Nghiêng đầu.
* **Lệnh chạy:**
  ```powershell
  py tests/test_pose_validation.py
  ```

---

## 5. ⚖️ Quy Tắc Veto & Consensus Filtering Trong Thực Tế

### Cơ chế 1: Strict Spoof Veto ($\ge 68\%$)
* **Ý nghĩa:** Trong bài toán bảo mật eKYC, sự an toàn là số 1. Thà từ chối 1 bức ảnh chất lượng kém để người dùng quét lại, còn hơn mở cửa cho kẻ gian lận dùng ảnh in/màn hình điện thoại.
* **Cách vận hành:**
  * Nếu **YOLO_4** hoặc **RF-DETR** khẳng định `SPOOF` với độ tin cậy $\ge 68\%$, hệ thống lập tức kích hoạt quyền **phủ quyết (Veto)**.
  * Điểm số của bên còn lại (dù là Real bao nhiêu %) sẽ bị **bỏ qua hoàn toàn**.
  * Kết quả chốt hạ là `SPOOF` với độ tự tin lấy theo điểm cao nhất của bên bắt được gian lận.

### Cơ chế 2: Consensus Filtering (Lược bỏ ảnh khi chỉ có 1 model nhận diện)
* **Ý nghĩa:** Khắc phục triệt để lỗi của các trường hợp như `ID: 108` (YOLO = N/A nhưng RF-DETR = Spoof 41.9%).
* **Cách vận hành:**
  * Khi **chỉ có 1 model bắt được mặt**, hệ thống coi đây là sai số góc chụp hoặc nhiễu camera $\rightarrow$ **Tự động lược bỏ ảnh (`DISCARD_NO_CONSENSUS`)**.
  * Không đưa ra phán đoán bừa, vẽ khung cảnh báo màu **Cam** và yêu cầu người dùng chụp lại ngay ngắn.

---

## 6. ⌨️ Bảng Phím Tắt Điều Khiển (Hotkeys)

| Phím | Tác Dụng |
|:---:|:---|
| **SPACE** hoặc **c** | Chụp ảnh và bắt đầu Full quy trình eKYC (AI + Active Liveness) |
| **s** | **Chụp nhanh & Lưu ngay** (Chạy Ensemble AI -> Xuất báo cáo không cần đợi Liveness) |
| **a** | Bật / Tắt chế độ tự động chụp khi mặt giữ yên chuẩn trong Oval 25 frames |
| **r** | Khởi tạo phiên eKYC mới (ảnh ID tiếp theo) |
| **q** hoặc **ESC** | Thoát ứng dụng an toàn |

---

## 7. 🛠️ Xử Lý Sự Cố Thường Gặp (Troubleshooting)

| Hiện Tượng | Nguyên Nhân | Cách Xử Lý |
|:---|:---|:---|
| Báo lỗi `DISCARD_NO_CONSENSUS` | Mặt quá nghiêng làm 1 trong 2 model không bắt được box | Nhìn thẳng vào giữa khung Oval và bấm chụp lại |
| Mặt thật bị báo `SPOOF (95%)` | Ánh sáng đèn LED chiếu thẳng gây lóa bóng trán hoặc camera bật chế độ làm mịn da | Điều chỉnh lại góc đèn, tắt bộ lọc làm đẹp trên Webcam |
| Báo `ANH SANG YEU (L < 55)` | Không gian phòng chụp quá tối, model không nhìn rõ vân da | Bật thêm đèn phòng hoặc di chuyển ra khu vực đủ sáng |
| Không mở được Camera | Camera đang bị ứng dụng khác chiếm giữ (Zoom, Teams, Chrome) | Tắt các ứng dụng đang dùng camera rồi chạy lại lệnh test |
