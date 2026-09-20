# KIẾN TRÚC PIPELINE XÁC THỰC 3 BƯỚC (3-STAGE E-KYC ARCHITECTURE)
> **Tài liệu chuẩn kiến trúc hệ thống E-KYC AI Server & ESP32-CAM Client (Mô hình Push-Image)**  
> *Được chuẩn hóa đồng bộ giữa `tests/test_pipeline_ensemble_full.py` và `server_module/esp32_challenge.py`*

---

## 1. TỔNG QUAN LUỒNG XỬ LÝ (PIPELINE OVERVIEW)

Hệ thống eKYC hoạt động theo mô hình tối ưu băng thông & tài nguyên phần cứng IoT (**Push-Image liên tục từ ESP32**):
- **Bước 1 (Snapshot AI):** Chụp 1 ẢNH TĨNH chất lượng cao để kiểm tra khuôn mặt và duyệt chống giả mạo chuyên sâu bằng cụm mô hình Ensemble (YOLO_4 + RF-DETR Transformer). Nếu SPOOF -> Fail-Fast dừng ngay.
- **Bước 2 & Bước 3 (Push-Image Thử thách động):** ESP32 chủ động chụp và gửi chuỗi frame liên tục (~200ms/frame) qua HTTP POST `/api/v1/esp32/challenge/step`:
  + **Bước 2 (Chớp mắt):** Theo dõi chu kỳ Mở -> Nhắm -> Mở (1 lần chớp tự nhiên).
  + **Bước 3 (Quay đầu):** Theo dõi góc quay đầu Delta Yaw theo hướng ngẫu nhiên (Trái/Phải).
- **Ưu điểm cốt lõi:**
  + ESP32 chỉ phục vụ 1 luồng video stream duy nhất trên cổng `:81` cho người dùng xem preview trên Web UI.
  + AI Server **KHÔNG** kéo stream MJPEG riêng qua mạng (`cv2.VideoCapture`), loại bỏ nguy cơ nghẽn mạng WiFi, tràn bộ nhớ threadpool và timeout kết nối.

```
                    [ KHỞI ĐẦU: NGƯỜI DÙNG TIẾP CẬN ]
                                   │
                                   ▼
 ┌───────────────────────────────────────────────────────────────────┐
 │ BƯỚC 1: ẢNH TĨNH (SNAPSHOT) - FACE DETECT & ENSEMBLE ANTI-SPOOF   │
 │ • Face Detect đa tầng (YOLO 0.28 -> CLAHE 0.22 -> MediaPipe)      │
 │ • Kiểm tra kích thước mặt & góc nhìn thẳng ban đầu (Yaw, Pitch)   │
 │ • Cụm Ensemble Anti-Spoof: YOLO_4 + RF-DETR Small                 │
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
                                │ BƯỚC 2: PUSH-IMAGE - EYE BLINK     │
                                │ • ESP32 gửi frame mỗi ~200ms       │
                                │ • EAR < 0.18 (nhắm), >= 0.22 (mở)  │
                                │ • Chu kỳ: Mở -> Nhắm -> Mở (1 lần) │
                                └─────────────────┬──────────────────┘
                                                  │ (Chớp mắt ĐẠT)
                                                  ▼
                                ┌────────────────────────────────────┐
                                │ BƯỚC 3: PUSH-IMAGE - HEAD MOVEMENT │
                                │ • Thử thách ngẫu nhiên TRÁI / PHẢI │
                                │ • Delta Yaw >= 4.5° (hoặc >= 7.5°) │
                                │ • Tích lũy >= 2 frame liên tiếp    │
                                └─────────────────┬──────────────────┘
                                                  │ (Quay đầu ĐẠT)
                                                  ▼
                                ┌────────────────────────────────────┐
                                │ KẾT QUẢ CUỐI CÙNG (FINAL DECISION) │
                                │ 1. Kích hoạt Relay mở cửa ESP32    │
                                │ 2. Bắn Webhook sang Node.js Backend│
                                │ 3. Dừng vòng lặp gửi ảnh           │
                                └────────────────────────────────────┘
```

---

## 2. CHI TIẾT TỪNG BƯỚC THỰC THI

### BƯỚC 1: Chụp Ảnh Tĩnh, Nhận Diện Mặt & Ensemble Anti-Spoofing (Snapshot AI)

- **Endpoint:** `POST /api/v1/esp32/challenge/start`
- **Đầu vào:** 1 ảnh tĩnh JPEG duy nhất chụp từ camera ESP32.
- **Các tác vụ xử lý:**
  1. **Face Detection 5 tầng:**
     - *Tầng 1:* YOLO Face Detection (`conf=0.28, min_size=20`).
     - *Tầng 2:* Tiền xử lý thích nghi sáng CLAHE (`conf=0.22`) nếu ảnh bị ngược sáng/tối.
     - *Tầng 3:* Hạ ngưỡng `conf=0.18`.
     - *Tầng 4 (MediaPipe Fallback):* Nếu YOLO bỏ sót (do góc nghiêng, tóc che, ánh sáng yếu), bộ nhận diện 468 landmarks BlazeFace tự động bắt khuôn mặt.
     - *Tầng 5 (Kiểm tra camera ngược):* Xoay 180°; nếu có mặt, cảnh báo `CAMERA_UPSIDE_DOWN`.
  2. **Kiểm tra Kích thước mặt:** Với ảnh $240 \times 240$, chiều cao khuôn mặt $\ge 35\text{px}$.
  3. **Kiểm tra Tư thế nhìn thẳng ban đầu:** $|\text{Yaw}| \le 25^\circ$, $|\text{Pitch}| \le 22^\circ$. Lưu `baseline_ear` và `baseline_pose`.
  4. **Cụm Mô hình Chống Giả Mạo Ensemble (Dual-Model):**
     - **Model 1:** `Anti_Spoof_YOLO_4.pt` (Trọng số 0.5).
     - **Model 2:** `RF-DETR Small Transformer` (Trọng số 0.5).
     - **Fusion:** Bounding Box IoU Matching + Soft-Voting + **Nguyên tắc Spoof Veto** (chỉ cần 1 model báo SPOOF trên 0.65 -> Veto SPOOF ngay).
  5. **Nguyên tắc Fail-Fast:**
     - Nếu **SPOOF** hoặc không tìm thấy mặt: Dừng ngay lập tức, trả về ảnh chụp viền đỏ/cam.
     - Nếu **REAL**: Khởi tạo `session_id`, sinh ngẫu nhiên hướng quay đầu Bước 3 (`TURN_LEFT` hoặc `TURN_RIGHT`).

---

### BƯỚC 2: Thử Thách Liveness Chớp Mắt (Push-Image)

- **Endpoint:** `POST /api/v1/esp32/challenge/step?session_id=...&step=eye_blink`
- **Đầu vào:** Chuỗi ảnh JPEG được ESP32 chủ động gửi liên tục (~200ms/frame).
- **Tiêu chuẩn tham chiếu (`test_pipeline_ensemble_full.py`):**
  - Trích xuất 468 MediaPipe Landmarks.
  - Tính toán $EAR$ (Eye Aspect Ratio):
    $$EAR = \frac{||p_{160} - p_{144}|| + ||p_{158} - p_{153}||}{2 \cdot ||p_{33} - p_{133}||}$$
  - Máy trạng thái:
    $$\text{MẮT MỞ } (EAR \ge 0.22) \longrightarrow \text{MẮT NHẮM } (EAR < 0.18 \text{ hoặc } \le 80\% \text{ baseline}) \longrightarrow \text{MẮT MỞ LẠI } (EAR \ge 0.22)$$
  - Khi hoàn thành $\ge 1$ lần chớp mắt: Trả về `passed: True, next_step: 'head_movement'`, chuyển sang Bước 3.

---

### BƯỚC 3: Thử Thách Liveness Quay Đầu Ngẫu Nhiên (Push-Image)

- **Endpoint:** `POST /api/v1/esp32/challenge/step?session_id=...&step=head_movement`
- **Đầu vào:** Chuỗi ảnh JPEG được ESP32 chủ động gửi liên tục (~200ms/frame).
- **Hành động yêu cầu:** Sinh ngẫu nhiên từ Bước 1 (`TURN_LEFT` hoặc `TURN_RIGHT`).
- **Tiêu chuẩn tham chiếu (`test_pipeline_ensemble_full.py`):**
  - Tính toán độ lệch góc: $\Delta \text{Yaw} = \text{Yaw}_{\text{hiện tại}} - \text{Yaw}_{\text{baseline}}$.
  - Điều kiện đạt:
    + `TURN_LEFT`: $\Delta \text{Yaw} \ge +4.5^\circ$ hoặc $\text{Yaw}_{\text{hiện tại}} \ge +7.5^\circ$.
    + `TURN_RIGHT`: $\Delta \text{Yaw} \le -4.5^\circ$ hoặc $\text{Yaw}_{\text{hiện tại}} \le -7.5^\circ$.
  - Tích lũy $\ge 2$ frame liên tiếp đạt chuẩn để xác nhận (`consecutive_turn_frames >= 2`).
  - **Lưu ý:** Không cần chạy lại Anti-Spoof ở bước này vì đã xác thực ở Bước 1.

---

### KẾT QUẢ CUỐI CÙNG: Ra Quyết Định & Tác Vụ Sau Xác Thực (Post-Verification)

Khi Bước 3 hoàn thành (`step == "completed"` và `approved == True`):
1. **Dừng vòng lặp:** Client Web UI / ESP32 dừng gửi ảnh `setInterval`.
2. **Kích hoạt Relay (Mở cửa):** Server tự động gửi HTTP GET tới `http://<esp32_ip>/open` kích hoạt relay trong 3 giây.
3. **Webhook Báo cáo:** Server tự động gửi webhook payload (kèm `crop_face_base64`, `confidence`, thời gian) tới Node.js Backend (`http://localhost:3000/api/ekyc/result`).

---

## 3. DANH MỤC CÁC API ENDPOINTS LIÊN QUAN

| Method | Endpoint | Mô tả | Đầu vào | Đầu ra chính |
| :--- | :--- | :--- | :--- | :--- |
| `POST` | `/api/v1/esp32/challenge/start` | **BƯỚC 1:** Tiếp nhận 1 ảnh snapshot, chạy Face Detect & Ensemble Anti-Spoof (Fail-Fast) | Binary JPEG / Multipart | `session_id`, `is_real`, `target_head_action`, `head_prompt`, `captured_image_base64` |
| `POST` | `/api/v1/esp32/challenge/step` | **BƯỚC 2 & 3:** ESP32 push từng frame kiểm tra chớp mắt (`eye_blink`) và quay đầu (`head_movement`). Tự kích hoạt relay + webhook khi hoàn thành | Binary JPEG / JSON `image_base64` | `passed`, `step`, `next_step`, `progress`, `approved`, `ear`, `delta` |
| `POST` | `/api/v1/esp32/challenge/reset` | Hủy phiên thử thách hiện tại khi timeout hoặc người dùng hủy | JSON `{ "session_id" }` | `success`, `message` |
| `POST` | `/api/v1/esp32/verify` | Chụp 1 ảnh duy nhất (Single-shot không qua thử thách động) | Binary JPEG / Multipart | `approved`, `is_real`, `captured_image_base64` |
| `GET` | `/api/v1/health` | Kiểm tra trạng thái sẵn sàng của các mô hình AI | None | `models_loaded: true`, `device: "cpu"/"cuda"` |

---

## 4. BẢNG ĐỒNG BỘ GIỮA CÁC MODULE CODE

| Khái niệm | File Client UI (`test_pipeline_ensemble_full.py`) | File AI Server (`esp32_challenge.py` + `app.py`) |
| :--- | :--- | :--- |
| **Bước 1** | `PipelineStage.PREVIEW_ALIGN` & `RUN_AI_STATIC` | `start_challenge(raw_frame, pipeline)` |
| **Bước 2** | `PipelineStage.LIVE_BLINK` (EAR mở -> nhắm -> mở) | `process_step(..., step_name="eye_blink")` |
| **Bước 3** | `PipelineStage.LIVE_HEAD_MOVEMENT` (quay đầu >= 2 frame) | `process_step(..., step_name="head_movement")` |
| **Fail-Fast** | `is_spoof_fail` $\rightarrow$ Dừng ngay | `not is_real` $\rightarrow$ Dừng ngay, trả ảnh viền đỏ |
| **Mở cửa** | `trigger_esp32_relay(esp32_ip)` | `trigger_esp32_relay(esp32_ip)` |
| **Báo cáo** | `send_nodejs_webhook(nodejs_url, payload)` | `dispatch_webhook_to_nodejs(payload)` |
