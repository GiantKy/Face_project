/**
 * ============================================================================
 * ESP32-S3-CAM TO FASTAPI AI SERVER & NODE.JS INTEGRATION (SINGLE-SHOT STATIC)
 * ============================================================================
 * 
 * Luồng hoạt động:
 * 1. ESP32-S3-CAM chụp ảnh JPEG tĩnh (VGA 640x480).
 * 2. Gửi dữ liệu ảnh dạng Binary Stream (image/jpeg) trực tiếp tới FastAPI AI Server:
 *    POST http://<AI_SERVER_IP>:8000/api/v1/esp32/verify
 * 3. FastAPI AI Server xử lý trọn gói:
 *    - YOLO Face Detection
 *    - 3D Pose Euler Angles
 *    - Ensemble Anti-Spoofing (YOLO_4 + RF-DETR Small)
 *    - Tự động đẩy kết quả Webhook sang Node.js Server (:3000).
 * 4. ESP32 nhận phản hồi JSON gọn nhẹ (approved: true/false, verdict: REAL/SPOOF).
 *    - Nếu approved: Nháy đèn Flash 2 lần, kích hoạt Relay mở cửa.
 *    - Nếu rejected: Nháy đèn Flash 4 lần báo từ chối.
 * 
 * ============================================================================
 */

#include "esp_camera.h"
#include "esp_http_server.h"
#include <WiFi.h>
#include <HTTPClient.h>
#include <WebServer.h>
#include <Preferences.h>
#include <ESPmDNS.h>
#include "board_config.h"
#include "web_ui.h"

// ============================================================================
// 1. CẤU HÌNH WIFI
// ============================================================================
const char *ssid = "68/14";
const char *password = "16102005";

// ============================================================================
// 2. CẤU HÌNH MÁY CHỦ AI
// ============================================================================
String ai_server_ip = "192.168.123.4";  // Mặc định IP máy tính chạy AI Server
int ai_server_port = 8000;
const char *ai_endpoint = "/api/v1/esp32/verify";
const char *device_id = "ESP32_S3_GATE_01";

// Web Server điều khiển trên cổng 80 & Stream Server trên cổng 81
WebServer server(80);
httpd_handle_t stream_httpd = NULL;
Preferences preferences;

bool isBusyProcessing = false;
bool cameraInitialized = false;

// ============================================================================
// CÁC HÀM TIỆN ÍCH HARDWARE
// ============================================================================
void blinkFlash(int times, int delayMs) {
#if defined(LED_GPIO_NUM) && (LED_GPIO_NUM >= 0)
  pinMode(LED_GPIO_NUM, OUTPUT);
  for (int i = 0; i < times; i++) {
    digitalWrite(LED_GPIO_NUM, HIGH);
    delay(delayMs);
    digitalWrite(LED_GPIO_NUM, LOW);
    delay(delayMs);
  }
#endif
}

camera_fb_t* capturePhotoSafe() {
  // Xả 1 frame đệm cũ để cảm biến đo sáng tươi mới theo ánh sáng hiện tại
  camera_fb_t *dummy = esp_camera_fb_get();
  if (dummy) {
    esp_camera_fb_return(dummy);
    delay(20);
  }
  camera_fb_t *fb = esp_camera_fb_get();
  if (!fb) {
    delay(40);
    fb = esp_camera_fb_get();
  }
  return fb;
}

// ============================================================================
// LIVE STREAM MJPEG SERVER (PORT 81) - ĐỘ PHÂN GIẢI 240x240 SIÊU MƯỢT
// ============================================================================
#define PART_BOUNDARY "123456789000000000000987654321"
static const char* _STREAM_CONTENT_TYPE = "multipart/x-mixed-replace;boundary=" PART_BOUNDARY;
static const char* _STREAM_BOUNDARY = "\r\n--" PART_BOUNDARY "\r\n";
static const char* _STREAM_PART = "Content-Type: image/jpeg\r\nContent-Length: %u\r\nX-Timestamp: %d.%06d\r\n\r\n";

static esp_err_t stream_handler(httpd_req_t *req) {
  camera_fb_t *fb = NULL;
  esp_err_t res = ESP_OK;
  char part_buf[128];

  res = httpd_resp_set_type(req, _STREAM_CONTENT_TYPE);
  if (res != ESP_OK) return res;

  httpd_resp_set_hdr(req, "Access-Control-Allow-Origin", "*");
  httpd_resp_set_hdr(req, "X-Framerate", "60");

  while (true) {
    fb = esp_camera_fb_get();
    if (!fb) {
      delay(15);
      continue;
    }
    res = httpd_resp_send_chunk(req, _STREAM_BOUNDARY, strlen(_STREAM_BOUNDARY));
    if (res == ESP_OK) {
      size_t hlen = snprintf(part_buf, sizeof(part_buf), _STREAM_PART, fb->len, fb->timestamp.tv_sec, fb->timestamp.tv_usec);
      res = httpd_resp_send_chunk(req, part_buf, hlen);
    }
    if (res == ESP_OK) {
      res = httpd_resp_send_chunk(req, (const char *)fb->buf, fb->len);
    }
    if (fb) {
      esp_camera_fb_return(fb);
      fb = NULL;
    }
    if (res != ESP_OK) {
      break;
    }
    yield();
  }
  return res;
}

void startStreamServer() {
  httpd_config_t config = HTTPD_DEFAULT_CONFIG();
  config.server_port = 81;
  config.ctrl_port = 32769;

  httpd_uri_t stream_uri = {
    .uri = "/stream",
    .method = HTTP_GET,
    .handler = stream_handler,
    .user_ctx = NULL
  };

  if (httpd_start(&stream_httpd, &config) == ESP_OK) {
    httpd_register_uri_handler(stream_httpd, &stream_uri);
    Serial.println("[STREAM] Live MJPEG Stream 240x240 san sang tai port 81: http://<IP>:81/stream");
  }
}

// ============================================================================
// GIAO DIỆN WEB ĐIỀU KHIỂN & CÁC ENDPOINT
// ============================================================================
void handleRoot() {
  String html = FPSTR(INDEX_HTML);
  html.replace("%AI_SERVER_IP%", ai_server_ip);
  server.send(200, "text/html", html);
}

// Endpoint trả về 1 ảnh JPEG xem trực tiếp
void handleCapture() {
  if (!cameraInitialized) {
    server.send(500, "text/plain", "Camera chua khoi tao");
    return;
  }
  camera_fb_t *fb = capturePhotoSafe();
  if (!fb) {
    server.send(500, "text/plain", "Capture Failed");
    return;
  }
  server.send_P(200, "image/jpeg", (const char *)fb->buf, fb->len);
  esp_camera_fb_return(fb);
}

// Endpoint mở cửa từ xa
void handleOpenDoor() {
  Serial.println("[RELAY] Lenh mo cua nhan duoc -> Mo Relay & nhay Flash");
  blinkFlash(2, 100);
  server.send(200, "application/json", "{\"status\":\"DOOR_OPENED\",\"approved\":true}");
}

// Endpoint gửi ảnh tĩnh sang AI Server
void handleSendToAI() {
  if (isBusyProcessing) {
    server.send(429, "application/json", "{\"error\":\"Thiet bi dang ban\"}");
    return;
  }
  isBusyProcessing = true;

  camera_fb_t *fb = capturePhotoSafe();
  if (!fb) {
    isBusyProcessing = false;
    server.send(500, "application/json", "{\"error\":\"Khong the chup anh tu camera\"}");
    return;
  }

  String fullUrl = "http://" + ai_server_ip + ":" + String(ai_server_port) + String(ai_endpoint);
  Serial.printf("\n[ESP32] Dang gui anh (%u bytes) toi AI Server: %s ...\n", fb->len, fullUrl.c_str());

  HTTPClient http;
  http.begin(fullUrl);
  http.addHeader("Content-Type", "image/jpeg");
  http.addHeader("X-Device-ID", device_id);
  http.setTimeout(25000);

  int httpCode = http.POST(fb->buf, fb->len);
  String responsePayload = http.getString();
  http.end();

  esp_camera_fb_return(fb);
  isBusyProcessing = false;

  if (httpCode == 200) {
    Serial.println("[ESP32] Phan hoi tu AI Server:");
    Serial.println(responsePayload);

    if (responsePayload.indexOf("\"approved\":true") >= 0) {
      Serial.println(">>> [eKYC] PASS! Xac thuc thanh cong (REAL).");
      blinkFlash(2, 100);
    } else {
      Serial.println(">>> [eKYC] REJECT! Tu choi xac thuc.");
      blinkFlash(4, 50);
    }
    server.send(200, "application/json", responsePayload);
  } else {
    Serial.printf("[HTTP ERROR] AI Server tra ve ma loi: %d\n", httpCode);
    String errJson = "{\"approved\":false,\"verdict\":\"ERROR\",\"message\":\"Loi ket noi AI Server (HTTP " + String(httpCode) + ")\"}";
    server.send(200, "application/json", errJson);
  }
}

// Endpoint Thử Thách Bước 1: Khởi tạo phiên liveness đa bước
void handleChallengeStart() {
  if (isBusyProcessing) {
    server.send(429, "application/json", "{\"error\":\"Thiet bi dang ban\"}");
    return;
  }
  isBusyProcessing = true;

  camera_fb_t *fb = capturePhotoSafe();
  if (!fb) {
    isBusyProcessing = false;
    server.send(500, "application/json", "{\"error\":\"Khong the chup anh tu camera\"}");
    return;
  }

  String fullUrl = "http://" + ai_server_ip + ":" + String(ai_server_port) + "/api/v1/esp32/challenge/start";
  Serial.printf("\n[ESP32] Gui anh Khoi tao thu thach toi: %s ...\n", fullUrl.c_str());

  HTTPClient http;
  http.begin(fullUrl);
  http.addHeader("Content-Type", "image/jpeg");
  http.addHeader("X-Device-ID", device_id);
  http.setTimeout(25000);

  int httpCode = http.POST(fb->buf, fb->len);
  String responsePayload = http.getString();
  http.end();

  esp_camera_fb_return(fb);
  isBusyProcessing = false;

  Serial.printf("[ESP32] Challenge Start Response (%d): %s\n", httpCode, responsePayload.c_str());
  if (httpCode <= 0 || responsePayload.length() == 0) {
    String errJson = "{\"success\":false,\"passed\":false,\"message\":\"Không thể kết nối tới AI Server tại " + ai_server_ip + ":" + String(ai_server_port) + " (Lỗi " + String(httpCode) + "). Vui lòng đảm bảo start_ai_server.bat đang chạy!\"}";
    server.send(200, "application/json", errJson);
  } else {
    server.send(httpCode == 200 ? 200 : 400, "application/json", responsePayload);
  }
}

// Endpoint Thử Thách Bước 2 & 3: Gửi ảnh chớp mắt / quay đầu
void handleChallengeStep() {
  if (isBusyProcessing) {
    server.send(429, "application/json", "{\"error\":\"Thiet bi dang ban\"}");
    return;
  }

  String sessionId = server.hasArg("session_id") ? server.arg("session_id") : "";
  String stepName = server.hasArg("step") ? server.arg("step") : "eye_blink";

  if (sessionId.length() == 0) {
    server.send(400, "application/json", "{\"error\":\"Missing session_id\"}");
    return;
  }

  isBusyProcessing = true;
  camera_fb_t *fb = capturePhotoSafe();
  if (!fb) {
    isBusyProcessing = false;
    server.send(500, "application/json", "{\"error\":\"Khong the chup anh tu camera\"}");
    return;
  }

  String fullUrl = "http://" + ai_server_ip + ":" + String(ai_server_port) + "/api/v1/esp32/challenge/step";
  Serial.printf("\n[ESP32] Gui anh Buoc [%s] toi: %s ...\n", stepName.c_str(), fullUrl.c_str());

  HTTPClient http;
  http.begin(fullUrl);
  http.addHeader("Content-Type", "image/jpeg");
  http.addHeader("X-Session-ID", sessionId);
  http.addHeader("X-Step", stepName);
  http.addHeader("X-Device-ID", device_id);
  http.setTimeout(25000);

  int httpCode = http.POST(fb->buf, fb->len);
  String responsePayload = http.getString();
  http.end();

  esp_camera_fb_return(fb);
  isBusyProcessing = false;

  if (httpCode == 200 && responsePayload.indexOf("\"approved\":true") >= 0) {
    Serial.println(">>> [eKYC MULTI-STAGE] APPROVED! Mo cua thanh cong!");
    blinkFlash(2, 100);
  }

  Serial.printf("[ESP32] Challenge Step Response (%d): %s\n", httpCode, responsePayload.c_str());
  if (httpCode <= 0 || responsePayload.length() == 0) {
    String errJson = "{\"success\":false,\"passed\":false,\"approved\":false,\"message\":\"Không thể kết nối tới AI Server tại " + ai_server_ip + ":" + String(ai_server_port) + " (Lỗi " + String(httpCode) + "). Vui lòng đảm bảo start_ai_server.bat đang chạy!\"}";
    server.send(200, "application/json", errJson);
  } else {
    server.send(httpCode == 200 ? 200 : 400, "application/json", responsePayload);
  }
}

// Lưu IP AI Server vào Flash NVS
void handleSetAiIp() {
  if (server.hasArg("ip")) {
    ai_server_ip = server.arg("ip");
    ai_server_ip.trim();
    preferences.begin("camera_ai", false);
    preferences.putString("ai_ip", ai_server_ip);
    preferences.end();
    Serial.printf("[NVS] Da cap nhat IP AI Server moi: %s\n", ai_server_ip.c_str());
    server.send(200, "text/plain", "OK");
  } else {
    server.send(400, "text/plain", "Missing ip arg");
  }
}

// ============================================================================
// KHỞI TẠO CAMERA SENSOR (ĐỘ PHÂN GIẢI 240x240 SIÊU MƯỢT)
// ============================================================================
bool initCamera() {
  camera_config_t config;
  config.ledc_channel = LEDC_CHANNEL_0;
  config.ledc_timer   = LEDC_TIMER_0;
  config.pin_d0       = Y2_GPIO_NUM;
  config.pin_d1       = Y3_GPIO_NUM;
  config.pin_d2       = Y4_GPIO_NUM;
  config.pin_d3       = Y5_GPIO_NUM;
  config.pin_d4       = Y6_GPIO_NUM;
  config.pin_d5       = Y7_GPIO_NUM;
  config.pin_d6       = Y8_GPIO_NUM;
  config.pin_d7       = Y9_GPIO_NUM;
  config.pin_xclk     = XCLK_GPIO_NUM;
  config.pin_pclk     = PCLK_GPIO_NUM;
  config.pin_vsync    = VSYNC_GPIO_NUM;
  config.pin_href     = HREF_GPIO_NUM;
  config.pin_sccb_sda = SIOD_GPIO_NUM;
  config.pin_sccb_scl = SIOC_GPIO_NUM;
  config.pin_pwdn     = PWDN_GPIO_NUM;
  config.pin_reset    = RESET_GPIO_NUM;
  config.xclk_freq_hz = 20000000;
  config.pixel_format = PIXFORMAT_JPEG;
  config.grab_mode    = CAMERA_GRAB_LATEST;

  if (psramFound()) {
    config.frame_size   = FRAMESIZE_VGA;  // 640x480 VGA độ phân giải cao
    config.jpeg_quality = 10;
    config.fb_count     = 2;
    config.fb_location  = CAMERA_FB_IN_PSRAM;
  } else {
    config.frame_size   = FRAMESIZE_VGA;
    config.jpeg_quality = 12;
    config.fb_count     = 1;
    config.fb_location  = CAMERA_FB_IN_DRAM;
  }

  esp_err_t err = esp_camera_init(&config);
  if (err != ESP_OK) {
    Serial.printf("\n[ERROR] Khoi tao camera that bai! 0x%x\n", err);
    return false;
  }

  sensor_t *s = esp_camera_sensor_get();
  if (s != NULL) {
    s->set_framesize(s, FRAMESIZE_VGA);    // 640x480 VGA chuẩn nét
    s->set_brightness(s, 2);               // Tăng sáng tối đa (+2) để thấy rõ mặt
    s->set_contrast(s, 0);                 // Để contrast = 0 (tránh bệt đen bóng mắt khi ngược sáng)
    s->set_saturation(s, 0);               // Màu tự nhiên
    s->set_sharpness(s, 2);                // Tăng nét chi tiết khuôn mặt
    
    // Tự động bù sáng & chống ngược sáng (Backlight Compensation):
    s->set_gainceiling(s, GAINCEILING_16X);// Tăng trần khuếch đại sáng lên 16X khi thiếu sáng
    s->set_exposure_ctrl(s, 1);            // Bật tự động phơi sáng (AEC)
    s->set_aec2(s, 1);                     // Bật thuật toán DSP AEC2 nâng cao
    s->set_ae_level(s, 2);                 // Bù phơi sáng mức cao nhất (+2) cứu sáng khuôn mặt
    s->set_gain_ctrl(s, 1);                // Bật tự động điều khiển Gain (AGC)
    s->set_bpc(s, 1);                      // Sửa điểm ảnh đen
    s->set_wpc(s, 1);                      // Sửa điểm ảnh trắng
    s->set_lenc(s, 1);                     // Bật Lens Correction chống tối 4 góc
    s->set_whitebal(s, 1);                 // Cân bằng trắng tự động (AWB)
    s->set_awb_gain(s, 1);                 // Gain cân bằng trắng
#if defined(CAMERA_MODEL_ESP32S3_EYE)
    s->set_vflip(s, 1);
#endif
  }

  for (int i = 0; i < 4; i++) {
    camera_fb_t *fb = esp_camera_fb_get();
    if (fb) esp_camera_fb_return(fb);
    delay(50);
  }

  Serial.println("[ESP32-S3] Camera FRAMESIZE_VGA (640x480) da san sang!");
  return true;
}

// ============================================================================
// SETUP
// ============================================================================
void setup() {
  Serial.begin(115200);
  delay(1000);

  Serial.println("\n========================================================");
  Serial.println("  ESP32-S3 CAM: SNAPSHOT & MULTI-STAGE eKYC (640x480)");
  Serial.println("========================================================");

  preferences.begin("camera_ai", true);
  String savedIp = preferences.getString("ai_ip", "");
  preferences.end();
  if (savedIp.length() > 6) {
    ai_server_ip = savedIp;
    Serial.printf("[NVS] Da load IP AI Server tu Flash: %s\n", ai_server_ip.c_str());
  }

  cameraInitialized = initCamera();

  // Kết nối WiFi
  Serial.printf("[WIFI] Dang ket noi toi: %s ", ssid);
  WiFi.begin(ssid, password);
  int retry = 0;
  while (WiFi.status() != WL_CONNECTED && retry < 35) {
    delay(500);
    Serial.print(".");
    retry++;
  }

  if (WiFi.status() == WL_CONNECTED) {
    Serial.println("\n[WIFI] Ket noi thanh cong!");
    Serial.print("[WIFI] Web UI: http://");
    Serial.println(WiFi.localIP());

    if (MDNS.begin("esp32cam")) {
      Serial.println("[mDNS] Truy cap Web qua: http://esp32cam.local");
    }
  } else {
    Serial.println("\n[WIFI] Ket noi that bai! Vui long kiem tra SSID va Password.");
  }

  server.on("/", HTTP_GET, handleRoot);
  server.on("/capture", HTTP_GET, handleCapture);
  server.on("/send-to-ai", HTTP_GET, handleSendToAI);
  server.on("/challenge-start", HTTP_GET, handleChallengeStart);
  server.on("/challenge-step", HTTP_GET, handleChallengeStep);
  server.on("/open", HTTP_GET, handleOpenDoor);
  server.on("/set-ai-ip", HTTP_GET, handleSetAiIp);
  server.begin();
  Serial.println("[WEB] Mini Web Server da san sang tai cong 80.");

  // Không cần mở stream port 81 (để giải phóng tài nguyên và tránh xung đột camera)
  // startStreamServer();
}

// ============================================================================
// LOOP
// ============================================================================
void loop() {
  server.handleClient();
  delay(5);
}
