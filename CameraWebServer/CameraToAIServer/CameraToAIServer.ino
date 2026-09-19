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
#include <WiFi.h>
#include <HTTPClient.h>
#include <WebServer.h>
#include <Preferences.h>
#include <ESPmDNS.h>
#include "board_config.h"

// ============================================================================
// 1. CẤU HÌNH WIFI
// ============================================================================
const char *ssid = "2D11";
const char *password = "vpkhoavtq92d11";

// ============================================================================
// 2. CẤU HÌNH MÁY CHỦ AI
// ============================================================================
String ai_server_ip = "192.168.0.100";  // Mặc định IP máy tính chạy AI Server
int ai_server_port = 8000;
const char *ai_endpoint = "/api/v1/esp32/verify";
const char *device_id = "ESP32_S3_GATE_01";

// Web Server mini trên cổng 80 của ESP32
WebServer server(80);
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
  camera_fb_t *fb = esp_camera_fb_get();
  if (!fb) {
    delay(60);
    fb = esp_camera_fb_get();
  }
  return fb;
}

// ============================================================================
// GIAO DIỆN WEB ĐIỀU KHIỂN CHỤP ẢNH TĨNH
// ============================================================================
void handleRoot() {
  String html = R"rawliteral(
<!DOCTYPE html>
<html lang="vi">
<head>
  <meta charset="UTF-8">
  <meta name="viewport" content="width=device-width, initial-scale=1.0">
  <title>ESP32-S3 Camera AI eKYC</title>
  <style>
    body { font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif; background: #0b0f19; color: #f1f5f9; text-align: center; margin: 0; padding: 16px; }
    .card { max-width: 520px; margin: 0 auto; background: #161e2e; padding: 24px; border-radius: 20px; border: 1px solid #1e293b; box-shadow: 0 15px 35px rgba(0,0,0,0.5); }
    h1 { font-size: 20px; color: #38bdf8; margin: 0 0 6px 0; }
    p.sub { color: #94a3b8; font-size: 13px; margin: 0 0 16px 0; }
    .ip-box { background: #0f172a; padding: 10px 14px; border-radius: 10px; margin-bottom: 16px; display: flex; align-items: center; justify-content: space-between; font-size: 13px; border: 1px solid #334155; }
    .ip-box input { background: #1e293b; border: 1px solid #475569; color: #fff; padding: 6px 10px; border-radius: 6px; width: 140px; font-family: monospace; font-size: 13px; }
    .ip-box button { background: #3b82f6; color: #fff; border: none; padding: 6px 12px; border-radius: 6px; cursor: pointer; font-size: 12px; font-weight: 600; }
    .img-box { width: 100%; border-radius: 14px; border: 2px solid #334155; overflow: hidden; background: #020617; min-height: 280px; display: flex; align-items: center; justify-content: center; margin-bottom: 16px; }
    .img-box img { width: 100%; height: auto; display: block; }
    .btn-main { background: linear-gradient(135deg, #0284c7, #2563eb); color: #fff; border: none; padding: 14px 20px; font-size: 16px; font-weight: 700; border-radius: 12px; cursor: pointer; width: 100%; box-shadow: 0 4px 15px rgba(37,99,235,0.4); transition: all 0.2s; }
    .btn-main:hover { transform: translateY(-1px); filter: brightness(1.1); }
    .btn-main:disabled { background: #475569; cursor: not-allowed; box-shadow: none; transform: none; }
    #resultBox { margin-top: 16px; padding: 14px; border-radius: 10px; font-size: 14px; font-weight: 600; line-height: 1.5; display: none; }
    .pass { background: rgba(16, 185, 129, 0.15); border: 1px solid #10b981; color: #34d399; }
    .fail { background: rgba(239, 68, 68, 0.15); border: 1px solid #ef4444; color: #f87171; }
    .load { background: rgba(59, 130, 246, 0.15); border: 1px solid #3b82f6; color: #60a5fa; }
    .node-link { margin-top: 14px; font-size: 12px; color: #64748b; }
    .node-link a { color: #38bdf8; text-decoration: none; }
  </style>
</head>
<body>
  <div class="card">
    <h1>📷 ESP32-S3 eKYC Camera</h1>
    <p class="sub">Chụp ảnh tĩnh & Xác thực chống giả mạo Ensemble AI</p>

    <div class="ip-box">
      <span>⚙️ Máy chủ AI:</span>
      <div>
        <input type="text" id="aiIp" value=")rawliteral" + ai_server_ip + R"rawliteral(">
        <button onclick="saveAiIp()">💾 Lưu</button>
      </div>
    </div>

    <div class="img-box">
      <img id="camImg" src="/capture" alt="Camera Preview">
    </div>

    <button id="snapBtn" class="btn-main" onclick="triggerVerify()">🚀 CHỤP & XÁC THỰC eKYC</button>
    <div id="resultBox"></div>

    <div class="node-link">
      🖥️ Kết quả được tự động đẩy sang Node.js Server: 
      <a href="http://localhost:3000" target="_blank">http://localhost:3000</a>
    </div>
  </div>

  <script>
    async function saveAiIp() {
      const ip = document.getElementById('aiIp').value.trim();
      if (!ip) return;
      await fetch('/set-ai-ip?ip=' + encodeURIComponent(ip));
      alert('Đã lưu IP Máy chủ AI: ' + ip);
    }

    async function triggerVerify() {
      const btn = document.getElementById('snapBtn');
      const box = document.getElementById('resultBox');
      const img = document.getElementById('camImg');

      btn.disabled = true;
      box.style.display = 'block';
      box.className = 'load';
      box.innerHTML = '⏳ Đang chụp ảnh và gửi tới FastAPI AI Server...';

      try {
        const res = await fetch('/send-to-ai');
        const data = await res.json();
        img.src = '/capture?t=' + Date.now();

        if (data.approved) {
          box.className = 'pass';
          box.innerHTML = `✅ <b>XÁC THỰC THÀNH CÔNG (REAL)</b><br>
                           Độ tin cậy: ${(data.confidence * 100).toFixed(1)}% | Thời gian: ${data.processing_time_ms}ms<br>
                           <i>🔓 Cửa đã mở & Dữ liệu đã lưu vào Node.js!</i>`;
        } else {
          box.className = 'fail';
          box.innerHTML = `❌ <b>TỪ CHỐI XÁC THỰC (${data.verdict})</b><br>
                           ${data.message || 'Không vượt qua bài kiểm tra'}<br>
                           <i>⚠️ Đã báo cáo sự kiện tới Node.js Dashboard!</i>`;
        }
      } catch (err) {
        box.className = 'fail';
        box.innerHTML = '❌ Lỗi kết nối: ' + err.message;
      } finally {
        btn.disabled = false;
      }
    }
  </script>
</body>
</html>
)rawliteral";
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
// KHỞI TẠO CAMERA SENSOR
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
    config.frame_size   = FRAMESIZE_UXGA;
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
    s->set_framesize(s, FRAMESIZE_VGA);  // Độ phân giải 640x480 chuẩn eKYC
    s->set_brightness(s, 1);
    s->set_contrast(s, 1);
#if defined(CAMERA_MODEL_ESP32S3_EYE)
    s->set_vflip(s, 1);
#endif
  }

  for (int i = 0; i < 2; i++) {
    camera_fb_t *fb = esp_camera_fb_get();
    if (fb) esp_camera_fb_return(fb);
    delay(50);
  }

  Serial.println("[ESP32-S3] Camera VGA 640x480 da san sang!");
  return true;
}

// ============================================================================
// SETUP
// ============================================================================
void setup() {
  Serial.begin(115200);
  delay(1000);

  Serial.println("\n========================================================");
  Serial.println("  ESP32-S3 CAM: STATIC SINGLE-SHOT eKYC SYSTEM");
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
    Serial.print("[WIFI] IP ESP32: http://");
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
  server.on("/set-ai-ip", HTTP_GET, handleSetAiIp);
  server.begin();
  Serial.println("[WEB] Mini Web Server da san sang tai cong 80.\n");
}

// ============================================================================
// LOOP
// ============================================================================
void loop() {
  server.handleClient();
  delay(5);
}
