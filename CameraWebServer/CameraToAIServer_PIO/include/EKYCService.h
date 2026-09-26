#ifndef EKYC_SERVICE_H
#define EKYC_SERVICE_H

#include <Arduino.h>
#include "esp_http_server.h"
#include <HTTPClient.h>
#include <Preferences.h>
#include "CameraManager.h"
#include "HardwareController.h"
#include "web_ui.h"

#define PART_BOUNDARY "123456789000000000000987654321"
static const char* _STREAM_CONTENT_TYPE = "multipart/x-mixed-replace;boundary=" PART_BOUNDARY;
static const char* _STREAM_BOUNDARY     = "\r\n--" PART_BOUNDARY "\r\n";
static const char* _STREAM_PART         = "Content-Type: image/jpeg\r\nContent-Length: %u\r\nX-Timestamp: %d.%06d\r\n\r\n";

/**
 * @class EKYCService
 * @brief Điều phối dịch vụ Web Server sử dụng esp_http_server chuẩn ESP-IDF:
 *        - Port 80: Phục vụ Web UI, REST API, Snapshot và luồng eKYC 3 bước với FastAPI AI Server.
 *        - Port 81: Dedicated MJPEG Live Streaming Server (tương tự chuẩn CameraWebServer).
 *        - FreeRTOS Mutex: Đảm bảo luồng stream và các tác vụ chụp ảnh AI không tranh chấp DMA.
 */
class EKYCService {
public:
    EKYCService(CameraManager& camera, HardwareController& hw, const String& defaultAiIp, int aiPort, const char* deviceId, int nodePort = 3000)
        : m_camera(camera),
          m_hw(hw),
          m_aiServerIp(defaultAiIp),
          m_aiServerPort(aiPort),
          m_nodePort(nodePort),
          m_deviceId(deviceId),
          m_isBusy(false),
          m_streamPusherEnabled(true),
          m_cameraHttpd(NULL),
          m_streamHttpd(NULL),
          m_cameraMutex(NULL),
          m_pusherTaskHandle(NULL) {}

    void begin() {
        // 1. Tạo Mutex bảo vệ truy cập camera an toàn giữa Stream Task và eKYC Tasks
        if (m_cameraMutex == NULL) {
            m_cameraMutex = xSemaphoreCreateMutex();
        }

        // 2. Tải IP AI Server đã lưu trong Flash NVS (nếu có)
        m_preferences.begin("camera_ai", true);
        String savedIp = m_preferences.getString("ai_ip", "");
        m_preferences.end();
        if (savedIp.length() > 6) {
            m_aiServerIp = savedIp;
            Serial.printf("[EKYCService] Da load IP Server tu Flash: %s\n", m_aiServerIp.c_str());
        }

        // 3. Khởi tạo Web Server chính trên Cổng 80 (API điều khiển Relay, LED, Trạng thái)
        httpd_config_t config = HTTPD_DEFAULT_CONFIG();
        config.max_uri_handlers = 16;
        config.server_port = 80;
        config.ctrl_port = 32768;

        Serial.printf("[EKYCService] Khoi dong Web Server tren cong: %d\n", config.server_port);
        if (httpd_start(&m_cameraHttpd, &config) == ESP_OK) {
            registerUri(m_cameraHttpd, "/",                HTTP_GET, rootHandler);
            registerUri(m_cameraHttpd, "/stream",          HTTP_GET, streamHandler);
            registerUri(m_cameraHttpd, "/capture",         HTTP_GET, captureHandler);
            registerUri(m_cameraHttpd, "/challenge-start", HTTP_GET, challengeStartHandler);
            registerUri(m_cameraHttpd, "/challenge-step",  HTTP_GET, challengeStepHandler);
            registerUri(m_cameraHttpd, "/open",            HTTP_GET, openDoorHandler);
            registerUri(m_cameraHttpd, "/set-ai-ip",       HTTP_GET, setAiIpHandler);
            registerUri(m_cameraHttpd, "/set-led",         HTTP_GET, setLedHandler);
            registerUri(m_cameraHttpd, "/set-camera",      HTTP_GET, setCameraHandler);
            registerUri(m_cameraHttpd, "/status",          HTTP_GET, statusHandler);
            Serial.println("[EKYCService] Web Server Cổng 80 da san sang.");
        } else {
            Serial.println("[EKYCService] LOI: Khong the khoi dong Web Server Cổng 80!");
        }

        // 4. Khởi tạo Dedicated MJPEG Stream Server trên Cổng 81 (chuẩn CameraWebServer Espressif)
        httpd_config_t stream_config = HTTPD_DEFAULT_CONFIG();
        stream_config.server_port = 81;
        stream_config.ctrl_port = 32769;
        stream_config.max_uri_handlers = 2;
        stream_config.stack_size = 8192;

        Serial.printf("[EKYCService] Khoi dong Dedicated Stream Server tren cong: %d\n", stream_config.server_port);
        if (httpd_start(&m_streamHttpd, &stream_config) == ESP_OK) {
            registerUri(m_streamHttpd, "/stream", HTTP_GET, streamHandler);
            Serial.println("[EKYCService] Stream Server Cổng 81 da san sang: /stream");
        } else {
            Serial.println("[EKYCService] LOI: Khong the khoi dong Stream Server Cổng 81!");
        }

        // 5. Khởi chạy FreeRTOS Task đẩy luồng Stream liên tục lên Node.js Server (:3000) trên Core 0
        xTaskCreatePinnedToCore(
            streamPusherTaskWrapper,
            "streamPusher",
            6144,
            this,
            2,
            &m_pusherTaskHandle,
            0 // Ghim vao Core 0 de khong anh huong toi Core 1 (WiFi/Loop)
        );
        Serial.printf("[EKYCService] Da khoi tao StreamPusher Task day len Node.js (%s:%d)\n", m_aiServerIp.c_str(), m_nodePort);
    }

    void handleClient() {
        vTaskDelay(pdMS_TO_TICKS(10));
    }

    String getAiServerIp() const {
        return m_aiServerIp;
    }

    void setStreamPusherEnabled(bool enabled) {
        m_streamPusherEnabled = enabled;
    }

private:
    void registerUri(httpd_handle_t server, const char* uri, httpd_method_t method, esp_err_t (*handler)(httpd_req_t *)) {
        httpd_uri_t uri_handler = {
            .uri       = uri,
            .method    = method,
            .handler   = handler,
            .user_ctx  = this
        };
        httpd_register_uri_handler(server, &uri_handler);
    }

    // =========================================================================
    // STATIC ROUTE HANDLERS WRAPPERS (Forwarding req->user_ctx to instance)
    // =========================================================================
    static esp_err_t rootHandler(httpd_req_t *req) {
        return ((EKYCService *)req->user_ctx)->handleRoot(req);
    }
    static esp_err_t captureHandler(httpd_req_t *req) {
        return ((EKYCService *)req->user_ctx)->handleCapture(req);
    }
    static esp_err_t challengeStartHandler(httpd_req_t *req) {
        return ((EKYCService *)req->user_ctx)->handleChallengeStart(req);
    }
    static esp_err_t challengeStepHandler(httpd_req_t *req) {
        return ((EKYCService *)req->user_ctx)->handleChallengeStep(req);
    }
    static esp_err_t openDoorHandler(httpd_req_t *req) {
        return ((EKYCService *)req->user_ctx)->handleOpenDoor(req);
    }
    static esp_err_t setAiIpHandler(httpd_req_t *req) {
        return ((EKYCService *)req->user_ctx)->handleSetAiIp(req);
    }
    static esp_err_t setLedHandler(httpd_req_t *req) {
        return ((EKYCService *)req->user_ctx)->handleSetLed(req);
    }
    static esp_err_t setCameraHandler(httpd_req_t *req) {
        return ((EKYCService *)req->user_ctx)->handleSetCamera(req);
    }
    static esp_err_t statusHandler(httpd_req_t *req) {
        return ((EKYCService *)req->user_ctx)->handleStatus(req);
    }
    static esp_err_t streamHandler(httpd_req_t *req) {
        return ((EKYCService *)req->user_ctx)->handleStream(req);
    }

    // =========================================================================
    // INSTANCE BUSINESS LOGIC
    // =========================================================================

    /**
     * @brief Dedicated MJPEG Streamer Handler (Cổng 81: /stream)
     *        Giữ kết nối TCP persistent, truyền liên tục frame nhị phân với 0 độ trễ.
     */
    esp_err_t handleStream(httpd_req_t *req) {
        camera_fb_t *fb = NULL;
        struct timeval _timestamp;
        esp_err_t res = ESP_OK;
        char part_buf[128];

        res = httpd_resp_set_type(req, _STREAM_CONTENT_TYPE);
        if (res != ESP_OK) return res;

        httpd_resp_set_hdr(req, "Access-Control-Allow-Origin", "*");
        httpd_resp_set_hdr(req, "X-Framerate", "60");
        httpd_resp_set_hdr(req, "Cache-Control", "no-cache, no-store, must-revalidate");
        httpd_resp_set_hdr(req, "Pragma", "no-cache");

        while (true) {
            // Khóa Mutex để đảm bảo an toàn nếu có tác vụ chụp ảnh AI khác đang gửi ảnh
            if (xSemaphoreTake(m_cameraMutex, pdMS_TO_TICKS(150)) == pdTRUE) {
                fb = esp_camera_fb_get();
                if (!fb) {
                    xSemaphoreGive(m_cameraMutex);
                    vTaskDelay(pdMS_TO_TICKS(10));
                    continue;
                }

                _timestamp.tv_sec = fb->timestamp.tv_sec;
                _timestamp.tv_usec = fb->timestamp.tv_usec;

                size_t fb_len = fb->len;
                uint8_t *fb_buf = fb->buf;

                res = httpd_resp_send_chunk(req, _STREAM_BOUNDARY, strlen(_STREAM_BOUNDARY));
                if (res == ESP_OK) {
                    size_t hlen = snprintf(part_buf, sizeof(part_buf), _STREAM_PART, fb_len, _timestamp.tv_sec, _timestamp.tv_usec);
                    res = httpd_resp_send_chunk(req, part_buf, hlen);
                }
                if (res == ESP_OK) {
                    res = httpd_resp_send_chunk(req, (const char *)fb_buf, fb_len);
                }

                esp_camera_fb_return(fb);
                fb = NULL;
                xSemaphoreGive(m_cameraMutex);

                if (res != ESP_OK) {
                    break; // Client đã đóng hoặc ngắt kết nối
                }
                vTaskDelay(pdMS_TO_TICKS(10)); // Nhường quyền truy cập camera cho streamPusherLoop
            } else {
                // Đang bận (ví dụ eKYC Step đang lấy frame), nhường CPU 15ms
                vTaskDelay(pdMS_TO_TICKS(15));
            }
        }
        return res;
    }

    static void streamPusherTaskWrapper(void *pvParameters) {
        ((EKYCService *)pvParameters)->streamPusherLoop();
    }

    /**
     * @brief Vòng lặp FreeRTOS Task đẩy luồng camera liên tục lên Node.js (:3000)
     *        SỬ DỤNG RAW TCP PERSISTENT SOCKET THAY VÌ HTTPClient:
     *        - Tắt Nagle Algorithm (setNoDelay) -> Gói tin đẩy tức thì, 0ms latency.
     *        - Bỏ qua overhead khởi tạo HTTPClient, không bị nghẽn readString() 1000ms.
     *        - Đạt tốc độ thực tế 20 - 30 FPS qua mạng WiFi.
     */
    void streamPusherLoop() {
        Serial.printf("[StreamPusher] Task khoi dong tren Core %d. Push toi: http://%s:%d/api/stream/frame\n",
                      xPortGetCoreID(), m_aiServerIp.c_str(), m_nodePort);

        WiFiClient client;
        client.setNoDelay(true);
        client.setTimeout(2);

        unsigned long lastPushLog = 0;
        int pushOkCount = 0;
        int pushErrCount = 0;
        char postHeader[256];

        while (true) {
            if (!m_streamPusherEnabled || WiFi.status() != WL_CONNECTED || !m_camera.isInitialized()) {
                if (client.connected()) client.stop();
                vTaskDelay(pdMS_TO_TICKS(500));
                continue;
            }

            // Duy trì kết nối TCP Keep-Alive luôn sẵn sàng
            if (!client.connected()) {
                if (!client.connect(m_aiServerIp.c_str(), m_nodePort)) {
                    pushErrCount++;
                    vTaskDelay(pdMS_TO_TICKS(300));
                    continue;
                }
                client.setNoDelay(true);
            }

            if (xSemaphoreTake(m_cameraMutex, pdMS_TO_TICKS(80)) == pdTRUE) {
                camera_fb_t *fb = m_camera.capturePhotoFast();
                if (fb) {
                    int hlen = snprintf(postHeader, sizeof(postHeader),
                        "POST /api/stream/frame HTTP/1.1\r\n"
                        "Host: %s:%d\r\n"
                        "Content-Type: image/jpeg\r\n"
                        "Content-Length: %u\r\n"
                        "X-ESP32-IP: %s\r\n"
                        "X-Device-ID: %s\r\n"
                        "Connection: keep-alive\r\n\r\n",
                        m_aiServerIp.c_str(), m_nodePort,
                        (unsigned int)fb->len,
                        WiFi.localIP().toString().c_str(),
                        m_deviceId
                    );

                    size_t w1 = client.write((const uint8_t*)postHeader, hlen);
                    size_t w2 = client.write(fb->buf, fb->len);
                    client.flush();
                    m_camera.returnFrameBuffer(fb);

                    if (w1 == (size_t)hlen && w2 > 0) {
                        pushOkCount++;
                        // Đọc nhanh phản hồi "OK" mà không bị nghẽn readString()
                        int waitMs = 0;
                        while (client.connected() && client.available() == 0 && waitMs < 25) {
                            vTaskDelay(pdMS_TO_TICKS(2));
                            waitMs += 2;
                        }
                        while (client.available()) {
                            client.read();
                        }
                    } else {
                        pushErrCount++;
                        client.stop();
                        vTaskDelay(pdMS_TO_TICKS(50));
                    }

                    if (millis() - lastPushLog >= 5000) {
                        float fps = (float)pushOkCount / 5.0f;
                        if (pushErrCount > 0 && pushOkCount == 0) {
                            Serial.printf("[StreamPusher] LOI ket noi Node.js toi %s:%d. Kiem tra start_nodejs_receiver.bat!\n",
                                          m_aiServerIp.c_str(), m_nodePort);
                        } else {
                            Serial.printf("[StreamPusher] Push frame OK -> Node.js: %d frames/5s (%.1f FPS)\n",
                                          pushOkCount, fps);
                        }
                        pushOkCount = 0;
                        pushErrCount = 0;
                        lastPushLog = millis();
                    }
                }
                xSemaphoreGive(m_cameraMutex);
            }

            vTaskDelay(pdMS_TO_TICKS(5));
        }
    }

    esp_err_t handleRoot(httpd_req_t *req) {
        String redirectHtml = "<!DOCTYPE html><html><head><meta charset=\"UTF-8\">"
            "<meta http-equiv=\"refresh\" content=\"1; url=http://" + m_aiServerIp + ":" + String(m_nodePort) + "/\">"
            "<title>ESP32 eKYC Node</title>"
            "<style>body{background:#0b0f19;color:#38bdf8;font-family:sans-serif;text-align:center;padding:50px;}</style></head>"
            "<body><h2>📷 ESP32-CAM eKYC Node</h2>"
            "<p>Đang chuyển hướng tới Node.js Web Dashboard...</p>"
            "<p><a href=\"http://" + m_aiServerIp + ":" + String(m_nodePort) + "/\" style=\"color:#fff;background:#2563eb;padding:10px 20px;border-radius:8px;text-decoration:none;\">Vào Web Dashboard (:3000)</a></p>"
            "</body></html>";
        httpd_resp_set_type(req, "text/html");
        httpd_resp_set_hdr(req, "Access-Control-Allow-Origin", "*");
        return httpd_resp_send(req, redirectHtml.c_str(), redirectHtml.length());
    }

    esp_err_t handleCapture(httpd_req_t *req) {
        if (!m_camera.isInitialized()) {
            httpd_resp_send_500(req);
            return ESP_FAIL;
        }

        camera_fb_t *fb = NULL;
        if (xSemaphoreTake(m_cameraMutex, pdMS_TO_TICKS(1000)) == pdTRUE) {
            fb = m_camera.capturePhotoSafe();
            if (!fb) {
                xSemaphoreGive(m_cameraMutex);
                httpd_resp_send_500(req);
                return ESP_FAIL;
            }

            httpd_resp_set_type(req, "image/jpeg");
            httpd_resp_set_hdr(req, "Content-Disposition", "inline; filename=capture.jpg");
            httpd_resp_set_hdr(req, "Access-Control-Allow-Origin", "*");
            esp_err_t res = httpd_resp_send(req, (const char *)fb->buf, fb->len);

            m_camera.returnFrameBuffer(fb);
            xSemaphoreGive(m_cameraMutex);
            return res;
        } else {
            httpd_resp_send_500(req);
            return ESP_FAIL;
        }
    }

    esp_err_t handleOpenDoor(httpd_req_t *req) {
        m_hw.openDoor();
        return sendJSON(req, 200, "{\"status\":\"DOOR_OPENED\",\"approved\":true}");
    }

    esp_err_t handleChallengeStart(httpd_req_t *req) {
        if (m_isBusy) {
            return sendJSON(req, 429, "{\"error\":\"Thiet bi dang ban\"}");
        }
        m_isBusy = true;

        // Báo hiệu LED RGB: Màu Tím (Bắt đầu Bước 1: Face Detect & Anti-Spoofing)
        m_hw.setStageIndicator("stage1");

        camera_fb_t *fb = NULL;
        if (xSemaphoreTake(m_cameraMutex, pdMS_TO_TICKS(2000)) != pdTRUE) {
            m_isBusy = false;
            m_hw.setStageIndicator("rejected");
            return sendJSON(req, 500, "{\"error\":\"Camera dang ban, khong the chup anh\"}");
        }

        // Bước 1: Tạm thời nâng lên VGA 640x480 sắc nét cho Anti-spoofing
        m_camera.setResolution(FRAMESIZE_VGA, 10);
        fb = m_camera.capturePhotoSafe();

        // Duy trì độ phân giải VGA 640x480 với Quality 20 theo chuẩn hệ thống
        m_camera.setResolution(FRAMESIZE_VGA, 20);

        if (!fb) {
            xSemaphoreGive(m_cameraMutex);
            m_isBusy = false;
            m_hw.setStageIndicator("rejected");
            return sendJSON(req, 500, "{\"error\":\"Khong the chup anh tu camera\"}");
        }

        String fullUrl = "http://" + m_aiServerIp + ":" + String(m_aiServerPort) + "/api/v1/esp32/challenge/start";
        Serial.printf("\n[EKYCService] Gui anh Khoi tao thu thach (VGA 640x480, %u bytes) toi: %s ...\n", fb->len, fullUrl.c_str());

        HTTPClient http;
        http.begin(fullUrl);
        http.addHeader("Content-Type", "image/jpeg");
        http.addHeader("X-Device-ID", m_deviceId);
        http.setTimeout(25000);

        int httpCode = http.POST(fb->buf, fb->len);
        String responsePayload = http.getString();
        http.end();

        m_camera.returnFrameBuffer(fb);
        xSemaphoreGive(m_cameraMutex);
        m_isBusy = false;

        Serial.printf("[EKYCService] Challenge Start Response (%d): %s\n", httpCode, responsePayload.c_str());
        if (httpCode <= 0 || responsePayload.length() == 0) {
            m_hw.setStageIndicator("rejected");
            String errJson = "{\"success\":false,\"passed\":false,\"message\":\"Không thể kết nối tới AI Server tại " + m_aiServerIp + ":" + String(m_aiServerPort) + " (Lỗi " + String(httpCode) + "). Vui lòng đảm bảo start_ai_server.bat đang chạy!\"}";
            return sendJSON(req, 200, errJson);
        } else {
            if (responsePayload.indexOf("\"success\":true") >= 0 && responsePayload.indexOf("\"is_real\":true") >= 0) {
                m_hw.setStageIndicator("stage2_blink");
            } else {
                m_hw.setStageIndicator("rejected");
            }
            return sendJSON(req, httpCode == 200 ? 200 : 400, responsePayload);
        }
    }

    esp_err_t handleChallengeStep(httpd_req_t *req) {
        if (m_isBusy) {
            return sendJSON(req, 429, "{\"error\":\"Thiet bi dang ban\"}");
        }

        String sessionId = "";
        String stepName = "eye_blink";
        getQueryParam(req, "session_id", sessionId);
        getQueryParam(req, "step", stepName);

        if (sessionId.length() == 0) {
            return sendJSON(req, 400, "{\"error\":\"Missing session_id\"}");
        }

        m_isBusy = true;

        if (stepName == "head_movement") {
            m_hw.setStageIndicator("stage3_turn");
        } else {
            m_hw.setStageIndicator("stage2_blink");
        }

        camera_fb_t *fb = NULL;
        if (xSemaphoreTake(m_cameraMutex, pdMS_TO_TICKS(1500)) != pdTRUE) {
            m_isBusy = false;
            return sendJSON(req, 500, "{\"error\":\"Camera dang ban, khong the lay frame\"}");
        }

        // Chụp nhanh frame hiện tại (độ phân giải giữ nguyên phù hợp cho stream cao tốc)
        fb = m_camera.capturePhotoFast();
        if (!fb) {
            xSemaphoreGive(m_cameraMutex);
            m_isBusy = false;
            return sendJSON(req, 500, "{\"error\":\"Khong the chup anh tu camera\"}");
        }

        String fullUrl = "http://" + m_aiServerIp + ":" + String(m_aiServerPort) + "/api/v1/esp32/challenge/step";
        Serial.printf("\n[EKYCService] Gui anh Buoc [%s] (%u bytes) toi: %s ...\n", stepName.c_str(), fb->len, fullUrl.c_str());

        HTTPClient http;
        http.begin(fullUrl);
        http.addHeader("Content-Type", "image/jpeg");
        http.addHeader("X-Session-ID", sessionId);
        http.addHeader("X-Step", stepName);
        http.addHeader("X-Device-ID", m_deviceId);
        http.setTimeout(25000);

        int httpCode = http.POST(fb->buf, fb->len);
        String responsePayload = http.getString();
        http.end();

        m_camera.returnFrameBuffer(fb);
        xSemaphoreGive(m_cameraMutex);
        m_isBusy = false;

        if (httpCode == 200 && responsePayload.indexOf("\"approved\":true") >= 0) {
            Serial.println(">>> [eKYC MULTI-STAGE] APPROVED! Mo cua thanh cong!");
            m_hw.openDoor();
        } else if (responsePayload.indexOf("\"passed\":true") >= 0 && responsePayload.indexOf("\"next_step\":\"head_movement\"") >= 0) {
            m_hw.setStageIndicator("stage3_turn");
        }

        Serial.printf("[EKYCService] Challenge Step Response (%d): %s\n", httpCode, responsePayload.c_str());
        if (httpCode <= 0 || responsePayload.length() == 0) {
            m_hw.setStageIndicator("rejected");
            String errJson = "{\"success\":false,\"passed\":false,\"approved\":false,\"message\":\"Không thể kết nối tới AI Server tại " + m_aiServerIp + ":" + String(m_aiServerPort) + " (Lỗi " + String(httpCode) + "). Vui lòng đảm bảo start_ai_server.bat đang chạy!\"}";
            return sendJSON(req, 200, errJson);
        } else {
            return sendJSON(req, httpCode == 200 ? 200 : 400, responsePayload);
        }
    }

    esp_err_t handleSetAiIp(httpd_req_t *req) {
        String newIp = "";
        if (getQueryParam(req, "ip", newIp) && newIp.length() > 6) {
            newIp.trim();
            m_aiServerIp = newIp;
            m_preferences.begin("camera_ai", false);
            m_preferences.putString("ai_ip", m_aiServerIp);
            m_preferences.end();
            Serial.printf("[EKYCService] Da cap nhat IP AI Server moi: %s\n", m_aiServerIp.c_str());
            httpd_resp_set_type(req, "text/plain");
            httpd_resp_set_hdr(req, "Access-Control-Allow-Origin", "*");
            return httpd_resp_send(req, "OK", 2);
        } else {
            httpd_resp_set_status(req, "400 Bad Request");
            return httpd_resp_send(req, "Missing or invalid ip arg", HTTPD_RESP_USE_STRLEN);
        }
    }

    esp_err_t handleSetLed(httpd_req_t *req) {
        String status = "";
        if (getQueryParam(req, "status", status)) {
            m_hw.setStageIndicator(status);
            httpd_resp_set_type(req, "text/plain");
            httpd_resp_set_hdr(req, "Access-Control-Allow-Origin", "*");
            return httpd_resp_send(req, "OK", 2);
        } else {
            httpd_resp_set_status(req, "400 Bad Request");
            return httpd_resp_send(req, "Missing status arg", HTTPD_RESP_USE_STRLEN);
        }
    }

    esp_err_t handleSetCamera(httpd_req_t *req) {
        String valStr = "";
        if (getQueryParam(req, "brightness", valStr)) {
            m_camera.setBrightness(valStr.toInt());
            Serial.printf("[EKYCService] Set Brightness -> %d\n", valStr.toInt());
        }
        if (getQueryParam(req, "ae_level", valStr)) {
            m_camera.setAeLevel(valStr.toInt());
            Serial.printf("[EKYCService] Set AE Level -> %d\n", valStr.toInt());
        }
        if (getQueryParam(req, "contrast", valStr)) {
            m_camera.setContrast(valStr.toInt());
            Serial.printf("[EKYCService] Set Contrast -> %d\n", valStr.toInt());
        }
        if (getQueryParam(req, "gainceiling", valStr)) {
            m_camera.setGainCeiling(valStr.toInt());
            Serial.printf("[EKYCService] Set GainCeiling -> %d\n", valStr.toInt());
        }

        sensor_t *s = esp_camera_sensor_get();
        char resp[256];
        if (s != nullptr) {
            snprintf(resp, sizeof(resp),
                "{\"status\":\"OK\",\"brightness\":%d,\"ae_level\":%d,\"contrast\":%d}",
                s->status.brightness, s->status.ae_level, s->status.contrast);
        } else {
            snprintf(resp, sizeof(resp), "{\"status\":\"OK\"}");
        }
        return sendJSON(req, 200, resp);
    }

    esp_err_t handleStatus(httpd_req_t *req) {
        char statusJson[256];
        snprintf(statusJson, sizeof(statusJson),
            "{\"status\":\"ONLINE\",\"ai_ip\":\"%s\",\"ai_port\":%d,\"device_id\":\"%s\",\"free_heap\":%u,\"psram\":%s}",
            m_aiServerIp.c_str(), m_aiServerPort, m_deviceId, (uint32_t)ESP.getFreeHeap(), psramFound() ? "true" : "false");
        return sendJSON(req, 200, statusJson);
    }

    // =========================================================================
    // HELPER FUNCTIONS
    // =========================================================================
    bool getQueryParam(httpd_req_t *req, const char *key, String &outVal) {
        size_t queryLen = httpd_req_get_url_query_len(req);
        if (queryLen == 0) return false;
        char *buf = (char *)malloc(queryLen + 1);
        if (!buf) return false;
        if (httpd_req_get_url_query_str(req, buf, queryLen + 1) == ESP_OK) {
            char val[128];
            if (httpd_query_key_value(buf, key, val, sizeof(val)) == ESP_OK) {
                outVal = String(val);
                free(buf);
                return true;
            }
        }
        free(buf);
        return false;
    }

    esp_err_t sendJSON(httpd_req_t *req, int httpStatus, const String &json) {
        httpd_resp_set_type(req, "application/json");
        httpd_resp_set_hdr(req, "Access-Control-Allow-Origin", "*");
        if (httpStatus == 429) {
            httpd_resp_set_status(req, "429 Too Many Requests");
        } else if (httpStatus == 400) {
            httpd_resp_set_status(req, "400 Bad Request");
        } else if (httpStatus == 500) {
            httpd_resp_set_status(req, "500 Internal Server Error");
        } else {
            httpd_resp_set_status(req, "200 OK");
        }
        return httpd_resp_send(req, json.c_str(), json.length());
    }

    CameraManager&      m_camera;
    HardwareController& m_hw;
    String              m_aiServerIp;
    int                 m_aiServerPort;
    int                 m_nodePort;
    const char*         m_deviceId;
    bool                m_isBusy;
    bool                m_streamPusherEnabled;

    httpd_handle_t      m_cameraHttpd;
    httpd_handle_t      m_streamHttpd;
    SemaphoreHandle_t   m_cameraMutex;
    TaskHandle_t        m_pusherTaskHandle;
    Preferences         m_preferences;
};

#endif // EKYC_SERVICE_H
