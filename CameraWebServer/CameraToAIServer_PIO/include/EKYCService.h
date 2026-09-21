#ifndef EKYC_SERVICE_H
#define EKYC_SERVICE_H

#include <Arduino.h>
#include <WebServer.h>
#include <HTTPClient.h>
#include <Preferences.h>
#include "CameraManager.h"
#include "HardwareController.h"
#include "web_ui.h"

/**
 * @class EKYCService
 * @brief Điều phối dịch vụ Web Server, tương tác HTTP Client với AI Server và chu trình eKYC 3 bước.
 */
class EKYCService {
public:
    EKYCService(CameraManager& camera, HardwareController& hw, const String& defaultAiIp, int aiPort, const char* deviceId)
        : m_camera(camera),
          m_hw(hw),
          m_server(80),
          m_aiServerIp(defaultAiIp),
          m_aiServerPort(aiPort),
          m_deviceId(deviceId),
          m_isBusy(false) {}

    void begin() {
        // Tải IP AI Server đã lưu trong Flash NVS (nếu có)
        m_preferences.begin("camera_ai", true);
        String savedIp = m_preferences.getString("ai_ip", "");
        m_preferences.end();
        if (savedIp.length() > 6) {
            m_aiServerIp = savedIp;
            Serial.printf("[EKYCService] Da load IP AI Server tu Flash: %s\n", m_aiServerIp.c_str());
        }

        // Đăng ký các Route endpoints
        m_server.on("/", HTTP_GET, [this]() { this->handleRoot(); });
        m_server.on("/capture", HTTP_GET, [this]() { this->handleCapture(); });
        m_server.on("/challenge-start", HTTP_GET, [this]() { this->handleChallengeStart(); });
        m_server.on("/challenge-step", HTTP_GET, [this]() { this->handleChallengeStep(); });
        m_server.on("/open", HTTP_GET, [this]() { this->handleOpenDoor(); });
        m_server.on("/set-ai-ip", HTTP_GET, [this]() { this->handleSetAiIp(); });

        m_server.begin();
        Serial.println("[EKYCService] Mini Web Server da san sang tai cong 80.");
    }

    void handleClient() {
        m_server.handleClient();
    }

    String getAiServerIp() const {
        return m_aiServerIp;
    }

private:
    void handleRoot() {
        String html = FPSTR(INDEX_HTML);
        html.replace("%AI_SERVER_IP%", m_aiServerIp);
        m_server.send(200, "text/html", html);
    }

    void handleCapture() {
        if (!m_camera.isInitialized()) {
            m_server.send(500, "text/plain", "Camera chua khoi tao");
            return;
        }
        camera_fb_t *fb = m_camera.capturePhotoSafe();
        if (!fb) {
            m_server.send(500, "text/plain", "Capture Failed");
            return;
        }
        m_server.send_P(200, "image/jpeg", (const char *)fb->buf, fb->len);
        m_camera.returnFrameBuffer(fb);
    }

    void handleOpenDoor() {
        m_hw.openDoor();
        m_server.send(200, "application/json", "{\"status\":\"DOOR_OPENED\",\"approved\":true}");
    }

    void handleChallengeStart() {
        if (m_isBusy) {
            m_server.send(429, "application/json", "{\"error\":\"Thiet bi dang ban\"}");
            return;
        }
        m_isBusy = true;

        // Bước 1: Nâng lên VGA 640x480 sắc nét
        m_camera.setResolution(FRAMESIZE_VGA, 10);

        camera_fb_t *fb = m_camera.capturePhotoSafe();
        if (!fb) {
            m_isBusy = false;
            m_server.send(500, "application/json", "{\"error\":\"Khong the chup anh tu camera\"}");
            return;
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
        m_isBusy = false;

        Serial.printf("[EKYCService] Challenge Start Response (%d): %s\n", httpCode, responsePayload.c_str());
        if (httpCode <= 0 || responsePayload.length() == 0) {
            String errJson = "{\"success\":false,\"passed\":false,\"message\":\"Không thể kết nối tới AI Server tại " + m_aiServerIp + ":" + String(m_aiServerPort) + " (Lỗi " + String(httpCode) + "). Vui lòng đảm bảo start_ai_server.bat đang chạy!\"}";
            m_server.send(200, "application/json", errJson);
        } else {
            m_server.send(httpCode == 200 ? 200 : 400, "application/json", responsePayload);
        }
    }

    void handleChallengeStep() {
        if (m_isBusy) {
            m_server.send(429, "application/json", "{\"error\":\"Thiet bi dang ban\"}");
            return;
        }

        String sessionId = m_server.hasArg("session_id") ? m_server.arg("session_id") : "";
        String stepName = m_server.hasArg("step") ? m_server.arg("step") : "eye_blink";

        if (sessionId.length() == 0) {
            m_server.send(400, "application/json", "{\"error\":\"Missing session_id\"}");
            return;
        }

        m_isBusy = true;

        // Bước 2 & 3: Tự động hạ về 240x240 để đạt FPS cao
        m_camera.setResolution(FRAMESIZE_240X240, 15);

        camera_fb_t *fb = m_camera.capturePhotoFast();
        if (!fb) {
            m_isBusy = false;
            m_server.send(500, "application/json", "{\"error\":\"Khong the chup anh tu camera\"}");
            return;
        }

        String fullUrl = "http://" + m_aiServerIp + ":" + String(m_aiServerPort) + "/api/v1/esp32/challenge/step";
        Serial.printf("\n[EKYCService] Gui anh Buoc [%s] (240x240, %u bytes) toi: %s ...\n", stepName.c_str(), fb->len, fullUrl.c_str());

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
        m_isBusy = false;

        if (httpCode == 200 && responsePayload.indexOf("\"approved\":true") >= 0) {
            Serial.println(">>> [eKYC MULTI-STAGE] APPROVED! Mo cua thanh cong!");
            m_hw.blinkFlash(2, 100);
            m_camera.setResolution(FRAMESIZE_240X240, 14);
        }

        Serial.printf("[EKYCService] Challenge Step Response (%d): %s\n", httpCode, responsePayload.c_str());
        if (httpCode <= 0 || responsePayload.length() == 0) {
            String errJson = "{\"success\":false,\"passed\":false,\"approved\":false,\"message\":\"Không thể kết nối tới AI Server tại " + m_aiServerIp + ":" + String(m_aiServerPort) + " (Lỗi " + String(httpCode) + "). Vui lòng đảm bảo start_ai_server.bat đang chạy!\"}";
            m_server.send(200, "application/json", errJson);
        } else {
            m_server.send(httpCode == 200 ? 200 : 400, "application/json", responsePayload);
        }
    }

    void handleSetAiIp() {
        if (m_server.hasArg("ip")) {
            m_aiServerIp = m_server.arg("ip");
            m_aiServerIp.trim();
            m_preferences.begin("camera_ai", false);
            m_preferences.putString("ai_ip", m_aiServerIp);
            m_preferences.end();
            Serial.printf("[EKYCService] Da cap nhat IP AI Server moi: %s\n", m_aiServerIp.c_str());
            m_server.send(200, "text/plain", "OK");
        } else {
            m_server.send(400, "text/plain", "Missing ip arg");
        }
    }

    CameraManager& m_camera;
    HardwareController& m_hw;
    WebServer m_server;
    Preferences m_preferences;
    String m_aiServerIp;
    int m_aiServerPort;
    const char* m_deviceId;
    bool m_isBusy;
};

#endif // EKYC_SERVICE_H
