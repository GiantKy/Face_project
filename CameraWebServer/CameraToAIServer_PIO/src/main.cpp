/**
 * ============================================================================
 * ESP32-S3-CAM TO FASTAPI AI SERVER & NODE.JS INTEGRATION (OOP ARCHITECTURE)
 * ============================================================================
 * 
 * Kiến trúc OOP phân tách trách nhiệm (Separation of Concerns):
 * 1. CameraManager:      Quản lý phần cứng camera OV2640, AEC/AGC, chuyển đổi độ phân giải.
 * 2. NetworkManager:     Quản lý kết nối WiFi và mDNS.
 * 3. HardwareController: Quản lý đèn Flash LED và Relay mở cửa.
 * 4. EKYCService:        Web Server, giao diện UI và quy trình 3 bước liveness với AI Server.
 * 
 * Bảo mật:
 * - Thông tin WiFi SSID & Password được lưu trong file `config_secret.h` (bị Git ignore).
 * ============================================================================
 */

#include <Arduino.h>
#include "config_secret.h"
#include "CameraManager.h"
#include "NetworkManager.h"
#include "HardwareController.h"
#include "EKYCService.h"

// ============================================================================
// KHỞI TẠO CÁC ĐỐI TƯỢNG (OOP OBJECT INSTANTIATION)
// ============================================================================
static CameraManager      cameraManager;
static HardwareController hardwareCtrl;
static NetworkManager     networkManager(WIFI_SSID, WIFI_PASSWORD);
static EKYCService        ekycService(cameraManager, hardwareCtrl, DEFAULT_AI_SERVER_IP, DEFAULT_AI_SERVER_PORT, DEFAULT_DEVICE_ID);

void setup() {
    Serial.begin(115200);
    delay(1000);

    Serial.println("\n========================================================");
    Serial.println("  ESP32-S3 CAM: SNAPSHOT & MULTI-STAGE eKYC [OOP]");
    Serial.println("========================================================");

    // 1. Khởi tạo điều khiển phần cứng
    hardwareCtrl.begin();

    // 2. Khởi tạo cảm biến Camera
    if (!cameraManager.begin()) {
        Serial.println("[MAIN] Loi khoi tao Camera! Vui long kiem tra phan cung.");
    }

    // 3. Kết nối mạng WiFi
    networkManager.connect();

    // 4. Khởi chạy Dịch vụ eKYC Web Server
    ekycService.begin();
}

void loop() {
    ekycService.handleClient();
    delay(5);
}
