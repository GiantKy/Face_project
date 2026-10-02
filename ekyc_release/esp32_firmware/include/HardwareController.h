#ifndef HARDWARE_CONTROLLER_H
#define HARDWARE_CONTROLLER_H

#include <Arduino.h>
#include "board_config.h"

// Chân LED RGB tích hợp trên board ESP32-S3 (ký hiệu 48 trên mạch)
#define RGB_LED_PIN 48

/**
 * @class HardwareController
 * @brief Điều khiển ngoại vi phần cứng:
 *        - Đèn LED RGB chân 48 (đổi màu theo từng stage)
 *        - Đèn Flash LED trợ sáng / cảnh báo
 *        - Relay kích hoạt mở cửa
 */
class HardwareController {
public:
    HardwareController() {}

    void begin() {
#if defined(LED_GPIO_NUM) && (LED_GPIO_NUM >= 0)
        pinMode(LED_GPIO_NUM, OUTPUT);
        digitalWrite(LED_GPIO_NUM, LOW);
#endif
        // Đặt trạng thái ban đầu: Tắt sạch rồi chuyển về Xanh Dương
        neopixelWrite(RGB_LED_PIN, 0, 0, 0);
        delayMicroseconds(60);
        setStageIndicator("idle");
    }

    /**
     * @brief Đặt màu cho LED RGB chân 48 (WS2812 / NeoPixel)
     *        Sử dụng trực tiếp API neopixelWrite chuẩn của ESP32-S3 Core.
     *        TUYỆT ĐỐI KHÔNG gọi digitalWrite trên chân NeoPixel vì sẽ khóa đường tín hiệu ở mức HIGH.
     */
    void setLedColor(uint8_t red, uint8_t green, uint8_t blue) {
        Serial.printf("[LED48] RGB Color -> R:%d, G:%d, B:%d\n", red, green, blue);
        // Xuất chuỗi xung RMT 800kHz cho WS2812
        neopixelWrite(RGB_LED_PIN, red, green, blue);
        // Giữ chân LOW ít nhất 50us để chip WS2812 chốt dữ liệu màu mới (Latch code)
        delayMicroseconds(60);
    }

    /**
     * @brief Ký hiệu màu sắc theo từng Stage của quy trình eKYC:
     * - "idle":          Xanh Dương dịu (Chờ người dùng)
     * - "stage1":        Màu Tím (Bước 1: Face Detect & Anti-Spoofing VGA)
     * - "stage2_blink":  Màu Vàng / Cam (Bước 2: Thử thách Chớp mắt)
     * - "stage3_turn":   Màu Xanh lơ Cyan (Bước 3: Thử thách Quay đầu)
     * - "approved":      Màu Xanh lá rực rỡ (Xác thực thành công, mở cửa)
     * - "rejected":      Màu Đỏ cảnh báo (Thất bại / Giả mạo / Hết giờ)
     */
    void setStageIndicator(const String& stage) {
        String s = stage;
        s.toLowerCase();
        s.trim();
        Serial.printf("[HardwareController] Switching LED Stage: %s\n", s.c_str());
        if (s == "idle" || s == "ready" || s == "preview") {
            setLedColor(0, 15, 60);          // Xanh dương dịu mắt (Idle/Chờ)
        } else if (s == "stage1" || s == "init" || s == "capture") {
            setLedColor(60, 0, 70);          // Tím vừa phải (Bước 1: Chụp ảnh & Anti-Spoofing)
        } else if (s == "stage2_blink" || s == "stage2" || s == "blink") {
            setLedColor(80, 40, 0);          // Vàng cam vừa phải (Bước 2: Chớp mắt)
        } else if (s == "stage3_turn" || s == "stage3" || s == "turn" || s == "head") {
            setLedColor(0, 60, 60);          // Xanh ngọc Cyan vừa phải (Bước 3: Quay đầu)
        } else if (s == "approved" || s == "pass" || s == "success") {
            setLedColor(0, 80, 0);           // Xanh lá êm dịu (Pass/Mở cửa)
        } else if (s == "rejected" || s == "reject" || s == "fail" || s == "timeout" || s == "spoof") {
            setLedColor(90, 0, 0);           // Đỏ rõ nét (Cảnh báo thất bại / timeout / spoof)
        } else if (s == "off") {
            setLedColor(0, 0, 0);
        }
    }

    /**
     * @brief Nháy đèn Flash để báo hiệu trạng thái
     */
    void blinkFlash(int times, int delayMs) {
#if defined(LED_GPIO_NUM) && (LED_GPIO_NUM >= 0)
        for (int i = 0; i < times; i++) {
            digitalWrite(LED_GPIO_NUM, HIGH);
            delay(delayMs);
            digitalWrite(LED_GPIO_NUM, LOW);
            delay(delayMs);
        }
#endif
    }

    /**
     * @brief Kích hoạt mở cửa (Relay, Flash & LED RGB Xanh Lá)
     */
    void openDoor() {
        Serial.println("[HardwareController] Kich hoat mo cua -> Mo Relay & nhay Flash & LED Green!");
        setStageIndicator("approved");
        blinkFlash(2, 100);
    }
};

#endif // HARDWARE_CONTROLLER_H
