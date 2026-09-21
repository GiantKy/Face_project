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
        // Đặt chân 48 làm OUTPUT
        pinMode(RGB_LED_PIN, OUTPUT);
        // Trạng thái chờ: Sáng màu Xanh Dương rõ ràng
        setStageIndicator("idle");
    }

    /**
     * @brief Đặt màu cho LED RGB chân 48 (WS2812 / NeoPixel)
     *        Trên ESP32 Arduino Core, neopixelWrite là hàm C chuẩn (không phải macro #define).
     */
    void setLedColor(uint8_t red, uint8_t green, uint8_t blue) {
        Serial.printf("[LED48] RGB Color -> R:%d, G:%d, B:%d\n", red, green, blue);
        // Gọi trực tiếp API neopixelWrite của ESP32-S3 Core
        neopixelWrite(RGB_LED_PIN, red, green, blue);
        
        // Đồng thời xuất trạng thái logic nếu là board dùng LED đơn
        if (red > 0 || green > 0 || blue > 0) {
            digitalWrite(RGB_LED_PIN, HIGH);
        } else {
            digitalWrite(RGB_LED_PIN, LOW);
        }
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
        Serial.printf("[HardwareController] Switching LED Stage: %s\n", stage.c_str());
        if (stage == "idle") {
            setLedColor(0, 50, 200);         // Xanh dương sáng rõ
        } else if (stage == "stage1") {
            setLedColor(180, 0, 220);        // Tím sáng rõ (Bước 1)
        } else if (stage == "stage2_blink") {
            setLedColor(255, 140, 0);        // Vàng cam rực rỡ (Bước 2 chớp mắt)
        } else if (stage == "stage3_turn") {
            setLedColor(0, 200, 200);        // Xanh ngọc Cyan sáng (Bước 3 quay đầu)
        } else if (stage == "approved") {
            setLedColor(0, 255, 0);          // Xanh lá tối đa (Pass/Mở cửa)
        } else if (stage == "rejected") {
            setLedColor(255, 0, 0);          // Đỏ rực rỡ (Cảnh báo)
        } else if (stage == "off") {
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
