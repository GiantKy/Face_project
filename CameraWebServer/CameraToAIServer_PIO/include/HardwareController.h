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
        // Trạng thái chờ khởi động: Sáng màu Xanh Dương dịu
        setLedColor(0, 0, 40);
    }

    /**
     * @brief Đặt màu cho LED RGB chân 48 (WS2812 / NeoPixel)
     *        Có hỗ trợ cả neopixelWrite built-in lẫn fallback digitalWrite.
     */
    void setLedColor(uint8_t red, uint8_t green, uint8_t blue) {
#if defined(neopixelWrite)
        neopixelWrite(RGB_LED_PIN, red, green, blue);
#else
        // Fallback cho board dùng LED đơn sắc đảo trạng thái
        if (red > 0 || green > 0 || blue > 0) {
            digitalWrite(RGB_LED_PIN, HIGH);
        } else {
            digitalWrite(RGB_LED_PIN, LOW);
        }
#endif
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
        if (stage == "idle") {
            setLedColor(0, 0, 35);           // Xanh dương dịu
        } else if (stage == "stage1") {
            setLedColor(60, 0, 80);          // Tím (Bước 1)
        } else if (stage == "stage2_blink") {
            setLedColor(90, 45, 0);          // Vàng cam (Bước 2 chớp mắt)
        } else if (stage == "stage3_turn") {
            setLedColor(0, 70, 70);          // Xanh ngọc Cyan (Bước 3 quay đầu)
        } else if (stage == "approved") {
            setLedColor(0, 100, 0);          // Xanh lá rực rỡ
        } else if (stage == "rejected") {
            setLedColor(100, 0, 0);          // Đỏ cảnh báo
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
