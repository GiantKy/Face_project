#ifndef HARDWARE_CONTROLLER_H
#define HARDWARE_CONTROLLER_H

#include <Arduino.h>
#include "board_config.h"

/**
 * @class HardwareController
 * @brief Điều khiển các ngoại vi phần cứng: Đèn Flash LED, Relay đóng mở cửa.
 */
class HardwareController {
public:
    HardwareController() {}

    void begin() {
#if defined(LED_GPIO_NUM) && (LED_GPIO_NUM >= 0)
        pinMode(LED_GPIO_NUM, OUTPUT);
        digitalWrite(LED_GPIO_NUM, LOW);
#endif
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
     * @brief Kích hoạt mở cửa (Relay & Flash)
     */
    void openDoor() {
        Serial.println("[HardwareController] Kich hoat mo cua -> Mo Relay & nhay Flash!");
        blinkFlash(2, 100);
    }
};

#endif // HARDWARE_CONTROLLER_H
