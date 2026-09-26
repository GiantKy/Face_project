#ifndef CAMERA_MANAGER_H
#define CAMERA_MANAGER_H

#include <Arduino.h>
#include "esp_camera.h"
#include "board_config.h"

/**
 * @class CameraManager
 * @brief Lớp đóng gói toàn bộ logic khởi tạo, cấu hình phần cứng ISP và chụp ảnh của ESP32-CAM.
 */
class CameraManager {
public:
    CameraManager() : m_initialized(false) {}

    /**
     * @brief Khởi tạo cảm biến camera với cấu hình tối đa FRAMESIZE_VGA trong PSRAM
     *        để phục vụ chuyển đổi độ phân giải linh hoạt (Hybrid Resolution).
     */
    bool begin() {
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
            config.frame_size   = FRAMESIZE_VGA;      // Độ phân giải VGA 640x480 trong PSRAM
            config.jpeg_quality = 20;                 // Quality 20 theo chuẩn hệ thống
            config.fb_count     = 2;
            config.fb_location  = CAMERA_FB_IN_PSRAM;
        } else {
            config.frame_size   = FRAMESIZE_VGA;
            config.jpeg_quality = 20;
            config.fb_count     = 1;
            config.fb_location  = CAMERA_FB_IN_DRAM;
        }

        esp_err_t err = esp_camera_init(&config);
        if (err != ESP_OK) {
            Serial.printf("\n[ERROR] Khoi tao camera that bai! 0x%x\n", err);
            m_initialized = false;
            return false;
        }

        sensor_t *s = esp_camera_sensor_get();
        if (s != nullptr) {
            s->set_framesize(s, FRAMESIZE_VGA);     // Độ phân giải VGA 640x480
            s->set_quality(s, 20);                  // Quality 20: Sắc nét, nén tối ưu băng thông WiFi
            s->set_brightness(s, 1);                // Brightness = +1 (Nâng sáng sàn để mặt không bị sập tối khi ngược sáng)
            s->set_contrast(s, 0);                  // Contrast = 0 (Giảm tương phản để mở rộng dải động WDR, chống dìm đen bóng râm mặt)
            s->set_saturation(s, 0);                // Saturation = 0 (tự nhiên)
            s->set_sharpness(s, 2);                 // Sharpness = +2 (sắc nét chi tiết mắt & da)
            s->set_denoise(s, 0);                   // De-Noise = 0 (TẮT khử nhiễu để tránh làm mờ/bệt chi tiết da)
            
            // Các chế độ phơi sáng, cân bằng trắng và khử quang sai:
            s->set_gainceiling(s, GAINCEILING_16X); // Nâng Gainceiling lên 16X để tự động bù sáng tối ưu trong phòng
            s->set_exposure_ctrl(s, 1);             // AEC1 Hardware Auto Exposure = ON (chạy phần cứng ổn định)
            s->set_aec2(s, 0);                      // TẮT AEC2 DSP: Tránh lỗi kéo dài màn trập gây mờ ảnh và tụt FPS
            s->set_ae_level(s, 1);                  // AE Level = +1: BÙ SÁNG NGƯỢC SÁNG (Backlight Compensation)! Ngăn mặt bị tối đen khi sau lưng có cửa sổ/đèn
            s->set_gain_ctrl(s, 1);                 // AGC Enable = ON
            s->set_bpc(s, 1);                       // BPC = ON
            s->set_wpc(s, 1);                       // WPC = ON
            s->set_raw_gma(s, 1);                   // GMA Enable (Gamma) = ON
            s->set_lenc(s, 0);                      // Lens Correction = OFF (TẮT để loại bỏ hoàn toàn quầng hồng/tím ở tâm)
            s->set_whitebal(s, 1);                  // AWB Enable = ON
            s->set_awb_gain(s, 1);                  // Advanced AWB Gain = ON
            s->set_dcw(s, 1);                       // Advanced AWB DCW = ON
            s->set_special_effect(s, 0);            // Special Effect = No Effect
            s->set_hmirror(s, 1);                   // H-Mirror = ON (đảo chiều ngang giúp quay đầu đúng hướng)
            s->set_vflip(s, 1);                     // V-Flip = ON
        }

        // Xả 10 frame khởi động để AEC & DMA ổn định độ sáng chuẩn ngay khi mở (tránh lúc đầu bị tối)
        for (int i = 0; i < 10; i++) {
            camera_fb_t *fb = esp_camera_fb_get();
            if (fb) esp_camera_fb_return(fb);
            delay(30);
        }

        Serial.println("[CameraManager] Camera da san sang! (Resolution: VGA 640x480, JPEG Quality: 20)");
        m_initialized = true;
        return true;
    }

    /**
     * @brief Thay đổi độ phân giải và chất lượng JPEG động khi đang chạy
     */
    void setResolution(framesize_t size, int quality) {
        sensor_t *s = esp_camera_sensor_get();
        if (s != nullptr) {
            if (s->status.framesize != size) {
                s->set_framesize(s, size);
                // Xả 2 frame đệm cũ trong DMA/PSRAM để tránh vỡ hình
                for (int i = 0; i < 2; i++) {
                    camera_fb_t *fb = esp_camera_fb_get();
                    if (fb) esp_camera_fb_return(fb);
                    delay(10);
                }
            }
            s->set_quality(s, quality);
        }
    }

    /**
     * @brief Điều chỉnh độ sáng (-2 đến 2)
     */
    void setBrightness(int val) {
        sensor_t *s = esp_camera_sensor_get();
        if (s != nullptr) s->set_brightness(s, constrain(val, -2, 2));
    }

    /**
     * @brief Điều chỉnh mục tiêu phơi sáng tự động AEC (-2 đến 2)
     */
    void setAeLevel(int val) {
        sensor_t *s = esp_camera_sensor_get();
        if (s != nullptr) s->set_ae_level(s, constrain(val, -2, 2));
    }

    /**
     * @brief Điều chỉnh tương phản (-2 đến 2)
     */
    void setContrast(int val) {
        sensor_t *s = esp_camera_sensor_get();
        if (s != nullptr) s->set_contrast(s, constrain(val, -2, 2));
    }

    /**
     * @brief Điều chỉnh Gain Ceiling (0: 2X, 1: 4X, 2: 8X, 3: 16X, 4: 32X)
     */
    void setGainCeiling(int level) {
        sensor_t *s = esp_camera_sensor_get();
        if (s != nullptr) {
            gainceiling_t gc = GAINCEILING_16X;
            switch(level) {
                case 0: gc = GAINCEILING_2X; break;
                case 1: gc = GAINCEILING_4X; break;
                case 2: gc = GAINCEILING_8X; break;
                case 3: gc = GAINCEILING_16X; break;
                case 4: gc = GAINCEILING_32X; break;
                case 5: gc = GAINCEILING_64X; break;
                case 6: gc = GAINCEILING_128X; break;
                default: gc = GAINCEILING_16X; break;
            }
            s->set_gainceiling(s, gc);
        }
    }

    /**
     * @brief Chụp ảnh an toàn lấy ngay frame mới nhất từ bộ đệm kép DMA/PSRAM
     */
    camera_fb_t* capturePhotoSafe() {
        camera_fb_t *fb = esp_camera_fb_get();
        if (!fb) {
            vTaskDelay(pdMS_TO_TICKS(10));
            fb = esp_camera_fb_get();
        }
        return fb;
    }

    /**
     * @brief Chụp nhanh không delay dành cho chuỗi frame thử thách cử động (Bước 2 & 3)
     */
    camera_fb_t* capturePhotoFast() {
        return esp_camera_fb_get();
    }

    void returnFrameBuffer(camera_fb_t* fb) {
        if (fb != nullptr) {
            esp_camera_fb_return(fb);
        }
    }

    bool isInitialized() const {
        return m_initialized;
    }

private:
    bool m_initialized;
};

#endif // CAMERA_MANAGER_H
