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
            config.frame_size   = FRAMESIZE_VGA;      // Cấp phát buffer tối đa VGA 640x480 trong PSRAM
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
            m_initialized = false;
            return false;
        }

        sensor_t *s = esp_camera_sensor_get();
        if (s != nullptr) {
            s->set_framesize(s, FRAMESIZE_240X240); // Mặc định preview ở 240x240 nhẹ nhàng
            s->set_brightness(s, 2);                // Tăng sáng tối đa (+2) cứu sáng khuôn mặt
            s->set_contrast(s, 0);                  // Contrast = 0 (tránh bệt đen kịt vùng shadow)
            s->set_saturation(s, 0);                // Màu tự nhiên
            s->set_sharpness(s, 0);                 // Độ nét chuẩn
            
            // Tự động bù sáng & phơi sáng mạnh:
            s->set_gainceiling(s, GAINCEILING_16X); // 16X khuếch đại sáng tối đa trong phòng
            s->set_exposure_ctrl(s, 1);             // Bật tự động phơi sáng (AEC)
            s->set_aec2(s, 1);                      // Bật thuật toán DSP AEC2 nâng cao
            s->set_ae_level(s, 2);                  // Bù phơi sáng mức cao nhất (+2) cứu sáng khuôn mặt
            s->set_gain_ctrl(s, 1);                 // Bật tự động điều khiển Gain (AGC)
            s->set_bpc(s, 1);                       // Sửa điểm ảnh đen
            s->set_wpc(s, 1);                       // Sửa điểm ảnh trắng
            s->set_lenc(s, 1);                      // Bật Lens Correction chống tối 4 góc
            s->set_whitebal(s, 1);                  // Cân bằng trắng tự động (AWB)
            s->set_awb_gain(s, 1);                  // Gain cân bằng trắng
#if defined(CAMERA_MODEL_ESP32S3_EYE)
            s->set_vflip(s, 1);
#endif
        }

        // Xả 4 frame khởi động để ổn định DMA
        for (int i = 0; i < 4; i++) {
            camera_fb_t *fb = esp_camera_fb_get();
            if (fb) esp_camera_fb_return(fb);
            delay(50);
        }

        Serial.println("[CameraManager] Camera da san sang! (Buffer VGA 640x480, Active 240x240)");
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
     * @brief Chụp ảnh an toàn có xả 5 frame đệm để AEC/AGC kịp đo sáng
     */
    camera_fb_t* capturePhotoSafe() {
        for (int i = 0; i < 5; i++) {
            camera_fb_t *dummy = esp_camera_fb_get();
            if (dummy) {
                esp_camera_fb_return(dummy);
                delay(30);
            }
        }
        camera_fb_t *fb = esp_camera_fb_get();
        if (!fb) {
            delay(40);
            fb = esp_camera_fb_get();
        }
        return fb;
    }

    /**
     * @brief Chụp nhanh không xả đệm dành cho chuỗi frame thử thách cử động (Bước 2 & 3)
     */
    camera_fb_t* capturePhotoFast() {
        camera_fb_t *fb = esp_camera_fb_get();
        if (!fb) {
            delay(10);
            fb = esp_camera_fb_get();
        }
        return fb;
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
