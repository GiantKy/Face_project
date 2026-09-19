#ifndef BOARD_CONFIG_H
#define BOARD_CONFIG_H

// ============================================================================
// CHỌN MODEL CAMERA PHÙ HỢP VỚI BOARD CỦA BẠN (Mở comment 1 dòng tương ứng)
// ============================================================================
//#define CAMERA_MODEL_WROVER_KIT          // ESP32 Wrover Kit
//#define CAMERA_MODEL_ESP_EYE             // ESP-EYE
//#define CAMERA_MODEL_AI_THINKER          // ESP32-CAM AI-Thinker
#define CAMERA_MODEL_ESP32S3_EYE         // ESP32-S3-CAM / ESP32-S3-EYE (Chuẩn cho ESP32-S3)
//#define CAMERA_MODEL_M5STACK_PSRAM       // M5Stack PSRAM
//#define CAMERA_MODEL_M5STACK_V2_PSRAM    // M5Camera version B
//#define CAMERA_MODEL_M5STACK_WIDE        // M5Camera wide
//#define CAMERA_MODEL_M5STACK_ESP32CAM    // M5Stack without PSRAM
//#define CAMERA_MODEL_M5STACK_UNITCAM     // M5Stack UnitCam
//#define CAMERA_MODEL_M5STACK_CAMS3_UNIT  // M5Stack CamS3 Unit
//#define CAMERA_MODEL_TTGO_T_JOURNAL      // TTGO T-Journal
//#define CAMERA_MODEL_XIAO_ESP32S3        // Seeed XIAO ESP32S3 Sense
//#define CAMERA_MODEL_DFRobot_FireBeetle2_ESP32S3
//#define CAMERA_MODEL_DFRobot_Romeo_ESP32S3

#include "camera_pins.h"

#endif  // BOARD_CONFIG_H
