#include <cstdio>
#include <cstdint>
#include "esp_camera.h"
#include "esp_err.h"
#include "esp_heap_caps.h"
#include "esp_timer.h"
#include "freertos/FreeRTOS.h"
#include "freertos/task.h"

#define CAMERA_ESP32S3_EYE_MODULE
#include "cameraPins.h"

static camera_config_t make_config() {
    camera_config_t c = {};
    c.pin_pwdn     = PWDN_GPIO_NUM;
    c.pin_reset    = RESET_GPIO_NUM;
    c.pin_xclk     = XCLK_GPIO_NUM;
    c.pin_sccb_sda = SIOD_GPIO_NUM;
    c.pin_sccb_scl = SIOC_GPIO_NUM;
    c.pin_d0       = Y2_GPIO_NUM;
    c.pin_d1       = Y3_GPIO_NUM;
    c.pin_d2       = Y4_GPIO_NUM;
    c.pin_d3       = Y5_GPIO_NUM;
    c.pin_d4       = Y6_GPIO_NUM;
    c.pin_d5       = Y7_GPIO_NUM;
    c.pin_d6       = Y8_GPIO_NUM;
    c.pin_d7       = Y9_GPIO_NUM;
    c.pin_vsync    = VSYNC_GPIO_NUM;
    c.pin_href     = HREF_GPIO_NUM;
    c.pin_pclk     = PCLK_GPIO_NUM;
    c.xclk_freq_hz = 20000000;
    c.ledc_timer   = LEDC_TIMER_0;
    c.ledc_channel = LEDC_CHANNEL_0;
    c.pixel_format = PIXFORMAT_GRAYSCALE;
    c.frame_size   = FRAMESIZE_QVGA;
    c.jpeg_quality = 12;
    c.fb_count     = 2;
    c.fb_location  = CAMERA_FB_IN_PSRAM;
    c.grab_mode    = CAMERA_GRAB_LATEST;
    return c;
}

extern "C" void app_main(void) {
    esp_chip_info_t info;
    esp_chip_info(&info);
    while (true) {
        printf("hello");



        vTaskDelay(pdMS_TO_TICKS(1));
    }
}