#include "esp_chip_info.h"
#include <cstdio>
#include <cstdint>
#include "esp_camera.h"
#include "esp_err.h"
#include "esp_heap_caps.h"
#include "esp_timer.h"
#include "freertos/FreeRTOS.h"
#include "freertos/projdefs.h"
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
    printf("Initialising DAC:\n");
    printf("    - PSRAM heap: %u (bytes)\n", (unsigned)heap_caps_get_total_size(MALLOC_CAP_SPIRAM));
    
    camera_config_t cfg = make_config();
    esp_err_t       err = esp_camera_init(&cfg);
    if (err != ESP_OK){
        printf("    ! CAMERA INIT FAILED: %s\n", esp_err_to_name(err));
        return;
    } else {
        printf("    - Camera successfully initialised\n");
    }
    sensor_t *s = esp_camera_sensor_get();
    camera_sensor_info_t *info = esp_camera_sensor_get_info(&s->id);

    int frames = 0;
    TickType_t windowStart = xTaskGetTickCount();
    while(true) {
        camera_fb_t *fb = esp_camera_fb_get();
        if (!fb) {
            vTaskDelay(pdMS_TO_TICKS(100));
        }

        ++frames;
        const TickType_t elapsed = xTaskGetTickCount() - windowStart;
        if (elapsed >= pdMS_TO_TICKS(1000)) {
            uint8_t lo   = 255;
            uint8_t hi   = 0;
            uint32_t sum = 0;
 
            for (size_t i = 0; i < fb->len; ++i) {
                const uint8_t p = fb->buf[i];
                if (p < lo) {
                    lo = p;
                }
                if (p > hi) {
                    hi = p;
                }
                sum += p;
            }

            printf("Image: %u x %u  \n    - %.1f fps\n    - min: %u\n    - mean: %u\n    - max %u\n\n",
                (unsigned)fb->width,
                (unsigned)fb->height,
                frames * 1000.0 / pdTICKS_TO_MS(elapsed),
                (unsigned)lo,
                (unsigned)(sum / fb->len),
                (unsigned)hi
            );
            frames = 0;
            windowStart = xTaskGetTickCount();
        }
        esp_camera_fb_return(fb);
    }
    
}