#include <cstdio>
#include <cstdint>
#include "esp_camera.h"
#include "esp_err.h"
#include "esp_heap_caps.h"
#include "esp_timer.h"
#include "freertos/FreeRTOS.h"
#include "freertos/task.h"

static camera_config_t make_config() {
    camera_config_t c = {};
    c.pin_pwdn = PWDN_GPIO_NUM;
}

extern "C" void app_main(void) {
    esp_chip_info_t info;
    esp_chip_info(&info);
    while (true) {
        printf("hello");



        vTaskDelay(pdMS_TO_TICKS(1));
    }
}