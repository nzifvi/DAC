#pragma once

#include <cstddef>
#include <cstdint>
#include "esp_camera.h"
#include "esp_err_.h"

class Frame {
    private:
        camera_fb_t *_fb = nullptr;

        void release() {
            if (_fb) {
                esp_camera_fb_return(_fb);
                _fb = nullptr;
            }

        }

    public:
        Frame() = default;
        explicit Frame(camera_fb_t *fb) : _fb(fb) {}
        ~Frame() {
            release();
        }

        Frame(const Frame &)              = delete;
        Frame &operator = (const Frame &) = delete;

        Frame(Frame &&other) noexcept : _fb(other._fb) {
            other._fb = nullptr;
        }
        Frame &operator = (Frame &&other) noexcept {
            if (this != &other) {
                release();
                _fb        = other._fb;
                other._fb  = nullptr;
            }
            return *this;
        }

        explicit operator bool() const {
            return _fb != nullptr;
        }
        const uint8_t *data() const {
            return _fb-> buf;
        }
        size_t size() const {
            return _fb->len;
        }
        uint16_t width() const {
            return (uint16_t)_fb->width;
        }
        uint16_t height() const {
            return (uint16_t)_fb->height;
        }
        pixformat_t format() const {
            return _fb->format;
        }
        uint64_t timestampUs() const {
            return (uint64_t)_fb->timestamp.tv_sec * 1000000ULL + (uint64_t)_fb->timestamp.tv_usec;
        }
};

class Camera {
    private:
        Settings _settings;
        bool     _running  = False;

        esp_err_t start(const Settngs &settings);
        bool restart(const Settings &newSettings);
    public:
};

