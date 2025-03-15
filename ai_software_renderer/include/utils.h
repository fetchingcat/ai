#ifndef UTILS_H
#define UTILS_H

#include <cstdint>
#include <string>
#include <vector>
#include "Matrix.h"

namespace Utils {
    // Math utilities
    float lerp(float a, float b, float t);
    Vector3 lerp(const Vector3& a, const Vector3& b, float t);
    float clamp(float value, float min, float max);
    float degToRad(float degrees);
    float radToDeg(float radians);
    
    // Color utilities
    uint32_t lerpColor(uint32_t c1, uint32_t c2, float t);
    uint32_t adjustBrightness(uint32_t color, float factor);
    void extractRGBA(uint32_t color, uint8_t& r, uint8_t& g, uint8_t& b, uint8_t& a);
    uint32_t makeRGBA(uint8_t r, uint8_t g, uint8_t b, uint8_t a = 255);
    
    // Performance measurement
    class Timer {
    public:
        Timer();
        void start();
        void stop();
        double elapsedMilliseconds() const;
        
    private:
        uint64_t startTime;
        uint64_t endTime;
        bool running;
    };
    
    // File operations
    bool fileExists(const std::string& filename);
    std::string readTextFile(const std::string& filename);
    bool writeTextFile(const std::string& filename, const std::string& content);
    
    // Debug helpers
    void logMessage(const std::string& message);
    void logError(const std::string& error);
    std::string formatString(const char* format, ...);
}

#endif // UTILS_H