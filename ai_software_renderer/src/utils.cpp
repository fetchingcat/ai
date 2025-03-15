#include "utils.h"
#include <cmath>
#include <fstream>
#include <iostream>
#include <chrono>
#include <cstdarg>

// Define M_PI if not defined (MSVC doesn't define it by default)
#ifndef M_PI
#define M_PI 3.14159265358979323846
#endif

namespace Utils {
    // Math utilities
    float lerp(float a, float b, float t) {
        return a + t * (b - a);
    }
    
    Vector3 lerp(const Vector3& a, const Vector3& b, float t) {
        return Vector3(
            lerp(a.x, b.x, t),
            lerp(a.y, b.y, t),
            lerp(a.z, b.z, t)
        );
    }
    
    float clamp(float value, float min, float max) {
        if (value < min) return min;
        if (value > max) return max;
        return value;
    }
    
    float degToRad(float degrees) {
        return degrees * (M_PI / 180.0f);
    }
    
    float radToDeg(float radians) {
        return radians * (180.0f / M_PI);
    }
    
    // Color utilities
    void extractRGBA(uint32_t color, uint8_t& r, uint8_t& g, uint8_t& b, uint8_t& a) {
        r = (color >> 24) & 0xFF;
        g = (color >> 16) & 0xFF;
        b = (color >> 8) & 0xFF;
        a = color & 0xFF;
    }
    
    uint32_t makeRGBA(uint8_t r, uint8_t g, uint8_t b, uint8_t a) {
        return ((uint32_t)r << 24) | ((uint32_t)g << 16) | ((uint32_t)b << 8) | a;
    }
    
    uint32_t lerpColor(uint32_t c1, uint32_t c2, float t) {
        uint8_t r1, g1, b1, a1;
        uint8_t r2, g2, b2, a2;
        
        extractRGBA(c1, r1, g1, b1, a1);
        extractRGBA(c2, r2, g2, b2, a2);
        
        uint8_t r = (uint8_t)lerp((float)r1, (float)r2, t);
        uint8_t g = (uint8_t)lerp((float)g1, (float)g2, t);
        uint8_t b = (uint8_t)lerp((float)b1, (float)b2, t);
        uint8_t a = (uint8_t)lerp((float)a1, (float)a2, t);
        
        return makeRGBA(r, g, b, a);
    }
    
    uint32_t adjustBrightness(uint32_t color, float factor) {
        uint8_t r, g, b, a;
        extractRGBA(color, r, g, b, a);
        
        // Adjust RGB values but clamp to valid range
        r = (uint8_t)clamp(r * factor, 0.0f, 255.0f);
        g = (uint8_t)clamp(g * factor, 0.0f, 255.0f);
        b = (uint8_t)clamp(b * factor, 0.0f, 255.0f);
        
        return makeRGBA(r, g, b, a);
    }
    
    // Timer implementation
    Timer::Timer() : startTime(0), endTime(0), running(false) {}
    
    void Timer::start() {
        startTime = std::chrono::high_resolution_clock::now().time_since_epoch().count();
        running = true;
    }
    
    void Timer::stop() {
        endTime = std::chrono::high_resolution_clock::now().time_since_epoch().count();
        running = false;
    }
    
    double Timer::elapsedMilliseconds() const {
        uint64_t end = running 
            ? std::chrono::high_resolution_clock::now().time_since_epoch().count() 
            : endTime;
        
        return (end - startTime) / 1000000.0; // Convert nanoseconds to milliseconds
    }
    
    // File operations
    bool fileExists(const std::string& filename) {
        std::ifstream file(filename);
        return file.good();
    }
    
    std::string readTextFile(const std::string& filename) {
        std::ifstream file(filename);
        if (!file.is_open()) {
            return "";
        }
        
        return std::string(
            (std::istreambuf_iterator<char>(file)),
            std::istreambuf_iterator<char>()
        );
    }
    
    bool writeTextFile(const std::string& filename, const std::string& content) {
        std::ofstream file(filename);
        if (!file.is_open()) {
            return false;
        }
        
        file << content;
        return file.good();
    }
    
    // Debug helpers
    void logMessage(const std::string& message) {
        std::cout << "[INFO] " << message << std::endl;
    }
    
    void logError(const std::string& error) {
        std::cerr << "[ERROR] " << error << std::endl;
    }
    
    std::string formatString(const char* format, ...) {
        // First, determine the required buffer size
        va_list args;
        va_start(args, format);
        va_list argsCopy;
        va_copy(argsCopy, args);
        
        int size = vsnprintf(nullptr, 0, format, argsCopy) + 1; // +1 for null terminator
        va_end(argsCopy);
        
        if (size <= 0) {
            va_end(args);
            return "";
        }
        
        // Allocate buffer and format the string
        std::vector<char> buffer(size);
        vsnprintf(buffer.data(), size, format, args);
        va_end(args);
        
        return std::string(buffer.data());
    }
}