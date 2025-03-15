#ifndef LIGHTING_H
#define LIGHTING_H

#include "Matrix.h"
#include <vector>

// Represents a directional light source (like the sun)
class DirectionalLight {
public:
    Vector3 direction;
    uint32_t color;
    float intensity;
    
    DirectionalLight();
    DirectionalLight(const Vector3& direction, uint32_t color, float intensity);
    
private:
    void normalizeDirection();
};

// Represents a point light source (like a light bulb)
class PointLight {
public:
    Vector3 position;
    uint32_t color;
    float intensity;
    float radius;
    
    PointLight();
    PointLight(const Vector3& position, uint32_t color, float intensity, float radius);
};

// Manages lights and calculates lighting for the scene
class LightingSystem {
public:
    std::vector<DirectionalLight> directionalLights;
    std::vector<PointLight> pointLights;
    uint32_t ambientColor;
    float ambientIntensity;
    
    LightingSystem();
    
    void addDirectionalLight(const DirectionalLight& light);
    void addPointLight(const PointLight& light);
    
    // Calculate lighting for flat shading
    uint32_t calculateFlatShading(const Vector3& normal, const Vector3& position, uint32_t baseColor);
    
private:
    // Helper methods for color manipulation
    void extractRGB(uint32_t color, uint8_t& r, uint8_t& g, uint8_t& b);
    uint32_t makeRGB(uint8_t r, uint8_t g, uint8_t b);
};

#endif // LIGHTING_H