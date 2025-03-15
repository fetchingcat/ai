#include "lighting.h"
#include "utils.h" // Add this include
#include "color.h"
#include <cmath>
#include <algorithm>

DirectionalLight::DirectionalLight() 
    : direction(Vector3(0.0f, -1.0f, 0.0f)), // Down by default
      color(Colors::WHITE),  // White light
      intensity(1.0f)
{
    // Normalize the direction
    normalizeDirection();
}

DirectionalLight::DirectionalLight(const Vector3& direction, uint32_t color, float intensity)
    : direction(direction), color(color), intensity(intensity)
{
    // Normalize the direction
    normalizeDirection();
}

void DirectionalLight::normalizeDirection() {
    // Normalize the direction vector
    float length = std::sqrt(direction.x * direction.x + 
                            direction.y * direction.y + 
                            direction.z * direction.z);
    
    if (length > 0.0001f) {
        direction.x /= length;
        direction.y /= length;
        direction.z /= length;
    }
}

PointLight::PointLight()
    : position(Vector3(0.0f, 0.0f, 0.0f)),
      color(Colors::WHITE),  // White light
      intensity(1.0f),
      radius(5.0f)
{
}

PointLight::PointLight(const Vector3& position, uint32_t color, float intensity, float radius)
    : position(position), color(color), intensity(intensity), radius(radius)
{
}

LightingSystem::LightingSystem()
    : ambientColor(Colors::WHITE),  // White ambient
      ambientIntensity(0.2f)
{
}

void LightingSystem::addDirectionalLight(const DirectionalLight& light) {
    directionalLights.push_back(light);
}

void LightingSystem::addPointLight(const PointLight& light) {
    pointLights.push_back(light);
}

// Helper function to normalize a vector
Vector3 normalizeVector(const Vector3& v) {
    float length = std::sqrt(v.x * v.x + v.y * v.y + v.z * v.z);
    if (length > 0.0001f) {
        return Vector3(v.x / length, v.y / length, v.z / length);
    }
    return v;
}

// Helper function to calculate dot product
float dotProduct(const Vector3& a, const Vector3& b) {
    return a.x * b.x + a.y * b.y + a.z * b.z;
}

// Calculate lighting for flat shading
uint32_t LightingSystem::calculateFlatShading(const Vector3& normal, const Vector3& position, uint32_t baseColor) {
    // Extract base color components
    uint8_t baseR, baseG, baseB, baseA;
    extractRGB(baseColor, baseR, baseG, baseB);
    baseA = baseColor & 0xFF; // Extract alpha
    
    // Calculate ambient lighting (dim light that's always present)
    float r = baseR * ambientIntensity;
    float g = baseG * ambientIntensity;
    float b = baseB * ambientIntensity;
    
    // Normalize the normal vector
    Vector3 normalizedNormal = normalizeVector(normal);
    
    // Add contribution from each directional light
    for (const auto& light : directionalLights) {
        // Calculate how directly this face is pointing at the light
        Vector3 lightDir = Vector3(
            -light.direction.x,
            -light.direction.y,
            -light.direction.z
        ); // Invert direction - light points toward surface
        
        float dp = dotProduct(normalizedNormal, lightDir);
        
        // Only apply light if it's hitting the front of the surface
        if (dp > 0) {
            // Extract light color
            uint8_t lightR, lightG, lightB;
            extractRGB(light.color, lightR, lightG, lightB);
            
            // Add this light's contribution (weighted by intensity and angle)
            float factor = dp * light.intensity;
            r += baseR * (lightR / 255.0f) * factor;
            g += baseG * (lightG / 255.0f) * factor;
            b += baseB * (lightB / 255.0f) * factor;
        }
    }
    
    // Clamp values to valid range
    r = std::max(0.0f, std::min(r, 255.0f));
    g = std::max(0.0f, std::min(g, 255.0f));
    b = std::max(0.0f, std::min(b, 255.0f));
    
    // Return the final color
    return makeRGB(static_cast<uint8_t>(r), static_cast<uint8_t>(g), static_cast<uint8_t>(b));
}

void LightingSystem::extractRGB(uint32_t color, uint8_t& r, uint8_t& g, uint8_t& b) {
    // Extract RGB components based on RGBA format (0xRRGGBBAA)
    r = (color >> 24) & 0xFF;
    g = (color >> 16) & 0xFF;
    b = (color >> 8) & 0xFF;
}

uint32_t LightingSystem::makeRGB(uint8_t r, uint8_t g, uint8_t b) {
    // Create color in RGBA format with full alpha
    return ((r & 0xFF) << 24) | ((g & 0xFF) << 16) | ((b & 0xFF) << 8) | 0xFF;
}