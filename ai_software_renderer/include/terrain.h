#pragma once

#include "Mesh.h"
#include "color.h"
#include <vector>
#include <string>
#include <functional> // Add this for std::function

class Terrain {
public:
    // Generate a terrain mesh from a height function
    static Mesh generateTerrain(
        int width, 
        int depth, 
        float gridSize = 1.0f,
        float heightScale = 1.0f
    );
    
    // Generate terrain from a noise function (creates rolling hills)
    static Mesh generateNoiseBasedTerrain(
        int width, 
        int depth, 
        float gridSize = 1.0f,
        float heightScale = 1.0f,
        float noiseScale = 0.1f,
        int octaves = 4
    );
    
    // Generate a simple test terrain (sine waves)
    static Mesh generateTestTerrain(
        int width, 
        int depth, 
        float gridSize = 1.0f,
        float heightScale = 1.0f
    );

private:
    // Function that determines color based on height and slope
    static uint32_t calculateTerrainColor(float height, float heightScale, const Vector3& normal);
    
    // Helper method to generate the actual mesh from any height function
    // Change from function pointer to std::function
    typedef std::function<float(float,float)> HeightFunction;
    
    static Mesh generateTerrainFromHeightFunction(
        int width, 
        int depth, 
        float gridSize,
        float heightScale,
        HeightFunction heightFunc
    );
    
    // Simple noise function (Perlin-like but simplified)
    static float noise(float x, float z);
    static float fractalNoise(float x, float z, int octaves, float persistence = 0.5f, float scale = 1.0f);
};