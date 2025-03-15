#include "terrain.h"
#include "color.h"
#include <cmath>
#include <algorithm>
#include <random>

// Generate a basic terrain mesh grid
Mesh Terrain::generateTerrain(int width, int depth, float gridSize, float heightScale) {
    // Default implementation uses noise-based terrain
    return generateNoiseBasedTerrain(width, depth, gridSize, heightScale);
}

// Generate a terrain using a simple sine wave pattern (good for testing)
Mesh Terrain::generateTestTerrain(int width, int depth, float gridSize, float heightScale) {
    // Simple height function using sine waves
    auto heightFunc = [](float x, float z) {
        return sinf(x * 0.2f) * cosf(z * 0.2f);
    };
    
    return generateTerrainFromHeightFunction(width, depth, gridSize, heightScale, heightFunc);
}

// Generate a more natural looking terrain using noise
Mesh Terrain::generateNoiseBasedTerrain(
    int width, int depth, float gridSize, float heightScale, 
    float noiseScale, int octaves) 
{
    // Create a noise-based height function
    auto heightFunc = [noiseScale, octaves](float x, float z) {
        return fractalNoise(x * noiseScale, z * noiseScale, octaves);
    };
    
    return generateTerrainFromHeightFunction(width, depth, gridSize, heightScale, heightFunc);
}

// Helper that implements the common mesh generation logic for any height function
Mesh Terrain::generateTerrainFromHeightFunction(
    int width, int depth, float gridSize, float heightScale, HeightFunction heightFunc) 
{
    Mesh terrain;
    
    // Create vertices
    std::vector<std::vector<int>> vertexIndices(width, std::vector<int>(depth, -1));
    
    // Calculate the center offset to center the terrain at the origin
    float xOffset = (width * gridSize) / 2.0f;
    float zOffset = (depth * gridSize) / 2.0f;
    
    // Generate vertices grid
    for (int z = 0; z < depth; z++) {
        for (int x = 0; x < width; x++) {
            // Calculate real-world position
            float xPos = x * gridSize - xOffset;
            float zPos = z * gridSize - zOffset;
            
            // Get height from the provided function
            float height = heightFunc(xPos, zPos) * heightScale;
            
            // Store vertex and save its index
            Vertex v(xPos, height, zPos, Colors::WHITE);  // Default color, will be updated
            vertexIndices[x][z] = terrain.vertices.size();
            terrain.addVertex(v);
        }
    }
    
    // Calculate normal vectors to determine colors
    // Create triangles and calculate normals
    for (int z = 0; z < depth - 1; z++) {
        for (int x = 0; x < width - 1; x++) {
            int topLeft = vertexIndices[x][z];
            int topRight = vertexIndices[x + 1][z];
            int bottomLeft = vertexIndices[x][z + 1];
            int bottomRight = vertexIndices[x + 1][z + 1];
            
            // Calculate normals for this quad (two triangles)
            Vector3 v1 = terrain.vertices[topLeft].position;
            Vector3 v2 = terrain.vertices[topRight].position;
            Vector3 v3 = terrain.vertices[bottomLeft].position;
            Vector3 v4 = terrain.vertices[bottomRight].position;
            
            // Calculate normal for first triangle
            Vector3 edge1 = Vector3(v2.x - v1.x, v2.y - v1.y, v2.z - v1.z);
            Vector3 edge2 = Vector3(v3.x - v1.x, v3.y - v1.y, v3.z - v1.z);
            Vector3 normal1 = Vector3(
                edge1.y * edge2.z - edge1.z * edge2.y,
                edge1.z * edge2.x - edge1.x * edge2.z,
                edge1.x * edge2.y - edge1.y * edge2.x
            );
            
            // Normalize it
            float len1 = sqrtf(normal1.x * normal1.x + normal1.y * normal1.y + normal1.z * normal1.z);
            if (len1 > 0.0001f) {
                normal1.x /= len1;
                normal1.y /= len1;
                normal1.z /= len1;
            }
            
            // Make sure normal points up
            if (normal1.y < 0) {
                normal1.x = -normal1.x;
                normal1.y = -normal1.y;
                normal1.z = -normal1.z;
            }
            
            // Set colors for vertices based on height and normal
            terrain.vertices[topLeft].color = calculateTerrainColor(v1.y, heightScale, normal1);
            terrain.vertices[topRight].color = calculateTerrainColor(v2.y, heightScale, normal1);
            terrain.vertices[bottomLeft].color = calculateTerrainColor(v3.y, heightScale, normal1);
            
            // Create triangle 1
            terrain.addTriangle(topLeft, topRight, bottomLeft);
            
            // Calculate normal for second triangle
            Vector3 edge3 = Vector3(v4.x - v3.x, v4.y - v3.y, v4.z - v3.z);
            Vector3 edge4 = Vector3(v2.x - v3.x, v2.y - v3.y, v2.z - v3.z);
            Vector3 normal2 = Vector3(
                edge3.y * edge4.z - edge3.z * edge4.y,
                edge3.z * edge4.x - edge3.x * edge4.z,
                edge3.x * edge4.y - edge3.y * edge4.x
            );
            
            // Normalize it
            float len2 = sqrtf(normal2.x * normal2.x + normal2.y * normal2.y + normal2.z * normal2.z);
            if (len2 > 0.0001f) {
                normal2.x /= len2;
                normal2.y /= len2;
                normal2.z /= len2;
            }
            
            // Make sure normal points up
            if (normal2.y < 0) {
                normal2.x = -normal2.x;
                normal2.y = -normal2.y;
                normal2.z = -normal2.z;
            }
            
            // Set color for the fourth vertex
            terrain.vertices[bottomRight].color = calculateTerrainColor(v4.y, heightScale, normal2);
            
            // Create triangle 2
            terrain.addTriangle(bottomLeft, topRight, bottomRight);
        }
    }
    
    // Generate edges from triangles
    terrain.generateEdgesFromTriangles();
    
    return terrain;
}

// Simple noise function (not true Perlin noise, but good enough for this example)
float Terrain::noise(float x, float z) {
    // Hash using bit manipulation, simplified from classic Perlin implementation
    int X = static_cast<int>(std::floor(x)) & 255;
    int Z = static_cast<int>(std::floor(z)) & 255;
    
    x -= std::floor(x);
    z -= std::floor(z);
    
    // Smooth interpolation curves
    float u = x * x * (3 - 2 * x);
    float v = z * z * (3 - 2 * z);
    
    // Random gradient values
    int seed = 1234;
    auto hash = [seed](int x, int z) {
        int h = (x * 16 + z) * seed;
        h = ((h << 13) ^ h) * (h * (h * h * 15731 + 789221) + 1376312589);
        return (h & 0x7fffffff) / float(0x7fffffff);
    };
    
    // Interpolate between grid point values
    float a = hash(X, Z);
    float b = hash(X + 1, Z);
    float c = hash(X, Z + 1);
    float d = hash(X + 1, Z + 1);
    
    float k1 = a + u * (b - a);
    float k2 = c + u * (d - c);
    
    return k1 + v * (k2 - k1);
}

// Generate fractal noise (multiple octaves summed together)
float Terrain::fractalNoise(float x, float z, int octaves, float persistence, float scale) {
    float amplitude = 1.0f;
    float frequency = scale;
    float noiseSum = 0.0f;
    float amplitudeSum = 0.0f;
    
    // Sum multiple noise octaves
    for (int i = 0; i < octaves; i++) {
        noiseSum += noise(x * frequency, z * frequency) * amplitude;
        amplitudeSum += amplitude;
        amplitude *= persistence;
        frequency *= 2.0f;
    }
    
    // Normalize
    return noiseSum / amplitudeSum;
}

// Determine terrain color based on height and slope
uint32_t Terrain::calculateTerrainColor(float height, float heightScale, const Vector3& normal) {
    // Calculate slope factor (1.0 = flat, 0.0 = vertical)
    float slopeFactor = normal.y; // Dot product of normal and up vector (0,1,0) = normal.y
    
    // Normalize height between 0 and 1
    float normalizedHeight = (height / heightScale + 1.0f) * 0.5f;
    normalizedHeight = std::max(0.0f, std::min(1.0f, normalizedHeight));
    
    // Create terrain zones based on height and slope
    float snowLine = 0.8f;   // Snow above this height
    float rockLine = 0.6f;   // Rock above this height
    float grassLine = 0.4f;  // Grass above this height
    float sandLine = 0.3f;   // Sand above this height
    
    // Steepness affecting the terrain type
    float steepRockThreshold = 0.7f;  // Slopes steeper than this become rocky
    
    // Define base terrain colors
    uint32_t SNOW_COLOR = Colors::WHITE;
    uint32_t ROCK_COLOR = 0x777777FF;  // Gray
    uint32_t GRASS_COLOR = 0x33AA33FF; // Green
    uint32_t SAND_COLOR = 0xDDCC77FF;  // Sandy
    uint32_t WATER_COLOR = 0x3366CCFF; // Blue
    
    // Helper to interpolate between colors
    auto lerpColor = [](uint32_t color1, uint32_t color2, float t) {
        int r1 = (color1 >> 24) & 0xFF;
        int g1 = (color1 >> 16) & 0xFF;
        int b1 = (color1 >> 8) & 0xFF;
        int a1 = color1 & 0xFF;
        
        int r2 = (color2 >> 24) & 0xFF;
        int g2 = (color2 >> 16) & 0xFF;
        int b2 = (color2 >> 8) & 0xFF;
        int a2 = color2 & 0xFF;
        
        int r = static_cast<int>(r1 + t * (r2 - r1));
        int g = static_cast<int>(g1 + t * (g2 - g1));
        int b = static_cast<int>(b1 + t * (b2 - b1));
        int a = static_cast<int>(a1 + t * (a2 - a1));
        
        return (r << 24) | (g << 16) | (b << 8) | a;
    };
    
    // Apply snow on very steep slopes
    if (slopeFactor < steepRockThreshold) {
        // Steeper areas are rocky, regardless of height
        float rockBlend = (steepRockThreshold - slopeFactor) / steepRockThreshold;
        return lerpColor(ROCK_COLOR, ROCK_COLOR, rockBlend);
    }
    
    // Determine color based on height
    if (normalizedHeight > snowLine) {
        float t = (normalizedHeight - snowLine) / (1.0f - snowLine);
        return lerpColor(ROCK_COLOR, SNOW_COLOR, t);
    } 
    else if (normalizedHeight > rockLine) {
        float t = (normalizedHeight - rockLine) / (snowLine - rockLine);
        return lerpColor(GRASS_COLOR, ROCK_COLOR, t);
    }
    else if (normalizedHeight > grassLine) {
        float t = (normalizedHeight - grassLine) / (rockLine - grassLine);
        return lerpColor(SAND_COLOR, GRASS_COLOR, t);
    }
    else if (normalizedHeight > sandLine) {
        float t = (normalizedHeight - sandLine) / (grassLine - sandLine);
        return lerpColor(WATER_COLOR, SAND_COLOR, t);
    }
    else {
        return WATER_COLOR;
    }
}