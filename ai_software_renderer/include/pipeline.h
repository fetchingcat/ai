#ifndef PIPELINE_H
#define PIPELINE_H

#include "Matrix.h"
#include "Mesh.h"
#include "Renderer.h"
#include "Camera.h"
#include <vector>

// Forward declaration
class LightingSystem;

// The rendering pipeline
class Pipeline {
public:
    // Global vertex buffer (transformed vertices)
    std::vector<Vector4> transformedVertices;
    
    // Reference to the current mesh being processed
    const Mesh* currentMesh;
    
    // Camera
    Camera camera;
    
    // Backface culling flag
    bool enableBackfaceCulling;
    
    // Constructor - declaration only, implementation in cpp file
    Pipeline();
    
    // Set up the pipeline with the renderer
    void initialize();
    
    // Clear the transformed vertex buffer
    void clearBuffers();
    
    // Process a mesh through the pipeline
    void processMesh(const Mesh& mesh, const Matrix4x4& worldMatrix);
    
    // Draw the processed mesh as a wireframe
    void drawWireframe(uint32_t color = 0xFFFFFFFF);
    
    // Draw the processed mesh as solid triangles with flat shading
    void drawSolidFlat(LightingSystem& lighting);
    
    // Draw the processed mesh as textured triangles with z-buffering
    void drawTextured(SDL_Surface* texture);
    
    // Toggle backface culling
    void toggleBackfaceCulling();
    
private:
    // Check if a triangle is facing the camera (returns true for front-facing triangles)
    bool isFrontFacing(const Vector3& v1, const Vector3& v2, const Vector3& v3);
};

#endif // PIPELINE_H