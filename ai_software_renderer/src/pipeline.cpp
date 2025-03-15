#include "Pipeline.h"
#include <cmath>
#include "lighting.h"  // Add this include

Pipeline::Pipeline() : currentMesh(nullptr), enableBackfaceCulling(true) {
    // Initialize with a default camera
    camera = Camera();
}

void Pipeline::initialize() {
    clearBuffers();
}

void Pipeline::clearBuffers() {
    transformedVertices.clear();
}

void Pipeline::processMesh(const Mesh& mesh, const Matrix4x4& worldMatrix) {
    // Store a reference to the current mesh for drawing later
    currentMesh = &mesh;
    
    // Combine matrices for vertex transformation
    Matrix4x4 viewMatrix = camera.viewMatrix;
    Matrix4x4 projMatrix = camera.projectionMatrix;
    
    // ModelView = View * Model
    Matrix4x4 modelViewMatrix = viewMatrix * worldMatrix;
    
    // ModelViewProjection = Projection * View * Model
    Matrix4x4 mvpMatrix = projMatrix * modelViewMatrix;
    
    // Clear previous transformed vertices
    transformedVertices.clear();
    
    // Transform each vertex
    for (const auto& vertex : mesh.vertices) {
        Vector4 transformedVertex = mvpMatrix.transform(Vector4(
            vertex.position.x, vertex.position.y, vertex.position.z, 1.0f
        ));
        
        transformedVertices.push_back(transformedVertex);
    }
}

bool Pipeline::isFrontFacing(const Vector3& v1, const Vector3& v2, const Vector3& v3) {
    // Calculate screen-space vectors along triangle edges
    float edge1x = v2.x - v1.x;
    float edge1y = v2.y - v1.y;
    float edge2x = v3.x - v1.x;
    float edge2y = v3.y - v1.y;
    
    // Calculate the 2D cross product (determinant)
    // This tells us if the triangle is wound clockwise or counter-clockwise in screen space
    float crossProduct = edge1x * edge2y - edge1y * edge2x;
    
    // A positive cross product means counter-clockwise winding in screen space,
    // which typically means the triangle is front-facing
    return crossProduct > 0; // CHANGED TO > 0
}

void Pipeline::toggleBackfaceCulling() {
    enableBackfaceCulling = !enableBackfaceCulling;
}

void Pipeline::drawWireframe(uint32_t color) {
    if (!currentMesh) {
        return;
    }
    
    // Draw each triangle's edges (instead of all edges)
    // This allows us to apply backface culling to wireframe mode
    for (const auto& triangle : currentMesh->triangles) {
        if (triangle.v1 >= transformedVertices.size() ||
            triangle.v2 >= transformedVertices.size() ||
            triangle.v3 >= transformedVertices.size()) {
            continue;
        }
        
        // Get transformed vertices
        Vector4& tv1 = transformedVertices[triangle.v1];
        Vector4& tv2 = transformedVertices[triangle.v2];
        Vector4& tv3 = transformedVertices[triangle.v3];
        
        // Skip if any vertex is behind the camera
        if (tv1.w <= 0 || tv2.w <= 0 || tv3.w <= 0) {
            continue;
        }
        
        // Perspective divide
        float x1 = (tv1.x / tv1.w) * (WINDOW_WIDTH / 2) + (WINDOW_WIDTH / 2);
        float y1 = (-tv1.y / tv1.w) * (WINDOW_HEIGHT / 2) + (WINDOW_HEIGHT / 2);
        float x2 = (tv2.x / tv2.w) * (WINDOW_WIDTH / 2) + (WINDOW_WIDTH / 2);
        float y2 = (-tv2.y / tv2.w) * (WINDOW_HEIGHT / 2) + (WINDOW_HEIGHT / 2);
        float x3 = (tv3.x / tv3.w) * (WINDOW_WIDTH / 2) + (WINDOW_WIDTH / 2);
        float y3 = (-tv3.y / tv3.w) * (WINDOW_HEIGHT / 2) + (WINDOW_HEIGHT / 2);
        
        // Backface culling - skip triangles facing away from the camera
        if (enableBackfaceCulling) {
            Vector3 screen1(x1, y1, 0);
            Vector3 screen2(x2, y2, 0);
            Vector3 screen3(x3, y3, 0);
            if (!isFrontFacing(screen1, screen2, screen3)) {
                continue;
            }
        }
        
        // Draw the edges of this triangle
        drawLine((int)x1, (int)y1, (int)x2, (int)y2, color);
        drawLine((int)x2, (int)y2, (int)x3, (int)y3, color);
        drawLine((int)x3, (int)y3, (int)x1, (int)y1, color);
    }
}

void Pipeline::drawSolidFlat(LightingSystem& lighting) {
    if (!currentMesh) {
        return;
    }
    
    // Draw each triangle
    for (const auto& triangle : currentMesh->triangles) {
        if (triangle.v1 >= transformedVertices.size() ||
            triangle.v2 >= transformedVertices.size() ||
            triangle.v3 >= transformedVertices.size()) {
            continue;
        }
        
        // Get transformed vertices
        Vector4& tv1 = transformedVertices[triangle.v1];
        Vector4& tv2 = transformedVertices[triangle.v2];
        Vector4& tv3 = transformedVertices[triangle.v3];
        
        // Skip if any vertex is behind the camera
        if (tv1.w <= 0 || tv2.w <= 0 || tv3.w <= 0) {
            continue;
        }
        
        // Perspective divide to get NDC coordinates
        float ndcx1 = tv1.x / tv1.w;
        float ndcy1 = tv1.y / tv1.w;
        float ndcz1 = tv1.z / tv1.w;
        
        float ndcx2 = tv2.x / tv2.w;
        float ndcy2 = tv2.y / tv2.w;
        float ndcz2 = tv2.z / tv2.w;
        
        float ndcx3 = tv3.x / tv3.w;
        float ndcy3 = tv3.y / tv3.w;
        float ndcz3 = tv3.z / tv3.w;
        
        // Convert to screen space
        int x1 = (int)((ndcx1 + 1.0f) * (WINDOW_WIDTH / 2));
        int y1 = (int)((-ndcy1 + 1.0f) * (WINDOW_HEIGHT / 2));
        int x2 = (int)((ndcx2 + 1.0f) * (WINDOW_WIDTH / 2));
        int y2 = (int)((-ndcy2 + 1.0f) * (WINDOW_HEIGHT / 2));
        int x3 = (int)((ndcx3 + 1.0f) * (WINDOW_WIDTH / 2));
        int y3 = (int)((-ndcy3 + 1.0f) * (WINDOW_HEIGHT / 2));
        
        // Backface culling - skip triangles facing away from the camera
        if (enableBackfaceCulling) {
            Vector3 screen1(x1, y1, 0);
            Vector3 screen2(x2, y2, 0);
            Vector3 screen3(x3, y3, 0);
            if (!isFrontFacing(screen1, screen2, screen3)) {
                continue;
            }
        }
        
        // Get original vertices for shading calculation
        Vector3 worldPos1 = currentMesh->vertices[triangle.v1].position;
        Vector3 worldPos2 = currentMesh->vertices[triangle.v2].position;
        Vector3 worldPos3 = currentMesh->vertices[triangle.v3].position;
        
        // Calculate face normal for lighting
        Vector3 edge1(worldPos2.x - worldPos1.x, worldPos2.y - worldPos1.y, worldPos2.z - worldPos1.z);
        Vector3 edge2(worldPos3.x - worldPos1.x, worldPos3.y - worldPos1.y, worldPos3.z - worldPos1.z);
        
        // Cross product to get normal
        float normalX = edge1.y * edge2.z - edge1.z * edge2.y;
        float normalY = edge1.z * edge2.x - edge1.x * edge2.z;
        float normalZ = edge1.x * edge2.y - edge1.y * edge2.x;
        
        // Normalize
        float length = sqrt(normalX * normalX + normalY * normalY + normalZ * normalZ);
        if (length > 0.0001f) {
            normalX /= length;
            normalY /= length;
            normalZ /= length;
        }
        
        Vector3 normal(normalX, normalY, normalZ);
        
        // Get the triangle center for lighting calculation
        Vector3 center = Vector3(
            (worldPos1.x + worldPos2.x + worldPos3.x) / 3.0f,
            (worldPos1.y + worldPos2.y + worldPos3.y) / 3.0f,
            (worldPos1.z + worldPos2.z + worldPos3.z) / 3.0f
        );
        
        // Get the base color from the first vertex (simple approach)
        uint32_t baseColor = currentMesh->vertices[triangle.v1].color;
        
        // Calculate the final color with lighting
        uint32_t finalColor = lighting.calculateFlatShading(normal, center, baseColor);
        
        // Draw filled triangle
        fillTriangle(x1, y1, x2, y2, x3, y3, finalColor);
    }
}