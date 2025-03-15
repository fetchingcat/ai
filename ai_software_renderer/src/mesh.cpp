#include "Mesh.h"
#include "color.h" // Add this include

Mesh::Mesh() : transform(Matrix4x4::identity()) {
}

void Mesh::addVertex(const Vertex& vertex) {
    vertices.push_back(vertex);
}

void Mesh::addEdge(int v1, int v2) {
    edges.push_back(Edge(v1, v2));
}

void Mesh::addTriangle(int v1, int v2, int v3) {
    triangles.push_back(Triangle(v1, v2, v3));
}

void Mesh::generateEdgesFromTriangles() {
    edges.clear();
    
    // For each triangle, add its three edges
    for (const auto& tri : triangles) {
        // Add edges in both directions to avoid duplicates
        // Check if edge already exists
        auto edgeExists = [this](int v1, int v2) {
            for (const auto& edge : edges) {
                if ((edge.v1 == v1 && edge.v2 == v2) || 
                    (edge.v1 == v2 && edge.v2 == v1)) {
                    return true;
                }
            }
            return false;
        };
        
        // Add edges if they don't already exist
        if (!edgeExists(tri.v1, tri.v2)) {
            edges.push_back(Edge(tri.v1, tri.v2));
        }
        
        if (!edgeExists(tri.v2, tri.v3)) {
            edges.push_back(Edge(tri.v2, tri.v3));
        }
        
        if (!edgeExists(tri.v3, tri.v1)) {
            edges.push_back(Edge(tri.v3, tri.v1));
        }
    }
}

// Remove default parameters from implementations
Mesh Mesh::createCube(float size, uint32_t color) {
    Mesh cube;
    float halfSize = size / 2.0f;
    
    // Create vertices (8 corners of the cube)
    cube.addVertex(Vertex(-halfSize, -halfSize, -halfSize, color)); // 0: left-bottom-back
    cube.addVertex(Vertex(halfSize, -halfSize, -halfSize, color));  // 1: right-bottom-back
    cube.addVertex(Vertex(halfSize, halfSize, -halfSize, color));   // 2: right-top-back
    cube.addVertex(Vertex(-halfSize, halfSize, -halfSize, color));  // 3: left-top-back
    cube.addVertex(Vertex(-halfSize, -halfSize, halfSize, color));  // 4: left-bottom-front
    cube.addVertex(Vertex(halfSize, -halfSize, halfSize, color));   // 5: right-bottom-front
    cube.addVertex(Vertex(halfSize, halfSize, halfSize, color));    // 6: right-top-front
    cube.addVertex(Vertex(-halfSize, halfSize, halfSize, color));   // 7: left-top-front
    
    // Create triangles (12 triangles, 2 per face)
    // Back face
    cube.addTriangle(0, 1, 2);
    cube.addTriangle(0, 2, 3);
    
    // Front face
    cube.addTriangle(4, 6, 5);
    cube.addTriangle(4, 7, 6);
    
    // Left face
    cube.addTriangle(0, 3, 7);
    cube.addTriangle(0, 7, 4);
    
    // Right face
    cube.addTriangle(1, 5, 6);
    cube.addTriangle(1, 6, 2);
    
    // Bottom face
    cube.addTriangle(0, 4, 5);
    cube.addTriangle(0, 5, 1);
    
    // Top face
    cube.addTriangle(3, 2, 6);
    cube.addTriangle(3, 6, 7);
    
    // Generate edges for wireframe rendering
    cube.generateEdgesFromTriangles();
    
    return cube;
}

// Remove default parameters from implementations
Mesh Mesh::createPyramid(float base, float height, uint32_t color) {
    Mesh pyramid;
    float halfBase = base / 2.0f;
    
    // Create vertices
    pyramid.addVertex(Vertex(-halfBase, 0, -halfBase, color)); // 0: left-bottom-back
    pyramid.addVertex(Vertex(halfBase, 0, -halfBase, color));  // 1: right-bottom-back
    pyramid.addVertex(Vertex(halfBase, 0, halfBase, color));   // 2: right-bottom-front
    pyramid.addVertex(Vertex(-halfBase, 0, halfBase, color));  // 3: left-bottom-front
    pyramid.addVertex(Vertex(0, height, 0, color));           // 4: top
    
    // Create triangles
    // Base
    pyramid.addTriangle(0, 2, 1);
    pyramid.addTriangle(0, 3, 2);
    
    // Sides
    pyramid.addTriangle(0, 1, 4);
    pyramid.addTriangle(1, 2, 4);
    pyramid.addTriangle(2, 3, 4);
    pyramid.addTriangle(3, 0, 4);
    
    // Generate edges for wireframe rendering
    pyramid.generateEdgesFromTriangles();
    
    return pyramid;
}

// Remove default parameters from implementations
Mesh Mesh::createSpaceship(float size, uint32_t bodyColor, uint32_t wingColor) {
    Mesh ship;
    
    // Scale all coordinates by size
    float s = size;
    
    // Body vertices (elongated tetrahedron shape)
    ship.addVertex(Vertex{Vector3(0.0f * s, 0.0f * s, 2.0f * s), bodyColor});   // 0: nose
    ship.addVertex(Vertex{Vector3(-0.5f * s, -0.25f * s, 0.0f * s), bodyColor}); // 1: left bottom
    ship.addVertex(Vertex{Vector3(0.5f * s, -0.25f * s, 0.0f * s), bodyColor});  // 2: right bottom
    ship.addVertex(Vertex{Vector3(0.0f * s, 0.25f * s, 0.0f * s), bodyColor});   // 3: top
    ship.addVertex(Vertex{Vector3(0.0f * s, 0.0f * s, -1.0f * s), bodyColor});   // 4: rear center

    // Wing vertices
    ship.addVertex(Vertex{Vector3(-1.5f * s, 0.0f * s, -0.5f * s), wingColor}); // 5: left wing tip
    ship.addVertex(Vertex{Vector3(1.5f * s, 0.0f * s, -0.5f * s), wingColor});  // 6: right wing tip
    ship.addVertex(Vertex{Vector3(0.0f * s, -0.5f * s, -0.5f * s), wingColor}); // 7: bottom fin

    // Body edges (forming the fuselage)
    ship.addEdge(0, 1);  // nose to left bottom
    ship.addEdge(0, 2);  // nose to right bottom
    ship.addEdge(0, 3);  // nose to top
    ship.addEdge(1, 2);  // left bottom to right bottom
    ship.addEdge(1, 3);  // left bottom to top
    ship.addEdge(2, 3);  // right bottom to top
    ship.addEdge(1, 4);  // left bottom to rear
    ship.addEdge(2, 4);  // right bottom to rear
    ship.addEdge(3, 4);  // top to rear

    // Wing edges
    ship.addEdge(1, 5);  // left bottom to left wing tip
    ship.addEdge(5, 4);  // left wing tip to rear
    ship.addEdge(2, 6);  // right bottom to right wing tip
    ship.addEdge(6, 4);  // right wing tip to rear
    ship.addEdge(4, 7);  // rear to bottom fin
    ship.addEdge(7, 1);  // bottom fin to left bottom
    ship.addEdge(7, 2);  // bottom fin to right bottom

    // ADD TRIANGLES FOR FLAT SHADING
    
    // Nose triangles (front pyramid)
    ship.addTriangle(0, 1, 2);  // nose, left bottom, right bottom
    ship.addTriangle(0, 2, 3);  // nose, right bottom, top
    ship.addTriangle(0, 3, 1);  // nose, top, left bottom
    
    // Body triangles
    ship.addTriangle(1, 3, 4);  // left bottom, top, rear
    ship.addTriangle(3, 2, 4);  // top, right bottom, rear
    ship.addTriangle(2, 1, 4);  // right bottom, left bottom, rear
    
    // Left wing triangle
    ship.addTriangle(1, 4, 5);  // left bottom, rear, left wing
    
    // Right wing triangle
    ship.addTriangle(2, 6, 4);  // right bottom, right wing, rear
    
    // Bottom fin triangles
    ship.addTriangle(1, 7, 4);  // left bottom, bottom fin, rear
    ship.addTriangle(7, 2, 4);  // bottom fin, right bottom, rear
    
    return ship;
}