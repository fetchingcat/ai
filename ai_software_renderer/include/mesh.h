#ifndef MESH_H
#define MESH_H

#include "Matrix.h"
#include "color.h"
#include <vector>
#include <string>

// A simple 3D vertex with position and color
struct Vertex {
    Vector3 position;
    uint32_t color;
    
    // Constructor for convenience
    Vertex(float x, float y, float z, uint32_t c)
        : position(x, y, z), color(c) {}
    
    // Constructor taking a Vector3 and color
    Vertex(const Vector3& pos, uint32_t c)
        : position(pos), color(c) {}
};

// Edge connects two vertices in the mesh
struct Edge {
    int v1, v2;
    
    // Constructor
    Edge(int _v1, int _v2) : v1(_v1), v2(_v2) {}
};

// Triangle connects three vertices in the mesh
struct Triangle {
    int v1, v2, v3;
    
    // Constructor
    Triangle(int _v1, int _v2, int _v3) : v1(_v1), v2(_v2), v3(_v3) {}
};

class Mesh {
public:
    std::vector<Vertex> vertices;
    std::vector<Edge> edges;
    std::vector<Triangle> triangles;
    Matrix4x4 transform;
    
    Mesh();
    
    void addVertex(const Vertex& vertex);
    void addEdge(int v1, int v2);
    void addTriangle(int v1, int v2, int v3);
    void generateEdgesFromTriangles();
    
    // Add default color parameters here in the declaration
    static Mesh createCube(float size, uint32_t color = Colors::WHITE);
    static Mesh createPyramid(float base, float height, uint32_t color = Colors::WHITE);
    static Mesh createSpaceship(float size, uint32_t bodyColor = Colors::SHIP_BODY, uint32_t wingColor = Colors::SHIP_WINGS);
};

#endif