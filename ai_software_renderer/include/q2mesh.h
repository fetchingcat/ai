#ifndef Q2MESH_H
#define Q2MESH_H

#include "Mesh.h"
#include "utils.h"
#include <string>
#include <vector>
#include <map>
#include <memory>

// MD2 file format constants
#define MD2_IDENT           (('2'<<24) + ('P'<<16) + ('D'<<8) + 'I') // "IDP2" magic number
#define MD2_VERSION         8                                         // MD2 version
#define MD2_MAX_TRIANGLES   4096                                      // Max triangles per model
#define MD2_MAX_VERTICES    2048                                      // Max vertices per model
#define MD2_MAX_TEXCOORDS   2048                                      // Max texture coords
#define MD2_MAX_FRAMES      512                                       // Max animation frames
#define MD2_MAX_SKINS       32                                        // Max skins
#define MD2_MAX_FRAMESIZE   (MD2_MAX_VERTICES * 4 + 128)              // Max frame size in bytes

// MD2 file header structure
struct MD2Header {
    int32_t ident;              // Magic number, always "IDP2"
    int32_t version;            // MD2 version, should be 8
    int32_t skinWidth;          // Texture width
    int32_t skinHeight;         // Texture height
    int32_t frameSize;          // Size of each frame in bytes
    int32_t numSkins;           // Number of skins
    int32_t numVertices;        // Number of vertices per frame
    int32_t numTexCoords;       // Number of texture coordinates
    int32_t numTriangles;       // Number of triangles
    int32_t numGlCommands;      // Number of OpenGL commands (not used)
    int32_t numFrames;          // Number of frames
    int32_t offsetSkins;        // Offset to skin data
    int32_t offsetTexCoords;    // Offset to texture coordinate data
    int32_t offsetTriangles;    // Offset to triangle data
    int32_t offsetFrames;       // Offset to frame data
    int32_t offsetGlCommands;   // Offset to OpenGL commands (not used)
    int32_t offsetEnd;          // End of file
};

// MD2 compressed vertex structure (used in file)
struct MD2CompressedVertex {
    uint8_t vertex[3];          // Compressed vertex position
    uint8_t lightNormalIndex;   // Index into normals table
};

// MD2 triangle structure
struct MD2Triangle {
    uint16_t vertexIndices[3];    // Vertex indices of triangle
    uint16_t textureIndices[3];   // Texture coordinate indices
};

// MD2 texture coordinate
struct MD2TexCoord {
    int16_t s, t;               // Texture coordinates
};

// MD2 frame
struct MD2Frame {
    float scale[3];             // Scale values to decompress vertices
    float translate[3];         // Translation values to decompress vertices
    char name[16];              // Frame name (e.g., "run1", "attack3", etc.)
    MD2CompressedVertex vertices[1]; // First vertex of the frame (others follow)
};

// Our internal frame representation
struct Q2Frame {
    std::string name;
    std::vector<Vector3> vertices;
};

// MD2 animation info
struct Q2Animation {
    std::string name;           // Animation name
    int startFrame;             // Starting frame
    int endFrame;               // Ending frame
    int fps;                    // Frames per second
};

// Main Quake 2 mesh class
class Q2Mesh {
public:
    Q2Mesh();
    ~Q2Mesh();

    // Load a Quake 2 model from file
    bool loadFromFile(const std::string& filename);
    
    // Get a mesh for a specific frame
    Mesh getMeshForFrame(int frameIndex);
    
    // Get an interpolated mesh between two frames
    Mesh getInterpolatedMesh(int frame1, int frame2, float interpolation);
    
    // Animation methods
    void setAnimation(const std::string& animName);
    void updateAnimation(float deltaTime);
    Mesh getCurrentMesh();
    
    // Get info about the model
    int getNumFrames() const { return frames.size(); }
    int getNumVertices() const { return numVertices; }
    int getNumTriangles() const { return triangles.size(); }
    
private:
    // Helper functions
    void loadAnimationTable();
    
    // Model data
    std::vector<Q2Frame> frames;
    std::vector<MD2Triangle> triangles;
    std::vector<MD2TexCoord> texCoords;
    std::vector<std::string> skins;
    
    // Animation data
    std::map<std::string, Q2Animation> animations;
    std::string currentAnimation;
    int currentFrame;
    int nextFrame;
    float interpolation;
    float animationTime;
    
    // Model info
    int numVertices;
    int skinWidth;
    int skinHeight;
};

#endif // Q2MESH_H