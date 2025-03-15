#include "q2mesh.h"
#include <fstream>
#include <iostream>
#include <cstring>
#include <algorithm>
#include <cmath>

// Normal table used by Quake 2 (pre-computed normal vectors for lighting)
static float normalTable[162][3] = {
    // This is a long table, I'll include a shortened version for brevity
    {-0.525731f, 0.000000f, 0.850651f},
    {-0.442863f, 0.238856f, 0.864188f},
    {-0.295242f, 0.000000f, 0.955423f},
    // ... (more normal vectors)
    {0.000000f, -1.000000f, 0.000000f}
};

// Standard Quake 2 animation table
static struct {
    const char* name;
    int firstFrame;
    int lastFrame;
    int fps;
    bool loop;
} animationTable[] = {
    {"stand",       0,  39,  9,  true},
    {"run",        40,  45,  10, true},
    {"attack",     46,  53,  10, true},
    {"pain1",      54,  57,  7,  true},
    {"pain2",      58,  61,  7,  true},
    {"pain3",      62,  65,  7,  true},
    {"jump",       66,  71,  7,  true},
    {"flip",       72,  83,  7,  true},
    {"salute",     84,  94,  7,  true},
    {"taunt",      95,  111, 10, true},
    {"wave",       112, 122, 7,  true},
    {"point",      123, 134, 6,  true},
    {"crstnd",     135, 153, 10, true},
    {"crwalk",     154, 159, 7,  true},
    {"crattak",    160, 168, 10, true},
    {"crpain",     169, 172, 7,  true},
    {"crdeath",    173, 177, 5,  false},
    {"death1",     178, 183, 7,  false},
    {"death2",     184, 189, 7,  false},
    {"death3",     190, 197, 7,  false}
};

Q2Mesh::Q2Mesh() 
    : currentFrame(0), nextFrame(1), interpolation(0.0f), animationTime(0.0f)
{
}

Q2Mesh::~Q2Mesh() {
}

bool Q2Mesh::loadFromFile(const std::string& filename) {
    std::ifstream file(filename, std::ios::binary);
    if (!file.is_open()) {
        std::cerr << "Error: Could not open MD2 file: " << filename << std::endl;
        return false;
    }
    
    // Read header
    MD2Header header;
    file.read(reinterpret_cast<char*>(&header), sizeof(MD2Header));
    
    // Check file validity
    if (header.ident != MD2_IDENT || header.version != MD2_VERSION) {
        std::cerr << "Error: Invalid MD2 file format or version" << std::endl;
        return false;
    }
    
    // Store model information
    numVertices = header.numVertices;
    skinWidth = header.skinWidth;
    skinHeight = header.skinHeight;
    
    // Read skin names
    file.seekg(header.offsetSkins, std::ios::beg);
    skins.resize(header.numSkins);
    for (int i = 0; i < header.numSkins; i++) {
        char skinName[64];
        file.read(skinName, 64);
        skins[i] = skinName;
    }
    
    // Read texture coordinates
    file.seekg(header.offsetTexCoords, std::ios::beg);
    texCoords.resize(header.numTexCoords);
    file.read(reinterpret_cast<char*>(texCoords.data()), sizeof(MD2TexCoord) * header.numTexCoords);
    
    // Read triangles
    file.seekg(header.offsetTriangles, std::ios::beg);
    triangles.resize(header.numTriangles);
    file.read(reinterpret_cast<char*>(triangles.data()), sizeof(MD2Triangle) * header.numTriangles);
    
    // Read frames
    file.seekg(header.offsetFrames, std::ios::beg);
    frames.resize(header.numFrames);
    
    // Allocate temporary buffer for frame data
    std::vector<char> frameData(header.frameSize);
    
    for (int i = 0; i < header.numFrames; i++) {
        // Read the frame data into buffer
        file.read(frameData.data(), header.frameSize);
        
        // Get the frame header
        MD2Frame* frame = reinterpret_cast<MD2Frame*>(frameData.data());
        
        // Set frame name
        frames[i].name = std::string(frame->name);
        // Remove frame number from name (e.g., "run1" -> "run")
        size_t nameLen = frames[i].name.find_first_of("0123456789");
        if (nameLen != std::string::npos) {
            frames[i].name = frames[i].name.substr(0, nameLen);
        }
        
        // Decompress vertices
        frames[i].vertices.resize(header.numVertices);
        for (int j = 0; j < header.numVertices; j++) {
            const MD2CompressedVertex& compVert = 
                *reinterpret_cast<MD2CompressedVertex*>(frameData.data() + 
                sizeof(float) * 6 + sizeof(char) * 16 + sizeof(MD2CompressedVertex) * j);
            
            // Decompress vertex position
            frames[i].vertices[j].x = (frame->scale[0] * compVert.vertex[0] + frame->translate[0]);
            frames[i].vertices[j].y = (frame->scale[1] * compVert.vertex[1] + frame->translate[1]);
            frames[i].vertices[j].z = (frame->scale[2] * compVert.vertex[2] + frame->translate[2]);
        }
    }
    
    file.close();
    
    // Load animation table
    loadAnimationTable();
    
    // Set default animation
    if (!animations.empty()) {
        setAnimation("stand");
    }
    
    std::cout << "Loaded MD2 model with " 
              << header.numFrames << " frames, "
              << header.numVertices << " vertices per frame, "
              << header.numTriangles << " triangles, "
              << animations.size() << " animations" << std::endl;
              
    return true;
}

void Q2Mesh::loadAnimationTable() {
    // Populate animation table from standard Quake 2 animations
    int numAnims = sizeof(animationTable) / sizeof(animationTable[0]);
    for (int i = 0; i < numAnims; i++) {
        Q2Animation anim;
        anim.name = animationTable[i].name;
        anim.startFrame = animationTable[i].firstFrame;
        anim.endFrame = animationTable[i].lastFrame;
        anim.fps = animationTable[i].fps;
        
        // Clamp frames to actual frame count
        if (anim.startFrame >= frames.size()) anim.startFrame = 0;
        if (anim.endFrame >= frames.size()) anim.endFrame = frames.size() - 1;
        
        // Only add valid animations
        if (anim.startFrame <= anim.endFrame) {
            animations[anim.name] = anim;
        }
    }
}

Mesh Q2Mesh::getMeshForFrame(int frameIndex) {
    if (frameIndex < 0 || frameIndex >= frames.size()) {
        std::cerr << "Error: Invalid frame index: " << frameIndex << std::endl;
        return Mesh(); // Return empty mesh
    }
    
    Mesh mesh;
    const Q2Frame& frame = frames[frameIndex];
    
    // Add vertices to mesh
    for (const auto& vertex : frame.vertices) {
        mesh.vertices.push_back({vertex, Colors::WHITE});
    }
    
    // Add triangles to mesh
    for (const auto& tri : triangles) {
        // Create a triangle using the existing Triangle type in your Mesh class
        // We need to update this based on how your Triangle class is defined
        Triangle triangle(tri.vertexIndices[0], tri.vertexIndices[1], tri.vertexIndices[2]);
        mesh.triangles.push_back(triangle);
        
        // Add edges
        mesh.addEdge(tri.vertexIndices[0], tri.vertexIndices[1]);
        mesh.addEdge(tri.vertexIndices[1], tri.vertexIndices[2]);
        mesh.addEdge(tri.vertexIndices[2], tri.vertexIndices[0]);
    }
    
    return mesh;
}

Mesh Q2Mesh::getInterpolatedMesh(int frame1, int frame2, float interp) {
    if (frame1 < 0 || frame1 >= frames.size() || frame2 < 0 || frame2 >= frames.size()) {
        std::cerr << "Error: Invalid frame indices for interpolation" << std::endl;
        return Mesh(); // Return empty mesh
    }
    
    if (interp < 0.0f) interp = 0.0f;
    if (interp > 1.0f) interp = 1.0f;
    
    const Q2Frame& f1 = frames[frame1];
    const Q2Frame& f2 = frames[frame2];
    
    Mesh mesh;
    
    // Interpolate vertices
    for (int i = 0; i < numVertices; i++) {
        Vector3 pos = Utils::lerp(f1.vertices[i], f2.vertices[i], interp);
        mesh.vertices.push_back({pos, Colors::WHITE});
    }
    
    // Add triangles to mesh (same as getMeshForFrame)
    for (const auto& tri : triangles) {
        // Create a triangle using the existing Triangle type in your Mesh class
        Triangle triangle(tri.vertexIndices[0], tri.vertexIndices[1], tri.vertexIndices[2]);
        mesh.triangles.push_back(triangle);
        
        // Add edges
        mesh.addEdge(tri.vertexIndices[0], tri.vertexIndices[1]);
        mesh.addEdge(tri.vertexIndices[1], tri.vertexIndices[2]);
        mesh.addEdge(tri.vertexIndices[2], tri.vertexIndices[0]);
    }
    
    return mesh;
}

void Q2Mesh::setAnimation(const std::string& animName) {
    auto it = animations.find(animName);
    if (it == animations.end()) {
        std::cerr << "Animation not found: " << animName << std::endl;
        return;
    }
    
    currentAnimation = animName;
    const auto& anim = it->second;
    currentFrame = anim.startFrame;
    nextFrame = (currentFrame + 1 <= anim.endFrame) ? currentFrame + 1 : anim.startFrame;
    interpolation = 0.0f;
    animationTime = 0.0f;
    
    std::cout << "Set animation: " << animName 
              << " (frames " << anim.startFrame << "-" << anim.endFrame 
              << ", " << anim.fps << " fps)" << std::endl;
}

void Q2Mesh::updateAnimation(float deltaTime) {
    if (animations.empty() || currentAnimation.empty()) {
        return;
    }
    
    const auto& anim = animations[currentAnimation];
    
    // Update animation time
    animationTime += deltaTime;
    
    // Calculate new interpolation value
    float frameTime = 1.0f / anim.fps;
    interpolation = fmodf(animationTime, frameTime) / frameTime;
    
    // Check if we need to advance to the next frame
    if (animationTime >= frameTime) {
        animationTime = fmodf(animationTime, frameTime);
        
        // Advance frame
        currentFrame = nextFrame;
        nextFrame++;
        
        // Loop back to start if needed
        if (nextFrame > anim.endFrame) {
            nextFrame = anim.startFrame;
        }
    }
}

Mesh Q2Mesh::getCurrentMesh() {
    return getInterpolatedMesh(currentFrame, nextFrame, interpolation);
}