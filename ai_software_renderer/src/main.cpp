#include "Renderer.h"
#include "Matrix.h"
#include "Mesh.h"
#include "Pipeline.h"
#include "Camera.h"
#include <stdio.h>
#include <stdbool.h>
#include <math.h>
#include <vector>
#include "lighting.h"
#include "color.h"
#include "terrain.h"
#include "utils.h" // Make sure to include utils.h for timer
#include "q2mesh.h" // Include Q2Mesh header

// Add a new enum to track rendering modes
enum RenderMode {
    WIREFRAME_ONLY,
    SOLID_ONLY,
    COMBINED
};

int main(int argc, char* argv[]) {
    // Initialize SDL
    if (!initSDL()) {
        printf("Failed to initialize SDL\n");
        return 1;
    }
    
    // Initialize pipeline
    Pipeline pipeline;
    pipeline.initialize();
    
    // Create a camera
    Camera camera(Vector3(0.0f, 1.0f, 5.0f), Vector3(0.0f, 0.0f, 0.0f));
    pipeline.camera = camera;
    
    // Create a lighting system
    LightingSystem lighting;
    
    // Add a directional light (like sunlight)
    DirectionalLight sunlight(
        Vector3(-0.5f, -1.0f, -0.5f), // Direction
        Colors::WARM_YELLOW,          // Color
        0.7f                          // Intensity - reduce from 1.0f to 0.7f
    );
    lighting.addDirectionalLight(sunlight);
    
    // Create a cube mesh
    Mesh cube = Mesh::createCube(1.0f, Colors::BLUE);
    
    // Load a Quake 2 model (you'll need to provide your own MD2 file)
    Q2Mesh q2Model;
    bool modelLoaded = q2Model.loadFromFile("assets/tris.md2");
    
    // Set initial animation
    if (modelLoaded) {
        q2Model.setAnimation("stand");
    }
    
    // Set up render mode
    RenderMode renderMode = COMBINED;
    
    // Game loop variables
    bool running = true;
    SDL_Event event;
    float rotationAngle = 0.0f;
    Utils::Timer frameTimer;
    
    // Print instructions
    printf("Controls:\n");
    printf("  ESC: Quit\n");
    printf("  SPACE: Toggle rendering mode\n");
    printf("  B: Toggle backface culling\n");
    printf("  1: Stand animation\n");
    printf("  2: Run animation\n");
    printf("  3: Attack animation\n");
    printf("  4: Pain animation\n");
    printf("  5: Death animation\n");
    
    // Display initial backface culling state
    printf("Backface Culling: %s\n", pipeline.enableBackfaceCulling ? "ON" : "OFF");
    
    // Main game loop
    while (running) {
        frameTimer.start();
        
        // Process events
        while (SDL_PollEvent(&event)) {
            if (event.type == SDL_QUIT) {
                running = false;
            }
            else if (event.type == SDL_KEYDOWN) {
                switch (event.key.keysym.sym) {
                    case SDLK_ESCAPE:
                        running = false;
                        break;
                    case SDLK_SPACE:
                    {
                        // Toggle rendering mode
                        renderMode = (RenderMode)(((int)renderMode + 1) % 3);
                        // Display current rendering mode
                        const char* modeNames[] = { "Wireframe Only", "Solid Only", "Combined" };
                        printf("Render Mode: %s\n", modeNames[renderMode]);
                        break;
                    }
                    case SDLK_b:
                        // Toggle backface culling
                        pipeline.toggleBackfaceCulling();
                        // Display the new state
                        printf("Backface Culling: %s\n", pipeline.enableBackfaceCulling ? "ON" : "OFF");
                        break;
                    case SDLK_1:
                        if (modelLoaded) q2Model.setAnimation("stand");
                        break;
                    case SDLK_2:
                        if (modelLoaded) q2Model.setAnimation("run");
                        break;
                    case SDLK_3:
                        if (modelLoaded) q2Model.setAnimation("attack");
                        break;
                    case SDLK_4:
                        if (modelLoaded) q2Model.setAnimation("pain1");
                        break;
                    case SDLK_5:
                        if (modelLoaded) q2Model.setAnimation("death1");
                        break;
                }
            }
        }
        
        // Clear the screen
        clearBuffer(Colors::BLACK);
        
        // Update rotation angle
        rotationAngle += 0.02f; // Rotation speed
        
        // HIDE CUBE - Comment out these lines
        /*
        // Create world matrix with rotation
        Matrix4x4 worldMatrix = Matrix4x4::rotationY(rotationAngle) * 
                               Matrix4x4::rotationX(rotationAngle * 0.5f) * 
                               Matrix4x4::translation(0.0f, 0.0f, 0.0f);
        
        // Process the cube through the pipeline
        pipeline.processMesh(cube, worldMatrix);
        
        // Render based on current mode
        switch (renderMode) {
            case WIREFRAME_ONLY:
                pipeline.drawWireframe(Colors::WHITE);
                break;
            case SOLID_ONLY:
                pipeline.drawSolidFlat(lighting);
                break;
            case COMBINED:
                pipeline.drawSolidFlat(lighting);
                pipeline.drawWireframe(Colors::WHITE);
                break;
        }
        */
        
        // Update animation based on elapsed time - multiply by animation speed factor
        float deltaTime = frameTimer.elapsedMilliseconds() / 1000.0f;
        float animationSpeedFactor = 3.0f; // Speed up animations
        if (modelLoaded) {
            q2Model.updateAnimation(deltaTime * animationSpeedFactor);
        }
        
        // FIX MODEL ORIENTATION - Apply rotation to fix the model orientation only, no continuous rotation
        // Update the model orientation to face the camera
        Matrix4x4 q2WorldMatrix = Matrix4x4::rotationY(3*M_PI/2) *    // 270° rotation (90° + 180° to turn it around)
                                  Matrix4x4::rotationX(-M_PI/2) *     // Fix up/down orientation
                                  Matrix4x4::scaling(0.05f, 0.05f, 0.05f) *  
                                  Matrix4x4::translation(0.0f, -0.5f, 0.0f);  // Adjusted Y position
        
        // Get current animated mesh
        if (modelLoaded) {
            Mesh currentMesh = q2Model.getCurrentMesh();
            
            // Process the animated mesh through the pipeline
            pipeline.processMesh(currentMesh, q2WorldMatrix);
            
            // Render based on current mode
            switch (renderMode) {
                case WIREFRAME_ONLY:
                    pipeline.drawWireframe(Colors::WHITE);
                    break;
                case SOLID_ONLY:
                    pipeline.drawSolidFlat(lighting);
                    break;
                case COMBINED:
                    pipeline.drawSolidFlat(lighting);
                    pipeline.drawWireframe(Colors::WHITE);
                    break;
            }
        }
        
        // Render to screen
        renderBuffer();
        
        // Cap the frame rate (optional)
        frameTimer.stop();
        double frameTime = frameTimer.elapsedMilliseconds();
        if (frameTime < 16.667) { // Target ~60 FPS
            SDL_Delay((Uint32)(16.667 - frameTime));
        }
    }
    
    // Clean up
    cleanup();
    return 0;
}