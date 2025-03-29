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
    COMBINED,
    TEXTURED     // New texture rendering mode
};

// Create a simple checkered texture without using SDL_image
SDL_Surface* createCrateTexture() {
    // Create surface
    const int TEXTURE_SIZE = 256;
    SDL_Surface* surface = SDL_CreateRGBSurface(0, TEXTURE_SIZE, TEXTURE_SIZE, 32, 
                                               0xFF000000, 0x00FF0000, 0x0000FF00, 0x000000FF);
    if (!surface) {
        printf("Surface could not be created! SDL Error: %s\n", SDL_GetError());
        return NULL;
    }
    
    // Lock the surface for direct access to the pixels
    SDL_LockSurface(surface);
    
    // Get pointer to pixel data
    Uint32* pixels = (Uint32*)surface->pixels;
    
    // Draw a checkered pattern
    const int CHECKER_SIZE = 32; // Size of each checker square
    const Uint32 DARK_BROWN = 0xFF804000;  // RGB: 128, 64, 0
    const Uint32 LIGHT_BROWN = 0xFFA06030; // RGB: 160, 96, 48
    
    // Add wood grain texture by varying the colors slightly
    for (int y = 0; y < TEXTURE_SIZE; y++) {
        for (int x = 0; x < TEXTURE_SIZE; x++) {
            // Determine base color from checker pattern
            int checkerX = (x / CHECKER_SIZE) % 2;
            int checkerY = (y / CHECKER_SIZE) % 2;
            Uint32 baseColor = (checkerX ^ checkerY) ? DARK_BROWN : LIGHT_BROWN;
            
            // Add some wood grain variation
            int grainVariation = ((x * 7) % 13) - 6; // Pseudo-random variation
            
            // Extract color components
            Uint8 r = (baseColor >> 16) & 0xFF;
            Uint8 g = (baseColor >> 8) & 0xFF;
            Uint8 b = baseColor & 0xFF;
            
            // Apply variation (clamped to valid range)
            r = (Uint8)SDL_max(0, SDL_min(255, (int)r + grainVariation));
            g = (Uint8)SDL_max(0, SDL_min(255, (int)g + grainVariation));
            b = (Uint8)SDL_max(0, SDL_min(255, (int)b + grainVariation));
            
            // Add border lines
            if (x % CHECKER_SIZE == 0 || y % CHECKER_SIZE == 0) {
                r = (Uint8)(r * 0.8); // Darken for borders
                g = (Uint8)(g * 0.8);
                b = (Uint8)(b * 0.8);
            }
            
            // Combine back into RGBA
            Uint32 color = (0xFF << 24) | (r << 16) | (g << 8) | b;
            
            // Set pixel
            pixels[y * surface->w + x] = color;
        }
    }
    
    // Unlock surface
    SDL_UnlockSurface(surface);
    
    return surface;
}

// Simpler createColorTexture helper
SDL_Surface* createColorTexture(Uint32 color) {
    // Create surface
    const int TEXTURE_SIZE = 64;
    SDL_Surface* surface = SDL_CreateRGBSurface(0, TEXTURE_SIZE, TEXTURE_SIZE, 32, 
                                               0xFF000000, 0x00FF0000, 0x0000FF00, 0x000000FF);
    if (!surface) {
        printf("Surface could not be created! SDL Error: %s\n", SDL_GetError());
        return NULL;
    }
    
    // Lock the surface for direct access to the pixels
    SDL_LockSurface(surface);
    
    // Get pointer to pixel data
    Uint32* pixels = (Uint32*)surface->pixels;
    
    // Fill with solid color
    for (int i = 0; i < TEXTURE_SIZE * TEXTURE_SIZE; i++) {
        pixels[i] = color;
    }
    
    // Unlock surface
    SDL_UnlockSurface(surface);
    
    return surface;
}

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
    
    // Create a cube mesh with texture coordinates
    Mesh texCube = Mesh::createTexturedCube(1.0f, Colors::WHITE);
    
    // Create texture at runtime
    SDL_Surface* cubeTexture = createCrateTexture();
    if (!cubeTexture) {
        printf("Failed to create texture. Falling back to color texture.\n");
        cubeTexture = createColorTexture(Colors::BLUE);
        
        if (!cubeTexture) {
            printf("Could not create any texture. Using default colored cube.\n");
        }
    }
    
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
    printf("  SPACE: Toggle rendering mode (Wireframe, Solid, Combined, Textured)\n");
    printf("  B: Toggle backface culling\n");
    printf("  1: Stand animation\n");
    printf("  2: Run animation\n");
    printf("  3: Attack animation\n");
    printf("  4: Pain animation\n");
    printf("  5: Death animation\n");
    printf("  T: Show textured cube\n");
    printf("  M: Show model\n");
    
    // Display initial backface culling state
    printf("Backface Culling: %s\n", pipeline.enableBackfaceCulling ? "ON" : "OFF");
    
    // Flag to switch between model and cube
    bool showModel = true;
    
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
                        renderMode = (RenderMode)(((int)renderMode + 1) % 4); // Now includes textured mode
                        // Display current rendering mode
                        const char* modeNames[] = { "Wireframe Only", "Solid Only", "Combined", "Textured" };
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
                    case SDLK_t:
                        showModel = false;
                        printf("Showing textured cube\n");
                        break;
                    case SDLK_m:
                        showModel = true;
                        printf("Showing model\n");
                        break;
                }
            }
        }
        
        // Clear the screen
        clearBuffer(Colors::BLACK);
        
        // Update rotation angle
        rotationAngle += 0.02f; // Rotation speed
        
        if (!showModel && cubeTexture) {
            // Create world matrix with rotation for the textured cube
            Matrix4x4 texCubeWorldMatrix = Matrix4x4::rotationY(rotationAngle) * 
                                         Matrix4x4::rotationX(rotationAngle * 0.5f) * 
                                         Matrix4x4::translation(0.0f, 0.0f, 0.0f);
            
            // Process the textured cube through the pipeline
            pipeline.processMesh(texCube, texCubeWorldMatrix);
            
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
                case TEXTURED:
                    pipeline.drawTextured(cubeTexture);
                    break;
            }
        }
        else if (modelLoaded) {
            // Update animation based on elapsed time - multiply by animation speed factor
            float deltaTime = frameTimer.elapsedMilliseconds() / 1000.0f;
            float animationSpeedFactor = 3.0f; // Speed up animations
            q2Model.updateAnimation(deltaTime * animationSpeedFactor);
            
            // Update the model orientation to face the camera
            Matrix4x4 q2WorldMatrix = Matrix4x4::rotationY(3*M_PI/2) *    // 270° rotation (90° + 180° to turn it around)
                                    Matrix4x4::rotationX(-M_PI/2) *     // Fix up/down orientation
                                    Matrix4x4::scaling(0.05f, 0.05f, 0.05f) *  
                                    Matrix4x4::translation(0.0f, -0.5f, 0.0f);  // Adjusted Y position
            
            // Get current animated mesh
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
                case TEXTURED:
                    // Fallback to solid rendering since model doesn't have texture coordinates
                    pipeline.drawSolidFlat(lighting);
                    break;
            }
        }
        
        // Render to screen
        renderBuffer();
        
        // Frame rate control and calculation
        frameTimer.stop();
        float fps = 1000.0f / frameTimer.elapsedMilliseconds();
        
        // Print FPS every 60 frames
        static int frameCount = 0;
        if (++frameCount >= 60) {
            printf("FPS: %.2f\n", fps);
            frameCount = 0;
        }
        
        // Limit to approximately 60 FPS
        if (frameTimer.elapsedMilliseconds() < 16) {
            SDL_Delay(16 - (Uint32)frameTimer.elapsedMilliseconds());
        }
    }
    
    // Clean up textures
    if (cubeTexture) {
        SDL_FreeSurface(cubeTexture);
    }
    
    // Cleanup SDL resources
    cleanup();
    
    return 0;
}