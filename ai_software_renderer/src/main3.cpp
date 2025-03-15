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

// Structure to store ship position and orientation in the fleet
struct ShipInfo {
    Vector3 position;
    float rotation;
    float verticalOffset;
    uint32_t color;
};

// Add a new enum to track rendering modes
enum RenderMode {
    WIREFRAME_ONLY,
    SOLID_ONLY,
    COMBINED
};

int main(int argc, char* args[]) {
    if (!initSDL()) {
        return 1;
    }
    
    // Create a spaceship mesh with white color for body and wings
    Mesh shipMesh = Mesh::createSpaceship(0.3f, Colors::SHIP_BODY, Colors::SHIP_WINGS);
    
    // Create the rendering pipeline
    Pipeline pipeline;
    pipeline.initialize();
    
    // Create lighting system
    LightingSystem lighting;
    
    // Add directional light (sun)
    DirectionalLight sun(Vector3(0.5f, -0.8f, -0.2f), Colors::WARM_YELLOW, 0.8f);
    lighting.addDirectionalLight(sun);
    
    // Add a cool blue "ambient" light as a directional light from below
    DirectionalLight fillLight(Vector3(0.0f, 0.5f, 0.0f), Colors::COOL_BLUE, 0.2f);
    lighting.addDirectionalLight(fillLight);
    
    // Update the initial point light to bright green with smaller radius
    PointLight movingLight(Vector3(0.0f, 2.0f, 0.0f), Colors::BRIGHT_GREEN, 5.0f, 6.0f);
    lighting.addPointLight(movingLight);
    
    // Grid size for the fleet
    const int GRID_SIZE = 20;
    const float GRID_SPACING = 0.8f;
    const float FORMATION_SIZE = GRID_SIZE * GRID_SPACING;
    
    // Create a fleet of ships arranged in a grid
    std::vector<ShipInfo> fleet;
    
    // Generate the fleet in a grid with some randomness
    for (int z = 0; z < GRID_SIZE; z++) {
        for (int x = 0; x < GRID_SIZE; x++) {
            ShipInfo ship;
            
            // Calculate position in grid
            float xPos = (x - GRID_SIZE/2.0f) * GRID_SPACING;
            float zPos = (z - GRID_SIZE/2.0f) * GRID_SPACING;
            
            // Add slight random variation to position and rotation
            float randOffsetX = (rand() % 100) / 500.0f - 0.1f;
            float randOffsetZ = (rand() % 100) / 500.0f - 0.1f;
            float randRotation = (rand() % 100) / 100.0f * 0.2f - 0.1f;
            
            ship.position = Vector3(xPos + randOffsetX, 0.0f, zPos + randOffsetZ);
            ship.rotation = randRotation;
            
            // Create vertical wave pattern based on position
            ship.verticalOffset = sinf(x * 0.3f) * cosf(z * 0.3f) * 0.5f;
            
            // All ships are white
            ship.color = Colors::WHITE;
            
            fleet.push_back(ship);
        }
    }
    
    // Set up camera at starting position
    pipeline.camera = Camera(Vector3(0.0f, 5.0f, FORMATION_SIZE), Vector3(0.0f, 0.0f, 0.0f));
    
    bool quit = false;
    SDL_Event e;
    
    int frame = 0;
    float cameraDistance = FORMATION_SIZE * 1.2f;  // Increased from 0.75f to 1.2f
    float cameraHeight = FORMATION_SIZE * 0.2f;    // Decreased from 0.3f to 0.2f
    
    // Camera control variables
    float baseVerticalAngle = 0.8f;  // Starting vertical angle
    const float angleStep = 0.05f;   // How much to change per key press
    
    // Add a render mode variable with combined as the default
    RenderMode currentRenderMode = COMBINED;
    
    // Add a variable to track light height offset
    float lightYOffset = 0.0f;
    const float lightMoveStep = 0.2f; // How much to move per key press
    
    // Main loop
    while (!quit) {
        // Handle events
        while (SDL_PollEvent(&e) != 0) {
            if (e.type == SDL_QUIT) {
                quit = true;
            }
            // Handle mouse wheel events for zoom
            else if (e.type == SDL_MOUSEWHEEL) {
                // Adjust camera distance based on scroll direction
                if (e.wheel.y > 0) {
                    // Scroll up - zoom in
                    cameraDistance *= 0.9f;  // Reduce distance by 10%
                }
                else if (e.wheel.y < 0) {
                    // Scroll down - zoom out
                    cameraDistance *= 1.1f;  // Increase distance by 10%
                }
                
                // Clamp camera distance to reasonable values
                cameraDistance = std::max(FORMATION_SIZE * 0.5f, std::min(FORMATION_SIZE * 3.0f, cameraDistance));
            }
            // Handle keyboard events
            else if (e.type == SDL_KEYDOWN) {
                switch (e.key.keysym.sym) {
                    case SDLK_UP:
                        // Move view more top-down
                        baseVerticalAngle -= angleStep;
                        break;
                    case SDLK_DOWN:
                        // Move view more side-on
                        baseVerticalAngle += angleStep;
                        break;
                    case SDLK_r:
                        // Reset view to default
                        baseVerticalAngle = 0.8f;
                        cameraDistance = FORMATION_SIZE * 1.2f;
                        break;
                    case SDLK_w:
                        // Toggle rendering mode
                        currentRenderMode = static_cast<RenderMode>((currentRenderMode + 1) % 3);
                        break;
                    case SDLK_u:
                        // Move light up
                        lightYOffset += lightMoveStep;
                        break;
                    case SDLK_d:
                        // Move light down
                        lightYOffset -= lightMoveStep;
                        break;
                }
                
                // Clamp vertical angle to reasonable values
                baseVerticalAngle = std::max(0.1f, std::min(1.5f, baseVerticalAngle));
            }
        }
        
        // Clear to pure black background - use 0x00000000 for RGBA or 0xFF000000 for ARGB
        clearBuffer(Colors::BLACK);
        
        frame++;
        float time = frame * 0.01f;
        
        // Move ships in a subtle wave pattern
        float waveTime = time * 0.2f;
        
        // Create cinematic camera path
        float horizontalAngle = time * 0.15f;  // Slow horizontal orbit
        
        // Change the vertical angle to be much more side-on
        // (using the user-adjusted baseVerticalAngle)
        float verticalAngle = baseVerticalAngle + 0.05f * sinf(time * 0.1f);  // Increased base angle from 0.5f to 0.8f
                                                                 // Reduced oscillation from 0.1f to 0.05f
        
        // Distance also changes slightly, moving in and out
        float dynamicDistance = cameraDistance * (1.0f + 0.1f * sinf(time * 0.05f));
        
        // In the main loop, update the point light position
        // Update the moving light position - make it move within the fleet
        movingLight.position.x = (GRID_SIZE/4.0f) * sinf(waveTime * 0.5f);
        movingLight.position.z = (GRID_SIZE/4.0f) * cosf(waveTime * 0.5f);
        // Add subtle vertical movement to the light plus manual offset
        movingLight.position.y = 0.2f + 0.3f * sinf(waveTime * 0.7f) + lightYOffset;
        movingLight.color = Colors::BRIGHT_GREEN;  // Make it bright green for testing
        
        // Remove the first point light and add the updated one
        if (!lighting.pointLights.empty()) {
            lighting.pointLights.clear();
        }
        lighting.addPointLight(movingLight);
        
        // Update camera position with cinematic orbit
        pipeline.camera.orbit(horizontalAngle, verticalAngle, dynamicDistance);
        
        // Draw all spaceships in the fleet
        for (const ShipInfo& ship : fleet) {
            // Calculate vertical motion based on wave pattern and ship's offset
            float verticalPos = ship.verticalOffset + 0.15f * sinf(waveTime + ship.position.x * 0.5f + ship.position.z * 0.5f);
            
            // Create transformation matrix for this ship
            Matrix4x4 shipTransform = 
                Matrix4x4::translation(ship.position.x, verticalPos, ship.position.z) *
                Matrix4x4::rotationY(ship.rotation + waveTime * 0.02f);
            
            // Process and draw the ship
            pipeline.processMesh(shipMesh, shipTransform);
            
            // Draw according to current render mode
            switch (currentRenderMode) {
                case WIREFRAME_ONLY:
                    // Only wireframe
                    pipeline.drawWireframe(Colors::WIREFRAME);  // Solid white
                    break;
                    
                case SOLID_ONLY:
                    // Only flat shading
                    pipeline.drawSolidFlat(lighting);
                    break;
                    
                case COMBINED:
                    // Both flat shading and wireframe
                    pipeline.drawSolidFlat(lighting);
                    pipeline.drawWireframe(Colors::SEMI_TRANSPARENT_WHITE);  // Semi-transparent white wireframe
                    break;
            }
        }
        
        // After drawing all ships, add this visualization for the point light:
        // Draw a simple marker for the point light position
        Matrix4x4 lightMarkerTransform = Matrix4x4::translation(
            movingLight.position.x, 
            movingLight.position.y, 
            movingLight.position.z
        ) * Matrix4x4::scaling(0.5f, 0.5f, 0.5f);

        // Create a small cube mesh for the light if you have the Mesh::createCube method
        Mesh lightMesh = Mesh::createCube(0.5f, Colors::LIGHT_MARKER);  // Same color as the light
        pipeline.processMesh(lightMesh, lightMarkerTransform);
        pipeline.drawWireframe(Colors::LIGHT_MARKER);  // Draw in light's color
        
        // Display the rendered scene
        renderBuffer();
    }
    
    // Clean up
    cleanup();
    return 0;
}
