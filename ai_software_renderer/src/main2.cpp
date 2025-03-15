
#include "Renderer.h"
#include <stdio.h>
#include <stdbool.h>
#include <math.h>
#include <vector>

int main(int argc, char* args[]) {
    if (!initSDL()) {
        cleanup();
        return 1;
    }
    
    bool quit = false;
    SDL_Event e;
    
    int frame = 0;
    
    // Main loop
    while (!quit) {
        // Handle events
        while (SDL_PollEvent(&e) != 0) {
            if (e.type == SDL_QUIT) {
                quit = true;
            }
        }
        
        // Clear to dark background
        clearBuffer(0xFF181818);
        
        frame++;
        float time = frame * 0.01f;
        
        // Draw various shapes to showcase the rendering capabilities
        
        // 1. Draw a polygon
        std::vector<SDL_Point> hexagon;
        int hexSize = 80;
        int hexX = 150;
        int hexY = 150;
        
        for (int i = 0; i < 6; i++) {
            float angle = i * M_PI / 3.0f;
            hexagon.push_back({
                hexX + (int)(hexSize * cos(angle + time * 0.3f)),
                hexY + (int)(hexSize * sin(angle + time * 0.3f))
            });
        }
        drawFilledPolygon(hexagon, 0xFF4285F4); // Google blue
        drawPolygon(hexagon, 0xFFFFFFFF);
        
        // 2. Draw circles
        int circleX = 400;
        int circleY = 150;
        int circleRadius = 70;
        drawFilledCircle(circleX, circleY, circleRadius, 0xFF34A853); // Google green
        drawCircle(circleX, circleY, circleRadius, 0xFFFFFFFF);
        
        // Orbit small circles around the main circle
        for (int i = 0; i < 8; i++) {
            float orbitAngle = time + i * M_PI / 4.0f;
            int smallX = circleX + (int)((circleRadius + 20) * cos(orbitAngle));
            int smallY = circleY + (int)((circleRadius + 20) * sin(orbitAngle));
            drawFilledCircle(smallX, smallY, 10, 0xFFEA4335); // Google red
        }
        
        // 3. Draw rounded rectangles
        int rectX = 100;
        int rectY = 300;
        int rectW = 200;
        int rectH = 120;
        int rectRadius = 30;
        drawFilledRoundedRect(rectX, rectY, rectW, rectH, rectRadius, 0xFFFBBC05); // Google yellow
        drawRoundedRect(rectX, rectY, rectW, rectH, rectRadius, 0xFFFFFFFF);
        
        // 4. Draw an animated triangle fan
        int fanCenterX = 500;
        int fanCenterY = 360;
        int fanRadius = 80;
        int fanSegments = 12;
        
        for (int i = 0; i < fanSegments; i++) {
            float startAngle = time + i * 2.0f * M_PI / fanSegments;
            float endAngle = time + (i + 1) * 2.0f * M_PI / fanSegments;
            
            int x1 = fanCenterX;
            int y1 = fanCenterY;
            int x2 = fanCenterX + (int)(fanRadius * cos(startAngle));
            int y2 = fanCenterY + (int)(fanRadius * sin(startAngle));
            int x3 = fanCenterX + (int)(fanRadius * cos(endAngle));
            int y3 = fanCenterY + (int)(fanRadius * sin(endAngle));
            
            // Alternative Google colors
            uint32_t colors[] = {
                0xFF4285F4, // Blue
                0xFF34A853, // Green
                0xFFFBBC05, // Yellow
                0xFFEA4335  // Red
            };
            
            drawFilledTriangle(x1, y1, x2, y2, x3, y3, colors[i % 4]);
        }
        
        // Draw a frame counter
        char buffer[32];
        sprintf(buffer, "Frame: %d", frame);
        
        // Render to screen
        renderBuffer();
        
        // Small delay to avoid hogging CPU
        SDL_Delay(16); // ~60fps
    }
    
    cleanup();
    return 0;
}
