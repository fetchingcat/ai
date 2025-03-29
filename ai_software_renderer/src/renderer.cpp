#include "renderer.h"
#include <stdlib.h>
#include <stdio.h>
#include <algorithm>
#include <limits>
#include <cmath>

// Pixel buffer that we'll render to
static uint32_t* pixelBuffer = NULL;

// Z-buffer (depth buffer)
static float* depthBuffer = NULL;

// SDL window and renderer
static SDL_Window* window = NULL;
static SDL_Renderer* renderer = NULL;
static SDL_Texture* texture = NULL;

bool initSDL() {
    if (SDL_Init(SDL_INIT_VIDEO) < 0) {
        printf("SDL could not initialize! SDL_Error: %s\n", SDL_GetError());
        return false;
    }
    
    window = SDL_CreateWindow("Software Renderer", 
                              SDL_WINDOWPOS_CENTERED, SDL_WINDOWPOS_CENTERED, 
                              WINDOW_WIDTH, WINDOW_HEIGHT, 
                              SDL_WINDOW_SHOWN);
    
    if (!window) {
        printf("Window could not be created! SDL_Error: %s\n", SDL_GetError());
        return false;
    }
    
    renderer = SDL_CreateRenderer(window, -1, SDL_RENDERER_ACCELERATED);
    if (!renderer) {
        printf("Renderer could not be created! SDL_Error: %s\n", SDL_GetError());
        return false;
    }
    
    texture = SDL_CreateTexture(renderer, 
                               SDL_PIXELFORMAT_RGBA8888, 
                               SDL_TEXTUREACCESS_STREAMING, 
                               WINDOW_WIDTH, WINDOW_HEIGHT);
    
    if (!texture) {
        printf("Texture could not be created! SDL_Error: %s\n", SDL_GetError());
        return false;
    }
    
    // Allocate pixel buffer
    pixelBuffer = (uint32_t*)malloc(WINDOW_WIDTH * WINDOW_HEIGHT * sizeof(uint32_t));
    if (!pixelBuffer) {
        printf("Failed to allocate pixel buffer!\n");
        return false;
    }
    
    // Allocate depth buffer
    depthBuffer = (float*)malloc(WINDOW_WIDTH * WINDOW_HEIGHT * sizeof(float));
    if (!depthBuffer) {
        printf("Failed to allocate depth buffer!\n");
        free(pixelBuffer);
        return false;
    }
    
    // Clear the depth buffer
    clearDepthBuffer();
    
    return true;
}

void clearBuffer(uint32_t color) {
    for (int i = 0; i < WINDOW_WIDTH * WINDOW_HEIGHT; i++) {
        pixelBuffer[i] = color;
    }
}

void clearDepthBuffer() {
    // Initialize depth buffer to maximum depth (far away)
    // Note: In our coordinate system, smaller Z values are farther from camera
    // Adjust the range based on your coordinate system
    for (int i = 0; i < WINDOW_WIDTH * WINDOW_HEIGHT; i++) {
        depthBuffer[i] = std::numeric_limits<float>::infinity();
    }
}

void putPixel(int x, int y, uint32_t color) {
    if (x >= 0 && x < WINDOW_WIDTH && y >= 0 && y < WINDOW_HEIGHT) {
        pixelBuffer[y * WINDOW_WIDTH + x] = color;
    }
}

void putPixelWithDepth(int x, int y, float z, uint32_t color) {
    if (x >= 0 && x < WINDOW_WIDTH && y >= 0 && y < WINDOW_HEIGHT) {
        int index = y * WINDOW_WIDTH + x;
        
        // Only draw the pixel if it's closer (smaller z) than what's already there
        if (z < depthBuffer[index]) {
            pixelBuffer[index] = color;
            depthBuffer[index] = z;
        }
    }
}

void drawLine(int x1, int y1, int x2, int y2, uint32_t color) {
    // Bresenham's line algorithm
    int dx = abs(x2 - x1);
    int dy = -abs(y2 - y1);
    int sx = x1 < x2 ? 1 : -1;
    int sy = y1 < y2 ? 1 : -1;
    int err = dx + dy;
    int e2;
    
    while (true) {
        putPixel(x1, y1, color);
        if (x1 == x2 && y1 == y2) break;
        e2 = 2 * err;
        if (e2 >= dy) {
            if (x1 == x2) break;
            err += dy;
            x1 += sx;
        }
        if (e2 <= dx) {
            if (y1 == y2) break;
            err += dx;
            y1 += sy;
        }
    }
}

void drawFilledTriangle(int x1, int y1, int x2, int y2, int x3, int y3, uint32_t color) {
    // Sort vertices by y-coordinate (y1 <= y2 <= y3)
    if (y1 > y2) {
        std::swap(y1, y2);
        std::swap(x1, x2);
    }
    if (y2 > y3) {
        std::swap(y2, y3);
        std::swap(x2, x3);
    }
    if (y1 > y2) {
        std::swap(y1, y2);
        std::swap(x1, x2);
    }
    
    // Flat bottom triangle
    if (y2 == y3) {
        // Sort x2 and x3 so that x2 is on the left
        if (x2 > x3) std::swap(x2, x3);
        
        for (int y = y1; y <= y2; y++) {
            float alpha = (float)(y - y1) / (float)(y2 - y1);
            int x_start = x1 + alpha * (x2 - x1);
            int x_end = x1 + alpha * (x3 - x1);
            
            if (x_start > x_end) std::swap(x_start, x_end);
            
            for (int x = x_start; x <= x_end; x++) {
                putPixel(x, y, color);
            }
        }
    }
    // Flat top triangle
    else if (y1 == y2) {
        // Sort x1 and x2 so that x1 is on the left
        if (x1 > x2) std::swap(x1, x2);
        
        for (int y = y1; y <= y3; y++) {
            float alpha = (float)(y - y1) / (float)(y3 - y1);
            int x_start = x1 + alpha * (x3 - x1);
            int x_end = x2 + alpha * (x3 - x2);
            
            if (x_start > x_end) std::swap(x_start, x_end);
            
            for (int x = x_start; x <= x_end; x++) {
                putPixel(x, y, color);
            }
        }
    }
    // General case - split into flat bottom and flat top triangles
    else {
        int x4 = x1 + ((float)(y2 - y1) / (float)(y3 - y1)) * (x3 - x1);
        
        // Draw flat bottom triangle (top half)
        for (int y = y1; y <= y2; y++) {
            float alpha = (y - y1) / (float)(y2 - y1);
            int x_start = x1 + alpha * (x2 - x1);
            int x_end = x1 + alpha * (x4 - x1);
            
            if (x_start > x_end) std::swap(x_start, x_end);
            
            for (int x = x_start; x <= x_end; x++) {
                putPixel(x, y, color);
            }
        }
        
        // Draw flat top triangle (bottom half)
        for (int y = y2; y <= y3; y++) {
            float alpha = (y - y2) / (float)(y3 - y2);
            int x_start = x2 + alpha * (x3 - x2);
            int x_end = x4 + alpha * (x3 - x4);
            
            if (x_start > x_end) std::swap(x_start, x_end);
            
            for (int x = x_start; x <= x_end; x++) {
                putPixel(x, y, color);
            }
        }
    }
}

void renderBuffer() {
    SDL_UpdateTexture(texture, NULL, pixelBuffer, WINDOW_WIDTH * sizeof(uint32_t));
    SDL_RenderClear(renderer);
    SDL_RenderCopy(renderer, texture, NULL, NULL);
    SDL_RenderPresent(renderer);
}

void cleanup() {
    if (pixelBuffer) free(pixelBuffer);
    if (depthBuffer) free(depthBuffer);
    if (texture) SDL_DestroyTexture(texture);
    if (renderer) SDL_DestroyRenderer(renderer);
    if (window) SDL_DestroyWindow(window);
    SDL_Quit();
}

// Add after drawFilledTriangle function

void drawPolygon(const std::vector<SDL_Point>& vertices, uint32_t color) {
    if (vertices.size() < 2) {
        return;
    }
    
    // Draw lines connecting each vertex
    for (size_t i = 0; i < vertices.size() - 1; i++) {
        drawLine(vertices[i].x, vertices[i].y, vertices[i + 1].x, vertices[i + 1].y, color);
    }
    
    // Connect the last vertex back to the first
    drawLine(vertices.back().x, vertices.back().y, vertices.front().x, vertices.front().y, color);
}

void drawFilledPolygon(const std::vector<SDL_Point>& vertices, uint32_t color) {
    if (vertices.size() < 3) {
        return;
    }
    
    // Simple triangle fan approach - works well for convex polygons
    SDL_Point center = {0, 0};
    
    // Calculate center point (average of all vertices)
    for (const auto& vertex : vertices) {
        center.x += vertex.x;
        center.y += vertex.y;
    }
    center.x /= vertices.size();
    center.y /= vertices.size();
    
    // Draw triangles from center to each edge
    for (size_t i = 0; i < vertices.size(); i++) {
        size_t next = (i + 1) % vertices.size();
        drawFilledTriangle(
            center.x, center.y,
            vertices[i].x, vertices[i].y, 
            vertices[next].x, vertices[next].y,
            color
        );
    }
}

void drawCircle(int centerX, int centerY, int radius, uint32_t color) {
    // Midpoint circle algorithm
    int x = radius;
    int y = 0;
    int err = 0;
    
    while (x >= y) {
        putPixel(centerX + x, centerY + y, color);
        putPixel(centerX + y, centerY + x, color);
        putPixel(centerX - y, centerY + x, color);
        putPixel(centerX - x, centerY + y, color);
        putPixel(centerX - x, centerY - y, color);
        putPixel(centerX - y, centerY - x, color);
        putPixel(centerX + y, centerY - x, color);
        putPixel(centerX + x, centerY - y, color);
        
        if (err <= 0) {
            y += 1;
            err += 2*y + 1;
        }
        
        if (err > 0) {
            x -= 1;
            err -= 2*x + 1;
        }
    }
}

void drawFilledCircle(int centerX, int centerY, int radius, uint32_t color) {
    // Scan line approach for filled circle
    for (int y = -radius; y <= radius; y++) {
        // Calculate width at this height
        int width = (int)sqrt(radius * radius - y * y);
        
        // Draw horizontal line
        for (int x = -width; x <= width; x++) {
            putPixel(centerX + x, centerY + y, color);
        }
    }
}

void drawRoundedRect(int x, int y, int width, int height, int radius, uint32_t color) {
    // Draw horizontal lines
    drawLine(x + radius, y, x + width - radius, y, color);
    drawLine(x + radius, y + height, x + width - radius, y + height, color);
    
    // Draw vertical lines
    drawLine(x, y + radius, x, y + height - radius, color);
    drawLine(x + width, y + radius, x + width, y + height - radius, color);
    
    // Draw corners
    // Top-left
    int centerX = x + radius;
    int centerY = y + radius;
    int lastX = -1, lastY = -1;
    
    for (int i = 90; i <= 180; i++) {
        float angle = i * 3.14159f / 180.0f;
        int dx = (int)(radius * cos(angle));
        int dy = (int)(radius * sin(angle));
        
        int nextX = centerX + dx;
        int nextY = centerY - dy;
        
        if (nextX != lastX || nextY != lastY) {
            putPixel(nextX, nextY, color);
            lastX = nextX;
            lastY = nextY;
        }
    }
    
    // Top-right
    centerX = x + width - radius;
    centerY = y + radius;
    lastX = -1;
    lastY = -1;
    
    for (int i = 0; i <= 90; i++) {
        float angle = i * 3.14159f / 180.0f;
        int dx = (int)(radius * cos(angle));
        int dy = (int)(radius * sin(angle));
        
        int nextX = centerX + dx;
        int nextY = centerY - dy;
        
        if (nextX != lastX || nextY != lastY) {
            putPixel(nextX, nextY, color);
            lastX = nextX;
            lastY = nextY;
        }
    }
    
    // Bottom-right
    centerX = x + width - radius;
    centerY = y + height - radius;
    lastX = -1;
    lastY = -1;
    
    for (int i = 0; i <= 90; i++) {
        float angle = i * 3.14159f / 180.0f;
        int dx = (int)(radius * cos(angle));
        int dy = (int)(radius * sin(angle));
        
        int nextX = centerX + dx;
        int nextY = centerY + dy;
        
        if (nextX != lastX || nextY != lastY) {
            putPixel(nextX, nextY, color);
            lastX = nextX;
            lastY = nextY;
        }
    }
    
    // Bottom-left
    centerX = x + radius;
    centerY = y + height - radius;
    lastX = -1;
    lastY = -1;
    
    for (int i = 90; i <= 180; i++) {
        float angle = i * 3.14159f / 180.0f;
        int dx = (int)(radius * cos(angle));
        int dy = (int)(radius * sin(angle));
        
        int nextX = centerX - dx;
        int nextY = centerY + dy;
        
        if (nextX != lastX || nextY != lastY) {
            putPixel(nextX, nextY, color);
            lastX = nextX;
            lastY = nextY;
        }
    }
}

void drawFilledRoundedRect(int x, int y, int width, int height, int radius, uint32_t color) {
    // Fill the main rectangle
    for (int cy = y + radius; cy <= y + height - radius; cy++) {
        for (int cx = x; cx <= x + width; cx++) {
            putPixel(cx, cy, color);
        }
    }
    
    // Fill top rectangle
    for (int cy = y; cy < y + radius; cy++) {
        for (int cx = x + radius; cx <= x + width - radius; cx++) {
            putPixel(cx, cy, color);
        }
    }
    
    // Fill bottom rectangle
    for (int cy = y + height - radius + 1; cy <= y + height; cy++) {
        for (int cx = x + radius; cx <= x + width - radius; cx++) {
            putPixel(cx, cy, color);
        }
    }
    
    // Fill the four corners
    // Top-left
    int centerX = x + radius;
    int centerY = y + radius;
    
    for (int cy = y; cy <= y + radius; cy++) {
        for (int cx = x; cx <= x + radius; cx++) {
            int dx = cx - centerX;
            int dy = cy - centerY;
            int distSqr = dx * dx + dy * dy;
            
            if (distSqr <= radius * radius) {
                putPixel(cx, cy, color);
            }
        }
    }
    
    // Top-right
    centerX = x + width - radius;
    centerY = y + radius;
    
    for (int cy = y; cy <= y + radius; cy++) {
        for (int cx = x + width - radius; cx <= x + width; cx++) {
            int dx = cx - centerX;
            int dy = cy - centerY;
            int distSqr = dx * dx + dy * dy;
            
            if (distSqr <= radius * radius) {
                putPixel(cx, cy, color);
            }
        }
    }
    
    // Bottom-right
    centerX = x + width - radius;
    centerY = y + height - radius;
    
    for (int cy = y + height - radius; cy <= y + height; cy++) {
        for (int cx = x + width - radius; cx <= x + width; cx++) {
            int dx = cx - centerX;
            int dy = cy - centerY;
            int distSqr = dx * dx + dy * dy;
            
            if (distSqr <= radius * radius) {
                putPixel(cx, cy, color);
            }
        }
    }
    
    // Bottom-left
    centerX = x + radius;
    centerY = y + height - radius;
    
    for (int cy = y + height - radius; cy <= y + height; cy++) {
        for (int cx = x; cx <= x + radius; cx++) {
            int dx = cx - centerX;
            int dy = cy - centerY;
            int distSqr = dx * dx + dy * dy;
            
            if (distSqr <= radius * radius) {
                putPixel(cx, cy, color);
            }
        }
    }
}

// Add this implementation
void fillTriangle(int x1, int y1, int x2, int y2, int x3, int y3, uint32_t color) {
    // Sort the vertices by y-coordinate (top to bottom)
    if (y1 > y2) {
        std::swap(x1, x2);
        std::swap(y1, y2);
    }
    if (y1 > y3) {
        std::swap(x1, x3);
        std::swap(y1, y3);
    }
    if (y2 > y3) {
        std::swap(x2, x3);
        std::swap(y2, y3);
    }
    
    // Compute slopes
    float slope1 = (y3 - y1) != 0 ? (float)(x3 - x1) / (y3 - y1) : 0;
    float slope2 = (y2 - y1) != 0 ? (float)(x2 - x1) / (y2 - y1) : 0;
    float slope3 = (y3 - y2) != 0 ? (float)(x3 - x2) / (y3 - y2) : 0;
    
    // Draw the triangle in two parts (top and bottom)
    float x_start, x_end;
    
    // First half (top)
    if (y2 > y1) {
        for (int y = y1; y < y2; y++) {
            x_start = x1 + (y - y1) * slope1;
            x_end = x1 + (y - y1) * slope2;
            
            if (x_start > x_end) {
                std::swap(x_start, x_end);
            }
            
            for (int x = (int)x_start; x <= (int)x_end; x++) {
                setPixel(x, y, color);
            }
        }
    }
    
    // Second half (bottom)
    if (y3 > y2) {
        for (int y = y2; y <= y3; y++) {
            x_start = x1 + (y - y1) * slope1;
            x_end = x2 + (y - y2) * slope3;
            
            if (x_start > x_end) {
                std::swap(x_start, x_end);
            }
            
            for (int x = (int)x_start; x <= (int)x_end; x++) {
                setPixel(x, y, color);
            }
        }
    }
}

// Add this implementation:
void setPixel(int x, int y, uint32_t color) {
    // Make sure the pixel is within the screen bounds
    if (x >= 0 && x < WINDOW_WIDTH && y >= 0 && y < WINDOW_HEIGHT) {
        // Set the pixel in our buffer
        pixelBuffer[y * WINDOW_WIDTH + x] = color;
    }
}

// Implement new 3D line drawing with z-interpolation
void drawLine3D(int x1, int y1, float z1, int x2, int y2, float z2, uint32_t color) {
    // Bresenham's line algorithm with z-interpolation
    int dx = abs(x2 - x1);
    int dy = -abs(y2 - y1);
    int sx = x1 < x2 ? 1 : -1;
    int sy = y1 < y2 ? 1 : -1;
    int err = dx + dy;
    int e2;
    
    // Calculate total steps for interpolation
    float totalSteps = std::max(dx, -dy);
    float currentStep = 0;
    
    while (true) {
        // For wireframe rendering, we want the lines to be always visible
        // regardless of depth, so we use putPixel instead of putPixelWithDepth
        if (x1 >= 0 && x1 < WINDOW_WIDTH && y1 >= 0 && y1 < WINDOW_HEIGHT) {
            putPixel(x1, y1, color);
        }
        
        if (x1 == x2 && y1 == y2) break;
        e2 = 2 * err;
        if (e2 >= dy) {
            if (x1 == x2) break;
            err += dy;
            x1 += sx;
            currentStep += 1;
        }
        if (e2 <= dx) {
            if (y1 == y2) break;
            err += dx;
            y1 += sy;
            currentStep += 1;
        }
    }
}

// Add 3D triangle drawing with depth interpolation
void drawFilledTriangle3D(
    int x1, int y1, float z1,
    int x2, int y2, float z2,
    int x3, int y3, float z3,
    uint32_t color) {
    
    // Sort vertices by y-coordinate (y1 <= y2 <= y3)
    if (y1 > y2) {
        std::swap(y1, y2);
        std::swap(x1, x2);
        std::swap(z1, z2);
    }
    if (y2 > y3) {
        std::swap(y2, y3);
        std::swap(x2, x3);
        std::swap(z2, z3);
    }
    if (y1 > y2) {
        std::swap(y1, y2);
        std::swap(x1, x2);
        std::swap(z1, z2);
    }
    
    // Calculate triangle area for barycentric coordinates
    float area = (x1 * (y2 - y3) + x2 * (y3 - y1) + x3 * (y1 - y2)) / 2.0f;
    
    // Flat bottom triangle
    if (y2 == y3) {
        // Sort x2 and x3 so that x2 is on the left
        if (x2 > x3) {
            std::swap(x2, x3);
            std::swap(z2, z3);
        }
        
        for (int y = y1; y <= y2; y++) {
            float alpha = (float)(y - y1) / (float)(y2 - y1);
            int x_start = x1 + alpha * (x2 - x1);
            int x_end = x1 + alpha * (x3 - x1);
            
            // Interpolate z at the edges
            float z_start = z1 + alpha * (z2 - z1);
            float z_end = z1 + alpha * (z3 - z1);
            
            if (x_start > x_end) {
                std::swap(x_start, x_end);
                std::swap(z_start, z_end);
            }
            
            for (int x = x_start; x <= x_end; x++) {
                // Interpolate z along the scanline
                float t = (x_end == x_start) ? 0 : (float)(x - x_start) / (float)(x_end - x_start);
                float z = z_start + t * (z_end - z_start);
                
                // Depth test when drawing
                putPixelWithDepth(x, y, z, color);
            }
        }
    }
    // Flat top triangle
    else if (y1 == y2) {
        // Sort x1 and x2 so that x1 is on the left
        if (x1 > x2) {
            std::swap(x1, x2);
            std::swap(z1, z2);
        }
        
        for (int y = y1; y <= y3; y++) {
            float alpha = (float)(y - y1) / (float)(y3 - y1);
            int x_start = x1 + alpha * (x3 - x1);
            int x_end = x2 + alpha * (x3 - x2);
            
            // Interpolate z at the edges
            float z_start = z1 + alpha * (z3 - z1);
            float z_end = z2 + alpha * (z3 - z2);
            
            if (x_start > x_end) {
                std::swap(x_start, x_end);
                std::swap(z_start, z_end);
            }
            
            for (int x = x_start; x <= x_end; x++) {
                // Interpolate z along the scanline
                float t = (x_end == x_start) ? 0 : (float)(x - x_start) / (float)(x_end - x_start);
                float z = z_start + t * (z_end - z_start);
                
                // Depth test when drawing
                putPixelWithDepth(x, y, z, color);
            }
        }
    }
    // General case - split into flat bottom and flat top triangles
    else {
        int x4 = x1 + ((float)(y2 - y1) / (float)(y3 - y1)) * (x3 - x1);
        float z4 = z1 + ((float)(y2 - y1) / (float)(y3 - y1)) * (z3 - z1);
        
        // Draw flat bottom triangle (top half)
        for (int y = y1; y <= y2; y++) {
            float alpha = (y - y1) / (float)(y2 - y1);
            int x_start = x1 + alpha * (x2 - x1);
            int x_end = x1 + alpha * (x4 - x1);
            
            // Interpolate z at the edges
            float z_start = z1 + alpha * (z2 - z1);
            float z_end = z1 + alpha * (z4 - z1);
            
            if (x_start > x_end) {
                std::swap(x_start, x_end);
                std::swap(z_start, z_end);
            }
            
            for (int x = x_start; x <= x_end; x++) {
                // Interpolate z along the scanline
                float t = (x_end == x_start) ? 0 : (float)(x - x_start) / (float)(x_end - x_start);
                float z = z_start + t * (z_end - z_start);
                
                // Depth test when drawing
                putPixelWithDepth(x, y, z, color);
            }
        }
        
        // Draw flat top triangle (bottom half)
        for (int y = y2; y <= y3; y++) {
            float alpha = (y - y2) / (float)(y3 - y2);
            int x_start = x2 + alpha * (x3 - x2);
            int x_end = x4 + alpha * (x3 - x4);
            
            // Interpolate z at the edges
            float z_start = z2 + alpha * (z3 - z2);
            float z_end = z4 + alpha * (z3 - z4);
            
            if (x_start > x_end) {
                std::swap(x_start, x_end);
                std::swap(z_start, z_end);
            }
            
            for (int x = x_start; x <= x_end; x++) {
                // Interpolate z along the scanline
                float t = (x_end == x_start) ? 0 : (float)(x - x_start) / (float)(x_end - x_start);
                float z = z_start + t * (z_end - z_start);
                
                // Depth test when drawing
                putPixelWithDepth(x, y, z, color);
            }
        }
    }
}

// Implement textured triangle drawing with z-buffer
void drawTexturedTriangle3D(
    int x1, int y1, float z1, float u1, float v1,
    int x2, int y2, float z2, float u2, float v2,
    int x3, int y3, float z3, float u3, float v3,
    SDL_Surface* texture) {
    
    // Sort vertices by y-coordinate (y1 <= y2 <= y3)
    if (y1 > y2) {
        std::swap(y1, y2); std::swap(x1, x2); std::swap(z1, z2);
        std::swap(u1, u2); std::swap(v1, v2);
    }
    if (y2 > y3) {
        std::swap(y2, y3); std::swap(x2, x3); std::swap(z2, z3);
        std::swap(u2, u3); std::swap(v2, v3);
    }
    if (y1 > y2) {
        std::swap(y1, y2); std::swap(x1, x2); std::swap(z1, z2);
        std::swap(u1, u2); std::swap(v1, v2);
    }
    
    // Calculate edges
    int totalHeight = y3 - y1;
    
    // First half of the triangle (flat bottom)
    for (int y = y1; y <= y2; y++) {
        int segmentHeight = y2 - y1;
        float alpha = segmentHeight == 0 ? 0 : (float)(y - y1) / segmentHeight;
        
        // Edge x points and interpolated values
        float x_a = x1 + (x2 - x1) * alpha;
        float x_b = x1 + (x3 - x1) * ((float)(y - y1) / totalHeight);
        
        // Interpolate z, u, v for the edges
        float z_a = z1 + (z2 - z1) * alpha;
        float z_b = z1 + (z3 - z1) * ((float)(y - y1) / totalHeight);
        
        float u_a = u1 + (u2 - u1) * alpha;
        float u_b = u1 + (u3 - u1) * ((float)(y - y1) / totalHeight);
        
        float v_a = v1 + (v2 - v1) * alpha;
        float v_b = v1 + (v3 - v1) * ((float)(y - y1) / totalHeight);
        
        // Ensure x_a is on the left
        if (x_a > x_b) {
            std::swap(x_a, x_b); std::swap(z_a, z_b);
            std::swap(u_a, u_b); std::swap(v_a, v_b);
        }
        
        // Draw horizontal line
        for (int x = (int)x_a; x <= (int)x_b; x++) {
            float t = (x_b == x_a) ? 0 : (float)(x - x_a) / (x_b - x_a);
            
            // Interpolate values along the scanline
            float z = z_a + (z_b - z_a) * t;
            float u = u_a + (u_b - u_a) * t;
            float v = v_a + (v_b - v_a) * t;
            
            // Sample texture (with bounds checking)
            int tx = (int)(u * texture->w);
            int ty = (int)(v * texture->h);
            
            // Clamp texture coordinates
            tx = std::max(0, std::min(tx, texture->w - 1));
            ty = std::max(0, std::min(ty, texture->h - 1));
            
            // Get pixel from texture
            uint32_t* pixels = (uint32_t*)texture->pixels;
            uint32_t color = pixels[ty * texture->w + tx];
            
            // Draw pixel with depth test
            putPixelWithDepth(x, y, z, color);
        }
    }
    
    // Second half of the triangle (flat top)
    for (int y = y2; y <= y3; y++) {
        int segmentHeight = y3 - y2;
        float alpha = segmentHeight == 0 ? 0 : (float)(y - y2) / segmentHeight;
        
        // Edge x points and interpolated values
        float x_a = x2 + (x3 - x2) * alpha;
        float x_b = x1 + (x3 - x1) * ((float)(y - y1) / totalHeight);
        
        // Interpolate z, u, v for the edges
        float z_a = z2 + (z3 - z2) * alpha;
        float z_b = z1 + (z3 - z1) * ((float)(y - y1) / totalHeight);
        
        float u_a = u2 + (u3 - u2) * alpha;
        float u_b = u1 + (u3 - u1) * ((float)(y - y1) / totalHeight);
        
        float v_a = v2 + (v3 - v2) * alpha;
        float v_b = v1 + (v3 - v1) * ((float)(y - y1) / totalHeight);
        
        // Ensure x_a is on the left
        if (x_a > x_b) {
            std::swap(x_a, x_b); std::swap(z_a, z_b);
            std::swap(u_a, u_b); std::swap(v_a, v_b);
        }
        
        // Draw horizontal line
        for (int x = (int)x_a; x <= (int)x_b; x++) {
            float t = (x_b == x_a) ? 0 : (float)(x - x_a) / (x_b - x_a);
            
            // Interpolate values along the scanline
            float z = z_a + (z_b - z_a) * t;
            float u = u_a + (u_b - u_a) * t;
            float v = v_a + (v_b - v_a) * t;
            
            // Sample texture (with bounds checking)
            int tx = (int)(u * texture->w);
            int ty = (int)(v * texture->h);
            
            // Clamp texture coordinates
            tx = std::max(0, std::min(tx, texture->w - 1));
            ty = std::max(0, std::min(ty, texture->h - 1));
            
            // Get pixel from texture
            uint32_t* pixels = (uint32_t*)texture->pixels;
            uint32_t color = pixels[ty * texture->w + tx];
            
            // Draw pixel with depth test
            putPixelWithDepth(x, y, z, color);
        }
    }
}