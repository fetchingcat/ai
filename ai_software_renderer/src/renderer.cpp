#include "renderer.h"
#include <stdlib.h>
#include <stdio.h>
#include <algorithm>

// Pixel buffer that we'll render to
static uint32_t* pixelBuffer = NULL;

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
    
    return true;
}

void clearBuffer(uint32_t color) {
    for (int i = 0; i < WINDOW_WIDTH * WINDOW_HEIGHT; i++) {
        pixelBuffer[i] = color;
    }
}

void putPixel(int x, int y, uint32_t color) {
    if (x >= 0 && x < WINDOW_WIDTH && y >= 0 && y < WINDOW_HEIGHT) {
        pixelBuffer[y * WINDOW_WIDTH + x] = color;
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