#ifndef RENDERER_H
#define RENDERER_H

#include <SDL.h>
#include <stdint.h>
#include <stdbool.h>
#include <vector> // Add this for polygon vertices

#define WINDOW_WIDTH 800
#define WINDOW_HEIGHT 600

// Initialize SDL and create window, renderer, and pixel buffer
bool initSDL();

// Clear the pixel buffer to a specific color
void clearBuffer(uint32_t color);

// Set a specific pixel to a color
void putPixel(int x, int y, uint32_t color);

// Add this declaration:
void setPixel(int x, int y, uint32_t color);

// Draw a line using Bresenham's algorithm
void drawLine(int x1, int y1, int x2, int y2, uint32_t color);

// Draw a filled triangle
void drawFilledTriangle(int x1, int y1, int x2, int y2, int x3, int y3, uint32_t color);

// Add this function declaration
void fillTriangle(int x1, int y1, int x2, int y2, int x3, int y3, uint32_t color);

// Draw a polygon outline
void drawPolygon(const std::vector<SDL_Point>& vertices, uint32_t color);

// Draw a filled polygon
void drawFilledPolygon(const std::vector<SDL_Point>& vertices, uint32_t color);

// Draw a circle outline
void drawCircle(int centerX, int centerY, int radius, uint32_t color);

// Draw a filled circle
void drawFilledCircle(int centerX, int centerY, int radius, uint32_t color);

// Draw a rounded rectangle
void drawRoundedRect(int x, int y, int width, int height, int radius, uint32_t color);

// Draw a filled rounded rectangle
void drawFilledRoundedRect(int x, int y, int width, int height, int radius, uint32_t color);

// Update texture with pixel buffer and present to screen
void renderBuffer();

// Free all resources
void cleanup();

#endif // RENDERER_H