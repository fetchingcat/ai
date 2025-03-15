#pragma once
#include <cstdint>

namespace Colors {
    // Basic colors
    constexpr uint32_t WHITE = 0xFFFFFFFF;
    constexpr uint32_t BLACK = 0x000000FF;
    constexpr uint32_t RED = 0xFF0000FF;
    constexpr uint32_t GREEN = 0x00FF00FF;
    constexpr uint32_t BLUE = 0x0000FFFF;
    
    // Additional colors
    constexpr uint32_t YELLOW = 0xFFFF00FF;
    constexpr uint32_t CYAN = 0x00FFFFFF;
    constexpr uint32_t MAGENTA = 0xFF00FFFF;
    constexpr uint32_t GRAY = 0x808080FF;
    constexpr uint32_t DARK_GRAY = 0x404040FF;
    constexpr uint32_t LIGHT_GRAY = 0xC0C0C0FF;
    
    // Lighting colors
    constexpr uint32_t WARM_YELLOW = 0xFFCC66FF;
    constexpr uint32_t COOL_BLUE = 0x6699CCFF;
    constexpr uint32_t AMBIENT_DARK = 0x202020FF;
    
    // Material colors
    constexpr uint32_t STEEL_BLUE = 0x4682B4FF;
    constexpr uint32_t FOREST_GREEN = 0x228B22FF;
    constexpr uint32_t BRICK_RED = 0xB22222FF;
    constexpr uint32_t GOLD = 0xFFD700FF;
    constexpr uint32_t SILVER = 0xC0C0C0FF;
    constexpr uint32_t BRONZE = 0xCD7F32FF;
    
    // Terrain colors
    constexpr uint32_t GRASS_GREEN = 0x7CFC00FF;
    constexpr uint32_t DIRT_BROWN = 0x8B4513FF;
    constexpr uint32_t SAND = 0xF4A460FF;
    constexpr uint32_t SNOW = 0xFFFAFAFF;
    constexpr uint32_t WATER_BLUE = 0x1E90FFFF;

    // Ship colors
    constexpr uint32_t SHIP_BODY = 0x778899FF;    // Light slate gray
    constexpr uint32_t SHIP_WINGS = 0x708090FF;   // Slate gray
    constexpr uint32_t SHIP_COCKPIT = 0x4682B4FF; // Steel blue
}