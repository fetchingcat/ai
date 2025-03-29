cmake_minimum_required(VERSION 3.10)
project(SoftwareRenderer)

set(CMAKE_CXX_STANDARD 11)

# SDL2 setup
set(SDL2_PATH "C:/SDL2" CACHE PATH "Path to SDL2 installation")
include_directories("${SDL2_PATH}/include")

# Include our own header directory
include_directories(${CMAKE_SOURCE_DIR}/include)

# Define source files
set(SOURCES 
    src/main.cpp
    src/renderer.cpp
    src/matrix.cpp
    src/mesh.cpp
    src/pipeline.cpp
    src/camera.cpp
    src/lighting.cpp
    src/terrain.cpp
    src/utils.cpp
    src/q2mesh.cpp
)

# Define header files
set(HEADERS
    include/renderer.h
    include/matrix.h
    include/mesh.h
    include/pipeline.h
    include/camera.h
    include/lighting.h
    include/terrain.h
    include/utils.h
    include/q2mesh.h
)

# Create main executable
add_executable(SoftwareRenderer ${SOURCES} ${HEADERS})

# Link against SDL2 libraries
if(MSVC)
    target_link_libraries(SoftwareRenderer
        "${SDL2_PATH}/lib/x64/SDL2.lib"
        "${SDL2_PATH}/lib/x64/SDL2main.lib"
    )
    
    # Add Windows-specific settings
    target_link_options(SoftwareRenderer PRIVATE "/SUBSYSTEM:CONSOLE")
    
    # Copy SDL2.dll to output directory
    add_custom_command(TARGET SoftwareRenderer POST_BUILD
        COMMAND ${CMAKE_COMMAND} -E copy_if_different
            "${SDL2_PATH}/lib/x64/SDL2.dll"
            $<TARGET_FILE_DIR:SoftwareRenderer>)
else()
    # Linux/Mac configuration (just in case)
    find_package(SDL2 REQUIRED)
    target_link_libraries(SoftwareRenderer ${SDL2_LIBRARIES})
endif()