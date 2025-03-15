#include "camera.h"
#include "renderer.h"
#include <cmath>

Camera::Camera()
    : position(0.0f, 0.0f, 5.0f),
      target(0.0f, 0.0f, 0.0f),
      up(0.0f, 1.0f, 0.0f),
      fov(60.0f * 3.14159f / 180.0f), // 60 degrees in radians
      aspectRatio(WINDOW_WIDTH / (float)WINDOW_HEIGHT),
      nearPlane(0.1f),
      farPlane(100.0f)
{
    updateMatrices();
}

Camera::Camera(const Vector3& position, const Vector3& target)
    : position(position),
      target(target),
      up(0.0f, 1.0f, 0.0f),
      fov(60.0f * 3.14159f / 180.0f), // 60 degrees in radians
      aspectRatio(WINDOW_WIDTH / (float)WINDOW_HEIGHT),
      nearPlane(0.1f),
      farPlane(100.0f)
{
    updateMatrices();
}

void Camera::updateMatrices() {
    viewMatrix = Matrix4x4::lookAt(position, target, up);
    projectionMatrix = Matrix4x4::perspective(fov, aspectRatio, nearPlane, farPlane);
}

void Camera::orbit(float horizontalAngle, float verticalAngle, float distance) {
    // Constrain vertical angle to avoid camera flipping
    verticalAngle = std::max(-85.0f * 3.14159f / 180.0f, 
                            std::min(85.0f * 3.14159f / 180.0f, verticalAngle));
    
    // Calculate new position based on spherical coordinates
    float x = distance * std::sin(verticalAngle) * std::sin(horizontalAngle);
    float y = distance * std::cos(verticalAngle);
    float z = distance * std::sin(verticalAngle) * std::cos(horizontalAngle);
    
    // Update camera position relative to target
    position.x = target.x + x;
    position.y = target.y + y;
    position.z = target.z + z;
    
    // Update matrices
    updateMatrices();
}

void Camera::setPosition(const Vector3& newPosition) {
    position = newPosition;
    updateMatrices();
}

void Camera::lookAt(const Vector3& newTarget) {
    target = newTarget;
    updateMatrices();
}