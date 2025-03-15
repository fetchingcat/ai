#ifndef CAMERA_H
#define CAMERA_H

#include "Matrix.h"

// Represents a 3D camera
class Camera {
public:
    Vector3 position;
    Vector3 target;
    Vector3 up;
    
    float fov;
    float aspectRatio;
    float nearPlane;
    float farPlane;
    
    Matrix4x4 viewMatrix;
    Matrix4x4 projectionMatrix;
    
    // Constructor with default values
    Camera();
    
    // Constructor with position and target
    Camera(const Vector3& position, const Vector3& target);
    
    // Update the view and projection matrices
    void updateMatrices();
    
    // Rotate camera around target (orbit camera)
    void orbit(float horizontalAngle, float verticalAngle, float distance);
    
    // Move camera to a new position
    void setPosition(const Vector3& newPosition);
    
    // Look at a specific point
    void lookAt(const Vector3& newTarget);
};

#endif // CAMERA_H