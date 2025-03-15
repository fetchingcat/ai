#ifndef MATRIX_H
#define MATRIX_H

#include <cmath>
#include <vector>
#include <string>

// Vector3 class for 3D points and vectors
class Vector3 {
public:
    float x, y, z;
    
    Vector3() : x(0.0f), y(0.0f), z(0.0f) {}
    Vector3(float x, float y, float z) : x(x), y(y), z(z) {}
    
    // Vector operations
    Vector3 operator+(const Vector3& v) const { return Vector3(x + v.x, y + v.y, z + v.z); }
    Vector3 operator-(const Vector3& v) const { return Vector3(x - v.x, y - v.y, z - v.z); }
    Vector3 operator*(float scalar) const { return Vector3(x * scalar, y * scalar, z * scalar); }
    Vector3 operator/(float scalar) const { return Vector3(x / scalar, y / scalar, z / scalar); }
    
    // Dot product
    float dot(const Vector3& v) const { return x * v.x + y * v.y + z * v.z; }
    
    // Cross product
    Vector3 cross(const Vector3& v) const {
        return Vector3(
            y * v.z - z * v.y,
            z * v.x - x * v.z,
            x * v.y - y * v.x
        );
    }
    
    // Length of vector
    float length() const { return std::sqrt(x * x + y * y + z * z); }
    
    // Normalize vector
    Vector3 normalize() const {
        float len = length();
        if (len < 1e-6f) return Vector3(0, 0, 0); // Avoid division by zero
        return Vector3(x / len, y / len, z / len);
    }
    
    // To string for debugging
    std::string toString() const;
};

// Vector4 class for homogeneous coordinates and colors with alpha
class Vector4 {
public:
    float x, y, z, w;
    
    Vector4() : x(0.0f), y(0.0f), z(0.0f), w(1.0f) {}
    Vector4(float x, float y, float z, float w) : x(x), y(y), z(z), w(w) {}
    explicit Vector4(const Vector3& v, float w = 1.0f) : x(v.x), y(v.y), z(v.z), w(w) {}
    
    // Convert back to Vector3 (with perspective division if w != 1)
    Vector3 toVector3() const { 
        if (std::abs(w) > 1e-6f) return Vector3(x / w, y / w, z / w);
        return Vector3(x, y, z);
    }
    
    std::string toString() const;
};

// 4x4 Matrix class for transformations
class Matrix4x4 {
public:
    // Matrix data in row-major order
    float m[4][4];
    
    // Default constructor - identity matrix
    Matrix4x4();
    
    // Constructor from 16 values (row-major order)
    Matrix4x4(
        float m00, float m01, float m02, float m03,
        float m10, float m11, float m12, float m13,
        float m20, float m21, float m22, float m23,
        float m30, float m31, float m32, float m33
    );
    
    // Matrix multiplication
    Matrix4x4 operator*(const Matrix4x4& other) const;
    
    // Transform a Vector4 by this matrix
    Vector4 transform(const Vector4& v) const;
    
    // Transform a Vector3 by this matrix (converts to Vector4 with w=1 internally)
    Vector3 transform(const Vector3& v) const;
    
    // Get the transpose of this matrix
    Matrix4x4 transpose() const;
    
    // Get the inverse of this matrix
    Matrix4x4 inverse() const;
    
    // Create transformation matrices
    static Matrix4x4 identity();
    static Matrix4x4 translation(float x, float y, float z);
    static Matrix4x4 translation(const Vector3& v);
    static Matrix4x4 scaling(float x, float y, float z);
    static Matrix4x4 scaling(float scale);
    static Matrix4x4 rotationX(float radians);
    static Matrix4x4 rotationY(float radians);
    static Matrix4x4 rotationZ(float radians);
    
    // Create view matrix (camera)
    static Matrix4x4 lookAt(const Vector3& eye, const Vector3& target, const Vector3& up);
    
    // Create projection matrices
    static Matrix4x4 perspective(float fovY, float aspect, float zNear, float zFar);
    static Matrix4x4 orthographic(float left, float right, float bottom, float top, float near, float far);
        
    // Debug string representation
    std::string toString() const;
};

#endif // MATRIX_H