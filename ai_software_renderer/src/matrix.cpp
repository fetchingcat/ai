#include "Matrix.h"
#include <sstream>
#include <iomanip>
#include <cmath>

// Vector3 methods
std::string Vector3::toString() const {
    std::stringstream ss;
    ss << "Vector3(" << x << ", " << y << ", " << z << ")";
    return ss.str();
}

// Vector4 methods
std::string Vector4::toString() const {
    std::stringstream ss;
    ss << "Vector4(" << x << ", " << y << ", " << z << ", " << w << ")";
    return ss.str();
}

// Matrix4x4 methods
Matrix4x4::Matrix4x4() {
    // Identity matrix
    for (int i = 0; i < 4; i++) {
        for (int j = 0; j < 4; j++) {
            m[i][j] = (i == j) ? 1.0f : 0.0f;
        }
    }
}

Matrix4x4::Matrix4x4(
    float m00, float m01, float m02, float m03,
    float m10, float m11, float m12, float m13,
    float m20, float m21, float m22, float m23,
    float m30, float m31, float m32, float m33
) {
    m[0][0] = m00; m[0][1] = m01; m[0][2] = m02; m[0][3] = m03;
    m[1][0] = m10; m[1][1] = m11; m[1][2] = m12; m[1][3] = m13;
    m[2][0] = m20; m[2][1] = m21; m[2][2] = m22; m[2][3] = m23;
    m[3][0] = m30; m[3][1] = m31; m[3][2] = m32; m[3][3] = m33;
}

Matrix4x4 Matrix4x4::operator*(const Matrix4x4& other) const {
    Matrix4x4 result;
    
    for (int i = 0; i < 4; i++) {
        for (int j = 0; j < 4; j++) {
            result.m[i][j] = 0.0f;
            for (int k = 0; k < 4; k++) {
                result.m[i][j] += m[i][k] * other.m[k][j];
            }
        }
    }
    
    return result;
}

Vector4 Matrix4x4::transform(const Vector4& v) const {
    Vector4 result;
    result.x = m[0][0] * v.x + m[0][1] * v.y + m[0][2] * v.z + m[0][3] * v.w;
    result.y = m[1][0] * v.x + m[1][1] * v.y + m[1][2] * v.z + m[1][3] * v.w;
    result.z = m[2][0] * v.x + m[2][1] * v.y + m[2][2] * v.z + m[2][3] * v.w;
    result.w = m[3][0] * v.x + m[3][1] * v.y + m[3][2] * v.z + m[3][3] * v.w;
    return result;
}

Vector3 Matrix4x4::transform(const Vector3& v) const {
    Vector4 v4(v, 1.0f);
    Vector4 result = transform(v4);
    return result.toVector3(); // This does perspective division if needed
}

Matrix4x4 Matrix4x4::transpose() const {
    Matrix4x4 result;
    for (int i = 0; i < 4; i++) {
        for (int j = 0; j < 4; j++) {
            result.m[i][j] = m[j][i];
        }
    }
    return result;
}

// Calculate determinant of 3x3 submatrix
static float det3x3(
    float a00, float a01, float a02,
    float a10, float a11, float a12,
    float a20, float a21, float a22
) {
    return a00 * (a11 * a22 - a12 * a21) -
           a01 * (a10 * a22 - a12 * a20) +
           a02 * (a10 * a21 - a11 * a20);
}

Matrix4x4 Matrix4x4::inverse() const {
    // Calculate cofactors and determinant
    float c00 = det3x3(m[1][1], m[1][2], m[1][3], m[2][1], m[2][2], m[2][3], m[3][1], m[3][2], m[3][3]);
    float c01 = -det3x3(m[1][0], m[1][2], m[1][3], m[2][0], m[2][2], m[2][3], m[3][0], m[3][2], m[3][3]);
    float c02 = det3x3(m[1][0], m[1][1], m[1][3], m[2][0], m[2][1], m[2][3], m[3][0], m[3][1], m[3][3]);
    float c03 = -det3x3(m[1][0], m[1][1], m[1][2], m[2][0], m[2][1], m[2][2], m[3][0], m[3][1], m[3][2]);

    float c10 = -det3x3(m[0][1], m[0][2], m[0][3], m[2][1], m[2][2], m[2][3], m[3][1], m[3][2], m[3][3]);
    float c11 = det3x3(m[0][0], m[0][2], m[0][3], m[2][0], m[2][2], m[2][3], m[3][0], m[3][2], m[3][3]);
    float c12 = -det3x3(m[0][0], m[0][1], m[0][3], m[2][0], m[2][1], m[2][3], m[3][0], m[3][1], m[3][3]);
    float c13 = det3x3(m[0][0], m[0][1], m[0][2], m[2][0], m[2][1], m[2][2], m[3][0], m[3][1], m[3][2]);

    float c20 = det3x3(m[0][1], m[0][2], m[0][3], m[1][1], m[1][2], m[1][3], m[3][1], m[3][2], m[3][3]);
    float c21 = -det3x3(m[0][0], m[0][2], m[0][3], m[1][0], m[1][2], m[1][3], m[3][0], m[3][2], m[3][3]);
    float c22 = det3x3(m[0][0], m[0][1], m[0][3], m[1][0], m[1][1], m[1][3], m[3][0], m[3][1], m[3][3]);
    float c23 = -det3x3(m[0][0], m[0][1], m[0][2], m[1][0], m[1][1], m[1][2], m[3][0], m[3][1], m[3][2]);

    float c30 = -det3x3(m[0][1], m[0][2], m[0][3], m[1][1], m[1][2], m[1][3], m[2][1], m[2][2], m[2][3]);
    float c31 = det3x3(m[0][0], m[0][2], m[0][3], m[1][0], m[1][2], m[1][3], m[2][0], m[2][2], m[2][3]);
    float c32 = -det3x3(m[0][0], m[0][1], m[0][3], m[1][0], m[1][1], m[1][3], m[2][0], m[2][1], m[2][3]);
    float c33 = det3x3(m[0][0], m[0][1], m[0][2], m[1][0], m[1][1], m[1][2], m[2][0], m[2][1], m[2][2]);

    // Calculate determinant using the cofactors of the first row
    float det = m[0][0] * c00 + m[0][1] * c01 + m[0][2] * c02 + m[0][3] * c03;

    // Check for singular matrix (non-invertible)
    if (std::abs(det) < 1e-6f) {
        return identity(); // Return identity matrix as fallback
    }

    // Calculate inverse matrix
    float invDet = 1.0f / det;
    
    return Matrix4x4(
        c00 * invDet, c10 * invDet, c20 * invDet, c30 * invDet,
        c01 * invDet, c11 * invDet, c21 * invDet, c31 * invDet,
        c02 * invDet, c12 * invDet, c22 * invDet, c32 * invDet,
        c03 * invDet, c13 * invDet, c23 * invDet, c33 * invDet
    );
}

Matrix4x4 Matrix4x4::identity() {
    return Matrix4x4(
        1.0f, 0.0f, 0.0f, 0.0f,
        0.0f, 1.0f, 0.0f, 0.0f,
        0.0f, 0.0f, 1.0f, 0.0f,
        0.0f, 0.0f, 0.0f, 1.0f
    );
}

Matrix4x4 Matrix4x4::translation(float x, float y, float z) {
    return Matrix4x4(
        1.0f, 0.0f, 0.0f, x,
        0.0f, 1.0f, 0.0f, y,
        0.0f, 0.0f, 1.0f, z,
        0.0f, 0.0f, 0.0f, 1.0f
    );
}

Matrix4x4 Matrix4x4::translation(const Vector3& v) {
    return translation(v.x, v.y, v.z);
}

Matrix4x4 Matrix4x4::scaling(float x, float y, float z) {
    return Matrix4x4(
        x,    0.0f, 0.0f, 0.0f,
        0.0f, y,    0.0f, 0.0f,
        0.0f, 0.0f, z,    0.0f,
        0.0f, 0.0f, 0.0f, 1.0f
    );
}

Matrix4x4 Matrix4x4::scaling(float scale) {
    return scaling(scale, scale, scale);
}

Matrix4x4 Matrix4x4::rotationX(float radians) {
    float c = std::cos(radians);
    float s = std::sin(radians);
    
    return Matrix4x4(
        1.0f, 0.0f, 0.0f, 0.0f,
        0.0f, c,    -s,   0.0f,
        0.0f, s,    c,    0.0f,
        0.0f, 0.0f, 0.0f, 1.0f
    );
}

Matrix4x4 Matrix4x4::rotationY(float radians) {
    float c = std::cos(radians);
    float s = std::sin(radians);
    
    return Matrix4x4(
        c,    0.0f, s,    0.0f,
        0.0f, 1.0f, 0.0f, 0.0f,
        -s,   0.0f, c,    0.0f,
        0.0f, 0.0f, 0.0f, 1.0f
    );
}

Matrix4x4 Matrix4x4::rotationZ(float radians) {
    float c = std::cos(radians);
    float s = std::sin(radians);
    
    return Matrix4x4(
        c,    -s,   0.0f, 0.0f,
        s,    c,    0.0f, 0.0f,
        0.0f, 0.0f, 1.0f, 0.0f,
        0.0f, 0.0f, 0.0f, 1.0f
    );
}

Matrix4x4 Matrix4x4::lookAt(const Vector3& eye, const Vector3& target, const Vector3& up) {
    // Calculate camera coordinate axes
    Vector3 zAxis = (eye - target).normalize(); // Forward
    Vector3 xAxis = up.cross(zAxis).normalize(); // Right
    Vector3 yAxis = zAxis.cross(xAxis); // Up
    
    // Create view matrix (rotation + translation)
    Matrix4x4 view(
        xAxis.x, xAxis.y, xAxis.z, -xAxis.dot(eye),
        yAxis.x, yAxis.y, yAxis.z, -yAxis.dot(eye),
        zAxis.x, zAxis.y, zAxis.z, -zAxis.dot(eye),
        0.0f,    0.0f,    0.0f,    1.0f
    );
    
    return view;
}

Matrix4x4 Matrix4x4::perspective(float fovY, float aspect, float zNear, float zFar) {
    float tanHalfFovY = std::tan(fovY / 2.0f);
    float f = 1.0f / tanHalfFovY;
    
    return Matrix4x4(
        f / aspect, 0.0f, 0.0f, 0.0f,
        0.0f, f, 0.0f, 0.0f,
        0.0f, 0.0f, (zFar + zNear) / (zNear - zFar), (2.0f * zFar * zNear) / (zNear - zFar),
        0.0f, 0.0f, -1.0f, 0.0f
    );
}

Matrix4x4 Matrix4x4::orthographic(float left, float right, float bottom, float top, float near, float far) {
    float width = right - left;
    float height = top - bottom;
    float depth = far - near;
    
    return Matrix4x4(
        2.0f / width, 0.0f, 0.0f, -(right + left) / width,
        0.0f, 2.0f / height, 0.0f, -(top + bottom) / height,
        0.0f, 0.0f, -2.0f / depth, -(far + near) / depth,
        0.0f, 0.0f, 0.0f, 1.0f
    );
}

std::string Matrix4x4::toString() const {
    std::stringstream ss;
    ss << "Matrix4x4[\n";
    for (int i = 0; i < 4; i++) {
        ss << "  [";
        for (int j = 0; j < 4; j++) {
            ss << std::setw(8) << std::fixed << std::setprecision(4) << m[i][j];
            if (j < 3) ss << ", ";
        }
        ss << "]\n";
    }
    ss << "]";
    return ss.str();
}