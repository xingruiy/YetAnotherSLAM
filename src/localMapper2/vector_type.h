#ifndef FUSION_MATH_TYPES_H
#define FUSION_MATH_TYPES_H

#include <cmath>
#include <sophus/se3.hpp>

using uchar = unsigned char;

//! 2-dimensional vector type
template <class T>
struct Vector2
{
    __host__ __device__ inline Vector2() : x(0), y(0) {}
    __host__ __device__ inline Vector2(const Vector2<T> &other) : x(other.x), y(other.y) {}
    __host__ __device__ inline Vector2(T x, T y) : x(x), y(y) {}
    __host__ __device__ inline Vector2(T val) : x(val), y(val) {}

    __host__ __device__ inline Vector2<T> operator+(const Vector2<T> &other) const
    {
        return Vector2<T>(x + other.x, y + other.y);
    }

    __host__ __device__ inline Vector2<T> operator-(const Vector2<T> &other) const
    {
        return Vector2<T>(x - other.x, y - other.y);
    }

    __host__ __device__ inline T dot(const Vector2<T> &V) const
    {
        return x * V.x + y * V.y;
    }

    __host__ __device__ inline T operator*(const Vector2<T> &other) const
    {
        return x * other.x + y * other.y;
    }

    __host__ __device__ inline Vector2<T> operator/(const T val) const
    {
        return Vector2<T>(x / val, y / val);
    }

    T x, y;
};

//! 3-dimensional vector type
template <class T>
struct Vector3
{
    __host__ __device__ inline Vector3() : x(0), y(0), z(0) {}
    __host__ __device__ inline Vector3(const Vector3<T> &other) : x(other.x), y(other.y), z(other.z) {}
    __host__ __device__ inline Vector3(T x, T y, T z) : x(x), y(y), z(z) {}
    __host__ __device__ inline Vector3(T x, T y) : x(x), y(y), z(1) {}
    __host__ __device__ inline Vector3(T val) : x(val), y(val), z(val) {}

    __host__ __device__ inline bool operator==(const Vector3<T> &other) const
    {
        return x == other.x && y == other.y && z == other.z;
    }

    __host__ __device__ inline Vector3<T> operator+=(const Vector3<T> &other)
    {
        x += other.x;
        y += other.y;
        z += other.z;
        return *this;
    }

    __host__ __device__ inline Vector3<T> operator+(const Vector3<T> &other) const
    {
        return Vector3<T>(x + other.x, y + other.y, z + other.z);
    }

    __host__ __device__ inline Vector3<T> operator-(const Vector3<T> &other) const
    {
        return Vector3<T>(x - other.x, y - other.y, z - other.z);
    }

    __host__ __device__ inline T operator*(const Vector3<T> &other) const
    {
        return x * other.x + y * other.y + z * other.z;
    }

    __host__ __device__ inline Vector3<T> operator/(const T val) const
    {
        return Vector3<T>(x / val, y / val, z / val);
    }

    __host__ __device__ inline Vector3<T> operator%(const T val) const
    {
        return Vector3<T>(x % val, y % val, z % val);
    }

    __host__ __device__ inline float norm() const
    {
        return sqrt((float)(x * x + y * y + z * z));
    }

    __host__ __device__ inline T dot(const Vector3<T> &other) const
    {
        return x * other.x + y * other.y + z * other.z;
    }

    __host__ __device__ inline Vector3<T> cross(const Vector3<T> &other) const
    {
        return Vector3<T>(y * other.z - z * other.y, z * other.x - x * other.z, x * other.y - y * other.x);
    }

    T x, y, z;
};

//! 4-dimensional vector type
template <class T>
struct Vector4
{
    __host__ __device__ inline Vector4() : x(0), y(0), z(0), w(0) {}
    __host__ __device__ inline Vector4(const Vector3<T> &other) : x(other.x), y(other.y), z(other.z), w(other.w) {}
    __host__ __device__ inline Vector4(T x, T y, T z, T w) : x(x), y(y), z(z), w(w) {}
    __host__ __device__ inline Vector4(T val) : x(val), y(val), z(val), w(val) {}
    __host__ __device__ inline Vector4(const Vector3<T> &other, const T &val) : x(other.x), y(other.y), z(other.z), w(val) {}

    __host__ __device__ inline Vector4<T> operator+(const Vector4<T> &other) const
    {
        return Vector4<T>(x + other.x, y + other.y, z + other.z, w + other.w);
    }

    __host__ __device__ inline Vector4<T> operator-(const Vector4<T> &other) const
    {
        return Vector4<T>(x - other.x, y - other.y, z - other.z, w - other.w);
    }

    __host__ __device__ inline T operator*(const Vector4<T> &other) const
    {
        return x * other.x + y * other.y + z * other.z + w * other.w;
    }

    __host__ __device__ inline Vector4<T> operator/(const T val) const
    {
        return Vector4<T>(x / val, y / val, z / val, w / val);
    }

    T x, y, z, w;
};

//! Vector aliases
using Vector2i = Vector2<int>;
using Vector2s = Vector2<short>;
using Vector2f = Vector2<float>;
using Vector2c = Vector2<uchar>;
using Vector2d = Vector2<double>;

using Vector3i = Vector3<int>;
using Vector3f = Vector3<float>;
using Vector3c = Vector3<uchar>;
using Vector3d = Vector3<double>;

using Vector4i = Vector4<int>;
using Vector4f = Vector4<float>;
using Vector4c = Vector4<uchar>;
using Vector4d = Vector4<double>;

//! Vector Maths
//! Defined basic mathematic operations
template <class T, class U>
__host__ __device__ inline Vector3<T> operator*(T S, Vector3<U> V)
{
    return Vector3<T>(V.x * S, V.y * S, V.z * S);
}

template <class T>
__host__ __device__ inline Vector3<T> ToVector3(const Vector4<T> &V)
{
    return Vector3<T>(V.x, V.y, V.z);
}

template <class T>
__host__ __device__ inline Vector3f operator*(const Vector3<T> &V, float S)
{
    return Vector3f(V.x * S, V.y * S, V.z * S);
}

template <class T>
__host__ __device__ inline Vector3f ToVector3f(const Vector3<T> &V)
{
    return Vector3f((float)V.x, (float)V.y, (float)V.z);
}

template <class T>
__host__ __device__ inline Vector3<T> floor(const Vector3<T> &V)
{
    return Vector3<T>(std::floor(V.x), std::floor(V.y), std::floor(V.z));
}

__host__ __device__ inline Vector3f normalised(const Vector3f &V)
{
    return V / V.norm();
}

__host__ __device__ inline Vector3i operator*(const Vector3i V, int S)
{
    return Vector3i(V.x * S, V.y * S, V.z * S);
}

__host__ __device__ inline Vector3f operator+(Vector3i V1, Vector3f V2)
{
    return Vector3f(V1.x + V2.x, V1.y + V2.y, V1.z + V2.z);
}

__host__ __device__ inline Vector3c ToVector3c(const Vector3f &V)
{
    return Vector3c((int)V.x, (int)V.y, (int)V.z);
}

__host__ __device__ inline Vector3i ToVector3i(const Vector3f &V)
{
    Vector3i b((int)V.x, (int)V.y, (int)V.z);
    b.x = b.x > V.x ? b.x - 1 : b.x;
    b.y = b.y > V.y ? b.y - 1 : b.y;
    b.z = b.z > V.z ? b.z - 1 : b.z;
    return b;
}

#endif
