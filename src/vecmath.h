#pragma once

#include <helper_math.h>

#ifdef M_PI
#undef M_PI
#endif

constexpr float M_PI          = 3.1415926535897932384626422832795028841971f;
constexpr float TWO_PI        = M_PI * 2.0f;
constexpr float M_PI_2        = M_PI / 2.0f;
constexpr float M_1_PI        = 1.0f / M_PI;
constexpr float M_1_TWOPI     = 1.0f / TWO_PI;
constexpr float M_4PI         = 4.0 * M_PI;
constexpr float M_1_4PI       = 1.0 / (4.0 * M_PI);
constexpr float M_1_TWO_PI_PI = 1.0f / M_PI / TWO_PI;

class vec2 : public float2
{
public:
    __host__ __device__ inline vec2()
    {
        x = 0;
        y = 0;
    }
    __host__ __device__ inline explicit vec2(float a) { x = y = a; }
    __host__ __device__ inline vec2(float _x, float _y)
    {
        x = _x;
        y = _y;
    }
    __host__ __device__ inline vec2(const float2& v) : float2(v) {}
    __host__ __device__ inline vec2& operator=(const float2& v)
    {
        x = v.x;
        y = v.y;
        return *this;
    }
    __host__ __device__ inline float& operator[](int n) { return (&x)[n]; }
    __host__ __device__ inline const float& operator[](int n) const { return (&x)[n]; }
};

class vec3 : public float3
{
public:
    __host__ __device__ inline vec3()
    {
        x = 0;
        y = 0;
        z = 0;
    }
    __host__ __device__ inline explicit vec3(float a) { x = y = z = a; }
    __host__ __device__ inline vec3(const vec2& v, float z_)
    {
        x = v.x;
        y = v.y;
        z = z_;
    }
    __host__ __device__ inline vec3(float _x, float _y, float _z)
    {
        x = _x;
        y = _y;
        z = _z;
    }
    __host__ __device__ inline vec3(const float3& v) : float3(v) {}
    __host__ __device__ inline vec3(const float4& v)
    {
        x = v.x;
        y = v.y;
        z = v.z;
    }
    __host__ __device__ inline vec3& operator=(const float3& v)
    {
        x = v.x;
        y = v.y;
        z = v.z;
        return *this;
    }
    __host__ __device__ inline float& operator[](int n) { return (&x)[n]; }
    __host__ __device__ inline const float& operator[](int n) const { return (&x)[n]; }
};

class vec4 : public float4
{
public:
    __host__ __device__ inline vec4()
    {
        x = 0;
        y = 0;
        z = 0;
        w = 0;
    }
    __host__ __device__ inline explicit vec4(float a) { x = y = z = w = a; }
    __host__ __device__ inline vec4(float _x, float _y, float _z, float _w)
    {
        x = _x;
        y = _y;
        z = _z;
        w = _w;
    }
    __host__ __device__ inline vec4(const float4& v) : float4(v) {}
    __host__ __device__ inline vec4(const float3& v, float f)
    {
        x = v.x;
        y = v.y;
        z = v.z;
        w = f;
    }
    __host__ __device__ inline vec4& operator=(const float4& v)
    {
        x = v.x;
        y = v.y;
        z = v.z;
        w = v.w;
        return *this;
    }
    __host__ __device__ inline float& operator[](int n) { return (&x)[n]; }
    __host__ __device__ inline const float& operator[](int n) const { return (&x)[n]; }

    __host__ __device__ inline operator vec3() const { return vec3(x, y, z); }
};

template <typename T>
struct stl_vector
{
    T*     _data = nullptr;
    size_t _size = 0;

    stl_vector(int size) : _size(size) { _data = new T[size]; }
    ~stl_vector() { delete[] _data; }
    T*       data() { return _data; }
    const T* data() const { return _data; }
    T&       operator[](size_t i) { return _data[i]; }
    const T& operator[](size_t i) const { return _data[i]; }
    T*       begin() { return _data; }
    const T* cbegin() const { return _data; }
    T*       end() { return _data + _size; }
    const T* cend() const { return _data + _size; }
};
