/*
 * Copyright 1993-2015 NVIDIA Corporation.  All rights reserved.
 *
 * Please refer to the NVIDIA end user license agreement (EULA) associated
 * with this source code for terms and conditions that govern your use of
 * this software. Any use, reproduction, disclosure, or distribution of
 * this software and related documentation outside the terms of the EULA
 * is strictly prohibited.
 *
 */

// Simple 3D volume renderer

#include <iostream>
#include <cuda_runtime.h>
#include <curand_kernel.h>
#include <helper_cuda.h>
#include <helper_math.h>
#include "param.h"

uchar2* compute_volume_value_bound(const unsigned char* volume,
                                   const cudaExtent&    extent,
                                   float                search_radius);

using std::cout;
using std::endl;
#define M_PI 3.14159265358979323846

typedef unsigned int  uint;
typedef unsigned char uchar;
//typedef unsigned short VolumeType;
typedef unsigned char VolumeType;

#define GetChannel(v, i) (&(v).x)[i]

__device__ float max_of(float a, float b, float c)
{
    return fmaxf(fmaxf(a, b), c);
}

__device__ float max_of(const float3& v)
{
    return fmaxf(fmaxf(v.x, v.y), v.z);
}

__device__ float min_of(float a, float b, float c)
{
    return fminf(fminf(a, b), c);
}

__device__ float min_of(const float3& v)
{
    return fminf(fminf(v.x, v.y), v.z);
}

__device__ float avg_of(float a, float b, float c)
{
    return (a + b + c) * 0.33333333333333333333333f;
}

__device__ float avg_of(const float3& v)
{
    return (v.x + v.y + v.z) * 0.33333333333333333333333f;
}

__device__ float sum_of(float a, float b, float c)
{
    return (a + b + c);
}

__device__ float sum_of(const float3& v)
{
    return (v.x + v.y + v.z);
}

class CudaRng
{
    curandStateXORWOW_t state;

public:
    __device__
    void init(unsigned int seed)
    {
        curand_init(seed, 0, 0, &state);
    }

    __device__
    float next()
    {
        return curand_uniform(&state);
    }
};

class FractalJuliaSet
{
    float radius;
    float4 cc;
    int maxIter;

    __device__
    float4 quatSq(float4 q)
    {
        float3 q_yzw = make_float3(q.y, q.z, q.w);

        float r0 = q.x * q.x - dot(q_yzw, q_yzw);
        float3 r_yzw = q_yzw * (q.x * 2);

        return make_float4(
            r0,
            r_yzw.x,
            r_yzw.y,
            r_yzw.z);
    }

    __device__
    float eval_fractal(const float3& pos, float radius, const float4& c, int maxIter){

        float4 q = make_float4(pos.x * radius,
                               pos.y * radius,
                               pos.z * radius, 0);

        int iter = 0;
        do
        {
            q = quatSq(q);
            q += c;
        } while (dot(q, q) < 10.0f && iter++ < maxIter);

        //     return iter * (iter>5);
        //     return iter / float(maxIter);
        //     return log((float)iter+1) / log((float)maxIter);
        return (iter > maxIter * 0.9);
    }

public:
    __device__
    float density(const float3& pos)
    {
        return eval_fractal(pos, radius, cc, maxIter);
    }

    __device__
    FractalJuliaSet()
    {
        radius = 1.4f;//  3.0f;
        //     setFloat4(cc, -1, 0.2, 0, 0);
        //     setFloat4(cc, -0.291,-0.399,0.339,0.437);
        //     setFloat4(cc, -0.2,0.4,-0.4,-0.4);
        //     setFloat4(cc, -0.213,-0.0410,-0.563,-0.560);
        //     setFloat4(cc, -0.2,0.6,0.2,0.2);
        //     setFloat4(cc, -0.162,0.163,0.560,-0.599);
        cc = make_float4(-0.2f, 0.8f, 0.0f, 0.0f);
        //     setFloat4(cc, -0.445,0.339,-0.0889,-0.562);
        //     setFloat4(cc, 0.185,0.478,0.125,-0.392);
        //     setFloat4(cc, -0.450,-0.447,0.181,0.306);
        //     setFloat4(cc, -0.218,-0.113,-0.181,-0.496);
        //     setFloat4(cc, -0.137,-0.630,-0.475,-0.046);
        //     setFloat4(cc, -0.125,-0.256,0.847,0.0895);

        //     maxIter = 20;
        maxIter = 30;
    }
};

namespace TextureVolume
{
__constant__ float3 c_box_min;
__constant__ float3 c_box_max;
__constant__ float3 c_world_to_normalized;
constexpr float search_radius = 0.05f;  // tweak for performance

cudaArray*                                          d_volumeArray = 0;
texture<VolumeType, 3, cudaReadModeNormalizedFloat> density_tex;  // 3D texture

cudaArray*                                      d_volume_bound_array = 0;
texture<uchar2, 3, cudaReadModeNormalizedFloat> density_bound_tex;

// find volume density bound (min, max) around each voxel
static void compute_volume_bound(const unsigned char* volume,
                                 const cudaExtent&    extent,
                                 const float3&        boxmin,
                                 const float3&        boxmax,
                                 const float3&        world_to_normalized)
{
    auto bound_volume = compute_volume_value_bound(volume, extent, search_radius);

    {
        // create 3D array
        cudaChannelFormatDesc channelDesc = cudaCreateChannelDesc<uchar2>();
        checkCudaErrors(cudaMalloc3DArray(&d_volume_bound_array, &channelDesc, extent));

        // copy data to 3D array
        cudaMemcpy3DParms copyParams = {0};
        copyParams.srcPtr   = make_cudaPitchedPtr(bound_volume, extent.width * sizeof(uchar2), extent.width, extent.height);
        copyParams.dstArray = d_volume_bound_array;
        copyParams.extent   = extent;
        copyParams.kind     = cudaMemcpyHostToDevice;
        checkCudaErrors(cudaMemcpy3D(&copyParams));

        // set texture parameters
        density_bound_tex.normalized = true;                      // access with normalized texture coordinates
        density_bound_tex.filterMode = cudaFilterModePoint;      // linear interpolation
        density_bound_tex.addressMode[0] = cudaAddressModeClamp;  // clamp texture coordinates
        density_bound_tex.addressMode[1] = cudaAddressModeClamp;

        // bind array to 3D texture
        checkCudaErrors(cudaBindTextureToArray(density_bound_tex, d_volume_bound_array, channelDesc));
    }

    delete[] bound_volume;
}

extern "C"
void init_cuda(void *h_volume, cudaExtent volumeSize)
{
    if (h_volume)
    {
        // create 3D array
        cudaChannelFormatDesc channelDesc = cudaCreateChannelDesc<VolumeType>();
        checkCudaErrors(cudaMalloc3DArray(&d_volumeArray, &channelDesc, volumeSize));

        // copy data to 3D array
        cudaMemcpy3DParms copyParams = {0};
        copyParams.srcPtr   = make_cudaPitchedPtr(h_volume,
            volumeSize.width * sizeof(VolumeType), volumeSize.width, volumeSize.height);
        copyParams.dstArray = d_volumeArray;
        copyParams.extent   = volumeSize;
        copyParams.kind     = cudaMemcpyHostToDevice;
        checkCudaErrors(cudaMemcpy3D(&copyParams));

        // set texture parameters
        density_tex.normalized = true;                      // access with normalized texture coordinates
        density_tex.filterMode = cudaFilterModeLinear;      // linear interpolation
        density_tex.addressMode[0] = cudaAddressModeClamp;  // clamp texture coordinates
        density_tex.addressMode[1] = cudaAddressModeClamp;

        // bind array to 3D texture
        checkCudaErrors(cudaBindTextureToArray(density_tex, d_volumeArray, channelDesc));
    }

    //
    float3 box_min =
        make_float3(-1.0f,
                    -(float)volumeSize.height / (float)volumeSize.width,
                    -(float)volumeSize.depth / (float)volumeSize.width);
    float3 box_max =
        make_float3(1.0f,
                    (float)volumeSize.height / (float)volumeSize.width,
                    (float)volumeSize.depth / (float)volumeSize.width);
    float3 world_to_normalized =
        make_float3(1.0f,
                    (float)volumeSize.width / (float)volumeSize.height,
                    (float)volumeSize.width / (float)volumeSize.depth);

    checkCudaErrors(cudaMemcpyToSymbol(c_box_min, &box_min, sizeof(float3)));
    checkCudaErrors(cudaMemcpyToSymbol(c_box_max, &box_max, sizeof(float3)));
    checkCudaErrors(cudaMemcpyToSymbol(c_world_to_normalized, &world_to_normalized, sizeof(float3)));

    compute_volume_bound(reinterpret_cast<unsigned char*>(h_volume), volumeSize, box_min, box_max, world_to_normalized);
}

extern "C"
void set_texture_filter_mode(bool bLinearFilter)
{
    density_tex.filterMode = bLinearFilter ? cudaFilterModeLinear : cudaFilterModePoint;
}

extern "C"
void free_cuda_buffers()
{
    checkCudaErrors(cudaFreeArray(d_volumeArray));
}

}

CudaRng *cuda_rng = nullptr;

class Frame
{
    float3 n, t, b; // normal, tangent, bitangent

public:
    __device__
    Frame(const float3& normal)
    {
        n = (normal);
        float3 a = fabs(n.x) > 0.1 ? make_float3(0, 1, 0) : make_float3(1, 0, 0);
        t = normalize(cross(a, n));
        b = cross(n, t);
    }
    __device__
    float3 toWorld(const float3& c) const
    {
        return t * c.x + b * c.y + n * c.z;
    }
    __device__
    const float3& normal() const
    {
        return n;
    }
    __device__
    const float3& tangent() const
    {
        return t;
    }
    __device__
    const float3& bitangent() const
    {
        return b;
    }
};

class HGPhaseFunction
{
    float g;

    // perfect inversion, pdf matches evaluation exactly
    __device__
    float3 sample(float rnd0, float rnd1) const
    {
        float cos_theta;
        if (fabs(g) > 1e-6f)
        {
            float s = 2.0f * rnd0 - 1.0f;
            float f = (1.0f - g * g) / (1.0f + g * s);
            cos_theta = (0.5f / g) * (1.0f + g * g - f * f);
            cos_theta = fmaxf(0.0f, fminf(1.0f, cos_theta));
        }
        else
        {
            cos_theta = 2.0f * rnd0 - 1.0f;
        }
        float sin_theta = sqrt(1.0f - cos_theta * cos_theta);
        float phi = 2.0f * M_PI * rnd1;
        float3 ret = make_float3(cos(phi) * sin_theta, sin(phi) * sin_theta, cos_theta);
        return ret;
    }

    __device__
    float evaluate(float cos_theta) const
    {
        return (1.0f - g * g) / (4.0f * M_PI * pow(1.0f + g * g - 2 * g * cos_theta, 1.5f));
    }

public:
    __device__
    HGPhaseFunction(float g)
        : g(g)
    {

    }

    __device__
    float3 sample(const Frame& frame, float rnd0, float rnd1) const
    {
        float3 s = sample(rnd0, rnd1);
        return frame.toWorld(s);
    }

    __device__
    float evaluate(const Frame& frame, const float3& dir) const
    {
        float cos_theta = dot(frame.normal(), dir);
        return evaluate(cos_theta);
    }
}; 

typedef struct
{
    float4 m[3];
} float3x4;

__constant__ float3x4 c_invViewMatrix;  // inverse view matrix

struct Ray
{
    float3 o;   // origin
    float3 d;   // direction
};

// intersect ray with a box
// http://www.siggraph.org/education/materials/HyperGraph/raytrace/rtinter3.htm

__device__
int intersectBox(Ray r, const float3& boxmin, const float3& boxmax, float *tnear, float *tfar)
{
    // compute intersection of ray with all six bbox planes
    float3 invR = make_float3(1.0f) / r.d;
    float3 tbot = invR * (boxmin - r.o);
    float3 ttop = invR * (boxmax - r.o);

    // re-order intersections to find smallest and largest on each axis
    float3 tmin = fminf(ttop, tbot);
    float3 tmax = fmaxf(ttop, tbot);

    // find the largest tmin and the smallest tmax
    float largest_tmin = fmaxf(fmaxf(tmin.x, tmin.y), fmaxf(tmin.x, tmin.z));
    float smallest_tmax = fminf(fminf(tmax.x, tmax.y), fminf(tmax.x, tmax.z));

    *tnear = largest_tmin;
    *tfar = smallest_tmax;

    return smallest_tmax > largest_tmin;
}

// transform vector by matrix (no translation)
__device__
float3 mul(const float3x4 &M, const float3 &v)
{
    float3 r;
    r.x = dot(v, make_float3(M.m[0]));
    r.y = dot(v, make_float3(M.m[1]));
    r.z = dot(v, make_float3(M.m[2]));
    return r;
}

// transform vector by matrix with translation
__device__
float4 mul(const float3x4 &M, const float4 &v)
{
    float4 r;
    r.x = dot(v, M.m[0]);
    r.y = dot(v, M.m[1]);
    r.z = dot(v, M.m[2]);
    r.w = 1.0f;
    return r;
}

__device__
float vol_sigma_t(const float3& pos, float density)
{
#if USE_OPENVDB
    // remap position to [0, 1] coordinates
    float3 pos_ = pos * TextureVolume::c_world_to_normalized;
    float  t    = tex3D(TextureVolume::density_tex,
                    pos_.x * 0.5f + 0.5f,
                    pos_.y * 0.5f + 0.5f,
                    pos_.z * 0.5f + 0.5f);
    //t           = clamp(t, 0.0f, 1.0f) * density;
    t           *= density;
    return t;
#elif 0
    float x  = pos.x * 0.5f + 0.5f;
    float y  = pos.y * 0.5f + 0.5f;
    float z  = pos.z * 0.5f + 0.5f;
    int   xi = (int)ceil(5.0 * x);
    int   yi = (int)ceil(5.0 * y);
    int   zi = (int)ceil(5.0 * z);
    return float((xi + yi + zi) & 0x01) * density;
#else
    FractalJuliaSet fract;
    return fract.density(pos * TextureVolume::c_world_to_normalized) * density;
#endif
}

// transmittance estimation by delta tracking
__device__
float Tr(
    const float3& boxMin,
    const float3& boxMax,
    const float3& start_point,
    const float3& end_point,
    float inv_sigma,
    float density,
    CudaRng& rng)
{
    Ray shadow_ray;
    shadow_ray.o = start_point;
    shadow_ray.d = normalize(end_point - start_point);

    float t_near, t_far;
    bool shade_vol = intersectBox(shadow_ray, boxMin, boxMax, &t_near, &t_far);
    if (!shade_vol)
    {
        return 1.0f;
    }
    if (t_near < 0.0f) t_near = 0.0f;     // clamp to near plane

    float max_t = fminf(t_far, length(start_point - end_point));

    float dist = t_near;

    for (;;)
    {
        dist += -log(rng.next()) * inv_sigma;
        if (dist >= max_t)
        {
            break;
        }
        float3 pos = shadow_ray.o + shadow_ray.d * dist;

        if (rng.next() < vol_sigma_t(pos, density) * inv_sigma)
        {
            break;
        }
    }
    return float(dist >= max_t);
}

// spectral delta tracking by sample reuse
__device__ float3 Tr_spectral(const float3& boxMin,
                              const float3& boxMax,
                              const float3& start_point,
                              const float3& end_point,
                              float         inv_sigma,
                              float         density,
                              const float3& sigma_t_spectral,
                              CudaRng&      rng)
{
    Ray shadow_ray;
    shadow_ray.o = start_point;
    shadow_ray.d = normalize(end_point - start_point);

    float t_near, t_far;
    bool  shade_vol = intersectBox(shadow_ray, boxMin, boxMax, &t_near, &t_far);
    if (!shade_vol)
    {
        return make_float3(1.0f);
    }
    if (t_near < 0.0f) t_near = 0.0f;  // clamp to near plane

    float max_t = fminf(t_far, length(start_point - end_point));

    float dist  = t_near;
    int   xterm = 0;
    int   yterm = 0;
    int   zterm = 0;

    for (;;)
    {
        dist += -log(rng.next()) * inv_sigma;
        if (dist >= max_t || (xterm && yterm && zterm))
        {
            break;
        }
        float3 pos = shadow_ray.o + shadow_ray.d * dist;

        float e   = rng.next();
        float den = vol_sigma_t(pos, density);

        if (!xterm && e < sigma_t_spectral.x * den * inv_sigma)
        {
            xterm = 1;
        }
        if (!yterm && e < sigma_t_spectral.y * den * inv_sigma)
        {
            yterm = 1;
        }
        if (!zterm && e < sigma_t_spectral.z * den * inv_sigma)
        {
            zterm = 1;
        }
    }
    return make_float3(1 - xterm, 1 - yterm, 1 - zterm);
}

// transmittance estimation by ratio tracking
__device__ float3 Trr(const float3& boxMin,
                      const float3& boxMax,
                      const float3& start_point,
                      const float3& end_point,
                      float         inv_sigma,
                      float         density,
                      const float3& sigma_t_spectral,
                      CudaRng&      rng)
{
    Ray shadow_ray;
    shadow_ray.o = start_point;
    shadow_ray.d = normalize(end_point - start_point);

    float3 w = make_float3(1.0f); // transmittance

    float t_near, t_far;
    bool  shade_vol = intersectBox(shadow_ray, boxMin, boxMax, &t_near, &t_far);
    if (!shade_vol)
    {
        return w;
    }
    if (t_near < 0.0f) t_near = 0.0f;  // clamp to near plane

    float max_t = fminf(t_far, length(start_point - end_point));

    float dist = t_near;

    for (;;)
    {
        dist += -log(rng.next()) * inv_sigma;
        if (dist >= max_t)
        {
            break;
        }
        float3 pos = shadow_ray.o + shadow_ray.d * dist;

        float den = vol_sigma_t(pos, density);
        w *= 1.0f - sigma_t_spectral * (den * inv_sigma);
    }
    return w;
}

__device__ __forceinline__ float3 background(const float3& dir)
{
    //return make_float4(0.15f, 0.20f, 0.25f) * 0.5f * (dir.y + 0.5);
    return (dir.y > -0.1f) ? make_float3(0.03, 0.07, 0.23) : make_float3(0.03, 0.03, 0.03);
}

// enable one of these for proper spectral rendering
// disable both for fast non spectral rendering (only works for achromatic media)
#define MULTI_CHANNEL 0
#define SPECTRAL_TRACKING 1

__global__ void
__d_render(float4 *d_output, CudaRng *rngs, const Param P)
{
    const float density = P.density;
    const float brightness = P.brightness;
    const float3 albedo = P.albedo;

    const float3 light_dir   = make_float3(0.5826, 0.7660, 0.2717);
    const float3 light_power = make_float3(2.6, 2.5, 2.3);

    const float3& boxMin = TextureVolume::c_box_min;
    const float3& boxMax = TextureVolume::c_box_max;

    uint x = blockIdx.x*blockDim.x + threadIdx.x;
    uint y = blockIdx.y*blockDim.y + threadIdx.y;

    if ((x >= P.width) || (y >= P.height)) return;

    CudaRng& rng = rngs[x + y * P.width];
    //float    u   = (x * 2.0f - P.width) / P.height;
    //float    v   = (y * 2.0f - P.height) / P.height;
    float    u   = (x * 2.0f - P.width) / P.width;
    float    v   = (y * 2.0f - P.height) / P.width;

    // calculate eye ray in world space
    float fovx = 54.43;

    Ray cr;
    cr.o = make_float3(mul(c_invViewMatrix, make_float4(0.0f, 0.0f, 0.0f, 1.0f)));
    cr.d = (make_float3(u, v, -1.0f / tan(fovx * 0.00872664626)));
    //cr.d = (make_float3(u, v, -2.0));
    cr.d = normalize(mul(c_invViewMatrix, cr.d));

    float3 radiance = make_float3(0.0f);
    float3 throughput = make_float3(1.0f, 1.0f, 1.0f);

#if MULTI_CHANNEL
    int   channel = fminf((1.0f - rng.next()) * 3.0f, 2.99999f);
    float sigma_t = density * GetChannel(P.sigma_t, channel);
#elif SPECTRAL_TRACKING
    float3 sigma_t_spectral = P.sigma_t;
    float3 sigma_s_spectral = sigma_t_spectral * P.albedo;
    float3 sigma_a_spectral = sigma_t_spectral - sigma_s_spectral;
    float  max_sigma_t =
        max_of(sigma_t_spectral.x, sigma_t_spectral.y, sigma_t_spectral.z);
#else
    float sigma_t = density;
#endif

    int i;
    for (i = 0; i < 10000; i++)
    {
        // find intersection with box
        float t_near, t_far;
        int hit = intersectBox(cr, boxMin, boxMax, &t_near, &t_far);

        if (!hit)
        {
            radiance += background(cr.d) * throughput;
            break;
        }

        if (t_near < 0.0f)
        {
            t_near = 0.0f;     // clamp to near plane
        }

        /// woodcock tracking / delta tracking
        float3 pos = cr.o + cr.d * t_near; // current position
        float dist = t_near;

        // hyperion trick
        float s = fmaxf(0.0f, fminf(1.0f, (i - 5) * 0.066666666666666666667f));
        float g = (1 - s) * P.g;
#if SPECTRAL_TRACKING
        float density_prime = (1 - s) * density + s * density * (1 - P.g);
        float sigma_t_prime = max_sigma_t * density_prime;  // max volume density is 1
        float3 inv_sigma_spectral = make_float3(1.0f) / (sigma_t_spectral * density_prime);
#else
        float sigma_t_prime = (1 - s) * sigma_t + s * sigma_t * (1 - P.g);
#endif

        HGPhaseFunction phase(g);

        float inv_sigma = 1.0f / sigma_t_prime;  // max volume density is 1

        bool through = false;
        // delta tracking scattering event sampling
        for (;;)
        {
            dist += -log(rng.next()) * inv_sigma;
            pos = cr.o + cr.d * dist;
            if (dist >= t_far)
            {
                through = true; // transmitted through the volume, probability is 1-exp(-optical_thickness)
                break;
            }

#if SPECTRAL_TRACKING
            float  den            = vol_sigma_t(pos, density_prime);
            float3 sigma_t_den    = sigma_t_spectral * den;
            float3 sigma_s_den    = sigma_s_spectral * den;
            float3 sigma_null_den = make_float3(sigma_t_prime) - sigma_t_den;

            // history aware avg
            float Ps = sum_of(fabsf(sigma_t_den.x * throughput.x),
                              fabsf(sigma_t_den.y * throughput.y),
                              fabsf(sigma_t_den.z * throughput.z));
            float Pn = sum_of(fabsf(sigma_null_den.x * throughput.x),
                              fabsf(sigma_null_den.y * throughput.y),
                              fabsf(sigma_null_den.z * throughput.z));

            // probability normalizer
            float c = Ps + Pn;

            // using reduced termination rates
            float e = rng.next() * c;
            if (e < Ps)
            {
                throughput *= sigma_s_den * (inv_sigma * c / Ps);
                break;
            }
            else
            {
                throughput *= sigma_null_den * (inv_sigma * c / Pn);
            }
#else
            if (rng.next() < vol_sigma_t(pos, sigma_t_prime) * inv_sigma)
            {
                break;
            }
#endif
        }

        // probability is exp(-optical_thickness)
        if (through)
        {
            radiance += background(cr.d) * throughput;
            break;
        }

#if !(SPECTRAL_TRACKING)
        throughput *= albedo;
#endif

        Frame frame(cr.d);

        // direct lighting
#if SPECTRAL_TRACKING
        // reuse path and estimate for all channels
        float3 a = Tr_spectral(boxMin, boxMax, pos, light_dir * 1e10f, inv_sigma, density_prime, sigma_t_spectral, rng);
        radiance += light_power * (throughput * phase.evaluate(frame, light_dir) * a);

        // ratio tracking
        //float3 a = Trr(boxMin, boxMax, pos, light_dir * 1e10f, inv_sigma, density_prime, sigma_t_spectral, rng);
        //radiance += light_power * (throughput * phase.evaluate(frame, light_dir) * a);
#else
        float a = Tr(boxMin, boxMax, pos, light_dir * 1e10f, inv_sigma, sigma_t_prime, rng);
        radiance += light_power * (throughput * phase.evaluate(frame, light_dir) * a);
#endif

        // scattered direction
        float3 new_dir = normalize(phase.sample(frame, rng.next(), rng.next()));
        cr.o = pos;
        cr.d = new_dir;
    }

    radiance *= brightness;

    // write output color
    float heat = i * 0.001;
#if MULTI_CHANNEL
    float4 color = make_float4(0.0f, 0.0f, 0.0f, heat);
    GetChannel(color, channel) = fmaxf(GetChannel(radiance, channel), 0.0f) * 3.0f;
    d_output[x + y * P.width] += color;
#else
    d_output[x + y * P.width] += make_float4(fmaxf(radiance.x, 0.0f),
                                             fmaxf(radiance.y, 0.0f),
                                             fmaxf(radiance.z, 0.0f),
                                             heat);
#endif
}

__device__ float2 vol_bound_minmax(const float3& pos)
{
#if USE_OPENVDB
    // remap position to [0, 1] coordinates
    float3 pos_ = pos * TextureVolume::c_world_to_normalized;
    return tex3D(TextureVolume::density_bound_tex,
                 pos_.x * 0.5f + 0.5f,
                 pos_.y * 0.5f + 0.5f,
                 pos_.z * 0.5f + 0.5f);
#else
    return make_float2(0.0f, 1.0f);
#endif
}

__device__ int intersectSuperVolume(Ray           r,
                                    const float3& boxmin,
                                    const float3& boxmax,
                                    float*        tnear,
                                    float*        tfar,
                                    float*        dmin,
                                    float*        dmax)
{
    // compute intersection of ray with all six bbox planes
    float3 invR = make_float3(1.0f) / r.d;
    float3 tbot = invR * (boxmin - r.o);
    float3 ttop = invR * (boxmax - r.o);

    // re-order intersections to find smallest and largest on each axis
    float3 tmin = fminf(ttop, tbot);
    float3 tmax = fmaxf(ttop, tbot);

    // find the largest tmin and the smallest tmax
    float largest_tmin  = max_of(tmin);
    float smallest_tmax = min_of(tmax);

    *tnear = fmaxf(largest_tmin, 0.0f);  // clamp to near plane
    *tfar  = fminf(smallest_tmax, TextureVolume::search_radius);

    float2 bound = vol_bound_minmax(r.o + r.d * (*tnear));
    *dmin        = bound.y;
    *dmax        = fmaxf(0.0001f, bound.x);

    return smallest_tmax > largest_tmin && smallest_tmax >= 1e-3f;
}

// using tracking restart (similar to regular tracking)
// if each TextureVolume::search_radius track length is exceeded
// the tracker is restarted with modified ray origin only
// and for each track a local max density is used to boost the mean free path
__global__ void __d_render_bounded_decomp(float4*     d_output,
                                          CudaRng*    rngs,
                                          const Param P)
{
    const float  density    = P.density;
    const float  brightness = P.brightness;
    const float3 albedo     = P.albedo;

    const float3 light_dir   = make_float3(0.5826, 0.7660, 0.2717);
    const float3 light_power = make_float3(2.6, 2.5, 2.3);

    const float3& boxMin = TextureVolume::c_box_min;
    const float3& boxMax = TextureVolume::c_box_max;

    uint x = blockIdx.x * blockDim.x + threadIdx.x;
    uint y = blockIdx.y * blockDim.y + threadIdx.y;

    if ((x >= P.width) || (y >= P.height)) return;

    CudaRng& rng = rngs[x + y * P.width];
    // float    u   = (x * 2.0f - P.width) / P.height;
    // float    v   = (y * 2.0f - P.height) / P.height;
    float u = (x * 2.0f - P.width) / P.width;
    float v = (y * 2.0f - P.height) / P.width;

    // calculate eye ray in world space
    float fovx = 54.43;

    Ray cr;
    cr.o =
        make_float3(mul(c_invViewMatrix, make_float4(0.0f, 0.0f, 0.0f, 1.0f)));
    cr.d = (make_float3(u, v, -1.0f / tan(fovx * 0.00872664626)));
    // cr.d = (make_float3(u, v, -2.0));
    cr.d = normalize(mul(c_invViewMatrix, cr.d));

    float3 radiance   = make_float3(0.0f);
    float3 throughput = make_float3(1.0f, 1.0f, 1.0f);

#if MULTI_CHANNEL
    int   channel = fminf((1.0f - rng.next()) * 3.0f, 2.9999998f);
    float sigma_t = density * GetChannel(P.sigma_t, channel);
#elif SPECTRAL_TRACKING
    float3 sigma_t_spectral = P.sigma_t;
    float3 sigma_s_spectral = sigma_t_spectral * P.albedo;
    float3 sigma_a_spectral = sigma_t_spectral - sigma_s_spectral;
    float  max_sigma_t      = max_of(sigma_t_spectral);
    float  min_sigma_t      = min_of(sigma_t_spectral);

    // decomp
    float  sigma_c_prime;
    float  distc;
    float  sigma_r_prime;
    float3 sigma_c_spectral;
    float  inv_sigma;
    float  inv_sigma_t;
#else
    float sigma_t = density;
#endif

    int num_scatters = 0;

    while (num_scatters < 10000)
    {
        // find intersection with box
        float t_near, t_far;
        float d_min, d_max;
        int   hit = intersectSuperVolume(
            cr, boxMin, boxMax, &t_near, &t_far, &d_min, &d_max);
        bool use_decomposition =
            d_min > 0.0f;  // faster then always using decomp

        if (!hit)
        {
            radiance += background(cr.d) * throughput;
            break;
        }

        /// woodcock tracking / delta tracking
        float3 pos  = cr.o + cr.d * t_near;  // current position
        float  dist = t_near;

        // hyperion trick
        float s = fmaxf(
            0.0f, fminf(1.0f, (num_scatters - 5) * 0.066666666666666666667f));
        float g                = (1 - s) * P.g;
        float reduction_factor = (1 - s) + s * (1 - P.g);
#if SPECTRAL_TRACKING
        float density_prime = reduction_factor * density;
        float sigma_t_prime =
            max_sigma_t * density_prime * d_max;  // max volume density is d_max
        float3 inv_sigma_spectral =
            make_float3(1.0f) / (sigma_t_spectral * density_prime);

        // analog decomposition tracking
        if (use_decomposition)
        {
            sigma_c_prime = min_sigma_t * density_prime * d_min;
            distc = dist - log(rng.next()) / fmaxf(sigma_c_prime, 1e-20f);
            sigma_r_prime    = fmaxf(sigma_t_prime - sigma_c_prime, 1e-20f);
            sigma_c_spectral = make_float3(sigma_c_prime);
        }
        else
        {
            distc            = 1e20f;
            sigma_c_spectral = make_float3(0);
        }
#else
        float sigma_t_prime = reduction_factor * sigma_t;
#endif

        HGPhaseFunction phase(g);

#if SPECTRAL_TRACKING
        inv_sigma_t = 1.0f / sigma_t_prime;  // max volume density is d_max
        if (use_decomposition)
        {
            inv_sigma = 1.0f / sigma_r_prime;  // for residual tracking
        }
        else
        {
            inv_sigma = inv_sigma_t;
        }
#else
        float inv_sigma = 1.0f / sigma_t_prime;  // max volume density is d_max
#endif

        bool through = false;
        // delta tracking scattering event sampling
        for (;;)
        {
#if SPECTRAL_TRACKING
            dist += -log(rng.next()) * inv_sigma;
            if (dist >= distc || dist >= t_far)
            {
                pos = cr.o + cr.d * distc;
                break;
            }
            else
            {
                pos = cr.o + cr.d * dist;
            }
#else
            dist += -log(rng.next()) * inv_sigma;
            if (dist >= t_far)
            {
                through = true;  // transmitted through the volume, probability
                                 // is 1-exp(-optical_thickness)
                break;
            }
            pos = cr.o + cr.d * dist;
#endif

#if SPECTRAL_TRACKING
            float  den            = vol_sigma_t(pos, density_prime);
            float3 sigma_t_den    = sigma_t_spectral * den - sigma_c_spectral;
            float3 sigma_s_den    = sigma_s_spectral * den - sigma_c_spectral;
            float3 sigma_null_den = make_float3(sigma_t_prime) - sigma_t_den;

            // history aware avg
            float Ps = sum_of(fabsf(sigma_t_den.x * throughput.x),
                              fabsf(sigma_t_den.y * throughput.y),
                              fabsf(sigma_t_den.z * throughput.z));
            float Pn = sum_of(fabsf(sigma_null_den.x * throughput.x),
                              fabsf(sigma_null_den.y * throughput.y),
                              fabsf(sigma_null_den.z * throughput.z));

            // probability normalizer
            float c = (Ps + Pn);

            // using reduced termination rates
            float e = rng.next() * c;
            if (e < Ps)
            {
                throughput *= sigma_s_den * (inv_sigma_t * c / (Ps));
                //++num_scatters;
                break;
            }
            else
            {
                throughput *= sigma_null_den * (inv_sigma_t * c / Pn);
            }
#else
            if (rng.next() < vol_sigma_t(pos, sigma_t_prime) * inv_sigma)
            {
                ++num_scatters;
                break;
            }
#endif
        }

#if SPECTRAL_TRACKING
        through = fminf(distc, dist) >= t_far;
        num_scatters += (!through);
#endif

        // tracking restart
        // probability is exp(-optical_thickness)
        if (through)
        {
            cr.o = cr.o + cr.d * t_far;
            continue;
        }

#if !(SPECTRAL_TRACKING)
        throughput *= albedo;
#endif

        Frame frame(cr.d);

        // direct lighting
#if SPECTRAL_TRACKING
        // reuse path and estimate for all channels
        float3 a = Tr_spectral(boxMin,
                               boxMax,
                               pos,
                               light_dir * 1e10f,
                               inv_sigma_t,
                               density_prime,
                               sigma_t_spectral,
                               rng);
        radiance +=
            light_power * (throughput * phase.evaluate(frame, light_dir) * a);
#else
        float a = Tr(boxMin,
                     boxMax,
                     pos,
                     light_dir * 1e10f,
                     inv_sigma,
                     sigma_t_prime,
                     rng);
        radiance +=
            light_power * (throughput * phase.evaluate(frame, light_dir) * a);
#endif

        // scattered direction
        float3 new_dir = normalize(phase.sample(frame, rng.next(), rng.next()));
        cr.o           = pos;
        cr.d           = new_dir;
    }

    radiance *= brightness;

    // write output color
    float heat = num_scatters * 0.001;
#if MULTI_CHANNEL
    float4 color = make_float4(0.0f, 0.0f, 0.0f, heat);
    GetChannel(color, channel) =
        fmaxf(GetChannel(radiance, channel), 0.0f) * 3.0f;
    d_output[x + y * P.width] += color;
#else
    d_output[x + y * P.width] += make_float4(fmaxf(radiance.x, 0.0f),
                                             fmaxf(radiance.y, 0.0f),
                                             fmaxf(radiance.z, 0.0f),
                                             heat);
#endif
}

extern "C" void copy_inv_view_matrix(float* invViewMatrix, size_t sizeofMatrix)
{
    checkCudaErrors(cudaMemcpyToSymbol(c_invViewMatrix, invViewMatrix, sizeofMatrix));
}




namespace XORShift
{ 
    // XOR shift PRNG
    unsigned int x = 123456789;
    unsigned int y = 362436069;
    unsigned int z = 521288629;
    unsigned int w = 88675123;
    inline unsigned int frand()
    {
        unsigned int t;
        t = x ^ (x << 11);
        x = y; y = z; z = w;
        return (w = (w ^ (w >> 19)) ^ (t ^ (t >> 8)));
    }
}

__global__
void __init_rng(CudaRng *rng, int width, int height, unsigned int *seeds)
{
    uint x = blockIdx.x * blockDim.x + threadIdx.x;
    uint y = blockIdx.y * blockDim.y + threadIdx.y;

    if ((x >= width) || (y >= height))
    {
        return;
    }

    int idx = x + y * width;
    rng[idx].init(seeds[idx]);
}

extern "C"
void init_rng(dim3 gridSize, dim3 blockSize, int width, int height)
{
    cout << "init cuda rng to " << width << " x " << height << endl;
    unsigned int *seeds;
    checkCudaErrors(cudaMalloc(&seeds, sizeof(unsigned int) * width * height));

    unsigned int* h_seeds = new unsigned int[width * height];
    for (int i = 0; i < width * height; ++i)
    {
        h_seeds[i] = XORShift::frand();
    }
    checkCudaErrors(
        cudaMemcpy(seeds, h_seeds, sizeof(unsigned int) * width * height, cudaMemcpyHostToDevice));
    delete h_seeds;

    checkCudaErrors(cudaMalloc(&cuda_rng, sizeof(CudaRng) * width * height));
    __init_rng << <gridSize, blockSize >> >(cuda_rng, width, height, seeds);
    checkCudaErrors(cudaFree(seeds));
}

extern "C"
void free_rng()
{
    cout << "free cuda rng" << endl;
    checkCudaErrors(cudaFree(cuda_rng));
}

extern "C"
void render_kernel(dim3 gridSize, dim3 blockSize, float4 *d_output, const Param& p)
{
    //__d_render<<<gridSize, blockSize>>>(d_output, cuda_rng, p);
    __d_render_bounded_decomp<<<gridSize, blockSize>>>(d_output, cuda_rng, p);
}

__global__
void __scale(float4 *dst, float4 *src, int size, float scale)
{
    int idx = threadIdx.x + blockIdx.x * blockDim.x;
    if (idx >= size)
    {
        return;
    }
    dst[idx] = src[idx] * scale;
}

extern "C"
void scale(float4 *dst, float4 *src, int size, float scale)
{
    __scale << <(size + 256 - 1) / 256, 256 >> >(dst, src, size, scale);
}

__global__ void __gamma_correct(float4* dst, float4* src, int size, float scale, float gamma)
{
    int idx = threadIdx.x + blockIdx.x * blockDim.x;
    if (idx >= size)
    {
        return;
    }
    float4 d = src[idx] * scale;
    dst[idx] = make_float4(powf(d.x, gamma), powf(d.y, gamma), powf(d.z, gamma), 1.0f);
}

extern "C" void gamma_correct(float4* dst, float4* src, int size, float scale, float gamma)
{
    __gamma_correct<<<(size + 256 - 1) / 256, 256>>>(dst, src, size, scale, 1.0f / gamma);
}
