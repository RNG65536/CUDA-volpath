#include <cuda_runtime.h>
#include <curand_kernel.h>
#include <helper_cuda.h>
#include <helper_math.h>

#include <iostream>

#include "param.h"
#include "vecmath.h"

//
// Configurations
//

#define BUNDLE_TEST 0

// enable to add explicit directional sun light
#define SUN_LIGHT 1

// enable to use passive envmap sampling
#define PASSIVE_ENVMAP 1

// enable one of these for proper spectral rendering
// disable both for fast non spectral rendering (only works for achromatic
// media)
#define MULTI_CHANNEL 0
#define SPECTRAL_TRACKING (!(MULTI_CHANNEL))

// allow transforming the bounding box, but slows down the renderer
#define USE_MODEL_TRANSFORM 0

// slower, but extended with float texture support
#define DYNAMIC_TEXTURE 0

#define FAST_RNG 1

#if USE_OPENVDB
uchar2* compute_volume_value_bound(const unsigned char* volume,
                                   const cudaExtent&    extent,
                                   float                search_radius);
float2* compute_volume_value_bound(const float*      volume,
                                   const cudaExtent& extent,
                                   float             search_radius);
#endif

using std::cout;
using std::endl;

typedef unsigned int  uint;
typedef unsigned char uchar;
// typedef unsigned short VolumeType;
typedef unsigned char VolumeType;

#define GetChannel(v, i) ((&(v).x)[i])

__device__ float MIS_balance_heuristic(float a, float b) { return a / (a + b); }

__device__ float MIS_power_heuristic(float a, float b) { return (a * a) / (a * a + b * b); }

__device__ float MIS_balance_heuristic(float a, float b, float c) { return a / (a + b + c); }

__device__ float MIS_power_heuristic(float a, float b, float c)
{
    return (a * a) / (a * a + b * b + c * c);
}

__device__ float max_of(float a, float b, float c) { return fmaxf(fmaxf(a, b), c); }

__device__ float max_of(const float3& v) { return fmaxf(fmaxf(v.x, v.y), v.z); }

__device__ float min_of(float a, float b, float c) { return fminf(fminf(a, b), c); }

__device__ float min_of(const float3& v) { return fminf(fminf(v.x, v.y), v.z); }

__device__ float avg_of(float a, float b, float c)
{
    return (a + b + c) * 0.33333333333333333333333f;
}

__device__ float avg_of(const float3& v) { return (v.x + v.y + v.z) * 0.33333333333333333333333f; }

__device__ float sum_of(float a, float b, float c) { return (a + b + c); }

__device__ float sum_of(const float3& v) { return (v.x + v.y + v.z); }

#if !(FAST_RNG)
class CudaRng
{
    curandStateXORWOW_t state;

public:
    __device__ void init(unsigned int seed) { curand_init(seed, 0, 0, &state); }

    __device__ float next() { return curand_uniform(&state); }
};
#else
__device__ unsigned int Hash(unsigned int seed)
{
    seed = (seed ^ 61) ^ (seed >> 16);
    seed *= 9;
    seed = seed ^ (seed >> 4);
    seed *= 0x27d4eb2d;
    seed = seed ^ (seed >> 15);
    return seed;
}

__device__ unsigned int RngNext(unsigned int& seed_x, unsigned int& seed_y)
{
    unsigned int result = seed_x * 0x9e3779bb;

    seed_y ^= seed_x;
    seed_x = ((seed_x << 26) | (seed_x >> (32 - 26))) ^ seed_y ^ (seed_y << 9);
    seed_y = (seed_x << 13) | (seed_x >> (32 - 13));

    return result;
}

__device__ float Rand(unsigned int& seed_x, unsigned int& seed_y)
{
    unsigned int u = 0x3f800000 | (RngNext(seed_x, seed_y) >> 9);
    return __uint_as_float(u) - 1.0;
}

class CudaRng
{
    unsigned int seed_x, seed_y;

public:
    __device__ void init(unsigned int pixel_x, unsigned int pixel_y, unsigned int frame_idx)
    {
        unsigned int s0 = (pixel_x << 16) | pixel_y;
        unsigned int s1 = frame_idx;

        seed_x = Hash(s0);
        seed_y = Hash(s1);
        RngNext(seed_x, seed_y);
    }

    __device__ float next() { return Rand(seed_x, seed_y); }
};
#endif

class FractalJuliaSet
{
    float  radius;
    float4 cc;
    int    maxIter;

    __device__ float4 quatSq(float4 q)
    {
        float3 q_yzw = make_float3(q.y, q.z, q.w);

        float  r0    = q.x * q.x - dot(q_yzw, q_yzw);
        float3 r_yzw = q_yzw * (q.x * 2);

        return make_float4(r0, r_yzw.x, r_yzw.y, r_yzw.z);
    }

    __device__ float eval_fractal(const float3& pos, float radius, const float4& c, int maxIter)
    {
        float4 q = make_float4(pos.x * radius, pos.y * radius, pos.z * radius, 0);

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
    __device__ float density(const float3& pos) { return eval_fractal(pos, radius, cc, maxIter); }

    __device__ FractalJuliaSet()
    {
        radius = 1.4f;  //  3.0f;
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
// constexpr float search_radius = 0.1f;
constexpr float search_radius = 0.05f;  // tweak for performance

#if DYNAMIC_TEXTURE
struct CudaTexture
{
    float3              min;
    int                 Nx;
    float3              max;
    int                 Nxy;
    float3              l_inv;
    int                 total;
    cudaTextureObject_t texture;

    template <typename T>
    __device__ __forceinline__ T fetch_gpu(const float3& pos) const
    {
        float3 p = (pos - min) * l_inv;
        return tex3D<T>(texture, p.x, p.y, p.z);
    }
};

template <typename T>
struct CudaArray
{
    cudaArray_t array;
    size_t      width;
    size_t      height;
    size_t      depth;
};

template <typename T>
CudaArray<T> create_cuda_array(T* data, int width, int height, int depth)
{
    CudaArray<T> a;

    cudaExtent            extent  = make_cudaExtent(width, height, depth);
    cudaChannelFormatDesc channel = cudaCreateChannelDesc<T>();
    checkCudaErrors(cudaMalloc3DArray(&a.array, &channel, extent));

    cudaMemcpy3DParms cp;
    memset(&cp, 0, sizeof(cp));
    cp.srcPtr   = make_cudaPitchedPtr((void*)data, width * sizeof(T), width, height);
    cp.dstArray = a.array;
    cp.extent   = extent;
    cp.kind     = cudaMemcpyHostToDevice;
    checkCudaErrors(cudaMemcpy3D(&cp));

    a.width  = width;
    a.height = width;
    a.depth  = width;
    return a;
}

template <typename T>
void free_cuda_array(CudaArray<T>& array)
{
    checkCudaErrors(cudaFreeArray(array.array));
}

template <typename T>
struct cudaTextureDesc get_texture_desc(bool linear_interp, bool normalized_coords);

template <>
struct cudaTextureDesc get_texture_desc<float>(bool linear_interp, bool normalized_coords)
{
    struct cudaTextureDesc tex;
    memset(&tex, 0, sizeof(cudaTextureDesc));
    tex.addressMode[0]   = cudaAddressModeClamp;
    tex.addressMode[1]   = cudaAddressModeClamp;
    tex.addressMode[2]   = cudaAddressModeClamp;
    tex.filterMode       = linear_interp ? cudaFilterModeLinear : cudaFilterModePoint;
    tex.readMode         = cudaReadModeElementType;
    tex.normalizedCoords = normalized_coords;
    return tex;
}

template <>
struct cudaTextureDesc get_texture_desc<float2>(bool linear_interp, bool normalized_coords)
{
    struct cudaTextureDesc tex;
    memset(&tex, 0, sizeof(cudaTextureDesc));
    tex.addressMode[0]   = cudaAddressModeClamp;
    tex.addressMode[1]   = cudaAddressModeClamp;
    tex.addressMode[2]   = cudaAddressModeClamp;
    tex.filterMode       = linear_interp ? cudaFilterModeLinear : cudaFilterModePoint;
    tex.readMode         = cudaReadModeElementType;
    tex.normalizedCoords = normalized_coords;
    return tex;
}

template <>
struct cudaTextureDesc get_texture_desc<uchar>(bool linear_interp, bool normalized_coords)
{
    struct cudaTextureDesc tex;
    memset(&tex, 0, sizeof(cudaTextureDesc));
    tex.addressMode[0]   = cudaAddressModeClamp;
    tex.addressMode[1]   = cudaAddressModeClamp;
    tex.addressMode[2]   = cudaAddressModeClamp;
    tex.filterMode       = linear_interp ? cudaFilterModeLinear : cudaFilterModePoint;
    tex.readMode         = cudaReadModeNormalizedFloat;
    tex.normalizedCoords = normalized_coords;
    return tex;
}

template <>
struct cudaTextureDesc get_texture_desc<uchar2>(bool linear_interp, bool normalized_coords)
{
    struct cudaTextureDesc tex;
    memset(&tex, 0, sizeof(cudaTextureDesc));
    tex.addressMode[0]   = cudaAddressModeClamp;
    tex.addressMode[1]   = cudaAddressModeClamp;
    tex.addressMode[2]   = cudaAddressModeClamp;
    tex.filterMode       = linear_interp ? cudaFilterModeLinear : cudaFilterModePoint;
    tex.readMode         = cudaReadModeNormalizedFloat;
    tex.normalizedCoords = normalized_coords;
    return tex;
}

template <typename T>
CudaTexture create_cuda_texture(CudaArray<T>& array,
                                const float3* box_min           = nullptr,
                                const float3* box_max           = nullptr,
                                bool          linear_interp     = true,
                                bool          normalized_coords = true)
{
    CudaTexture t;
    t.Nx  = array.width;
    t.Nxy = array.width * array.height;
    t.min = box_min ? *box_min
                    : make_float3(-1.0f,
                                  -(float)array.height / (float)array.width,
                                  -(float)array.depth / (float)array.width);
    t.max = box_max ? *box_max
                    : make_float3(1.0f,
                                  (float)array.height / (float)array.width,
                                  (float)array.depth / (float)array.width);
    t.l_inv = 1.0f / (t.max - t.min);
    t.total = array.width * array.height * array.depth;  // note that this may overflow

    struct cudaResourceDesc res;
    memset(&res, 0, sizeof(cudaResourceDesc));
    res.resType         = cudaResourceTypeArray;
    res.res.array.array = array.array;

    struct cudaTextureDesc tex = get_texture_desc<T>(linear_interp, normalized_coords);

    checkCudaErrors(cudaCreateTextureObject(&t.texture, &res, &tex, nullptr));
    return t;
}

void free_cuda_texture(CudaTexture& texture)
{
    checkCudaErrors(cudaDestroyTextureObject(texture.texture));
}

static CudaArray<uchar>  h_volumeArray;
static CudaArray<uchar2> h_volume_bound_array;
static CudaArray<float>  h_volumeArray_f;
static CudaArray<float2> h_volume_bound_array_f;

__constant__ CudaTexture density_tex;
static CudaTexture       h_density_tex;

__constant__ CudaTexture density_bound_tex;
static CudaTexture       h_density_bound_tex;

static float3 box_min;
static float3 box_max;
bool          linear_interp = false;
bool          quantized     = true;

extern "C" void init_cuda(void*         h_volume,
                          cudaExtent    volumeSize,
                          bool          quantized_,
                          const float3* boxmin,
                          const float3* boxmax)
{
    if (!h_volume)
    {
        fprintf_s(stderr, "cannot init without host volume\n");
        exit(1);
    }

    if (boxmin && boxmax)
    {
        box_min = *boxmin;
        box_max = *boxmax;
    }
    else
    {
        box_min = make_float3(-1.0f,
                              -(float)volumeSize.height / (float)volumeSize.width,
                              -(float)volumeSize.depth / (float)volumeSize.width);
        box_max = make_float3(1.0f,
                              (float)volumeSize.height / (float)volumeSize.width,
                              (float)volumeSize.depth / (float)volumeSize.width);
    }

    quantized = quantized_;
    if (quantized)
    {
        h_volumeArray = create_cuda_array(reinterpret_cast<uchar*>(h_volume),
                                          volumeSize.width,
                                          volumeSize.height,
                                          volumeSize.depth);
        h_density_tex = create_cuda_texture(h_volumeArray, &box_min, &box_max, linear_interp, true);

        auto bound_volume = compute_volume_value_bound(
            reinterpret_cast<uchar*>(h_volume), volumeSize, search_radius);

        h_volume_bound_array =
            create_cuda_array(bound_volume, volumeSize.width, volumeSize.height, volumeSize.depth);
        h_density_bound_tex =
            create_cuda_texture(h_volume_bound_array, &box_min, &box_max, false, true);

        delete[] bound_volume;
    }
    else
    {
        h_volumeArray_f = create_cuda_array(reinterpret_cast<float*>(h_volume),
                                            volumeSize.width,
                                            volumeSize.height,
                                            volumeSize.depth);
        h_density_tex =
            create_cuda_texture(h_volumeArray_f, &box_min, &box_max, linear_interp, true);

        auto bound_volume = compute_volume_value_bound(
            reinterpret_cast<float*>(h_volume), volumeSize, search_radius);

        h_volume_bound_array_f =
            create_cuda_array(bound_volume, volumeSize.width, volumeSize.height, volumeSize.depth);
        h_density_bound_tex =
            create_cuda_texture(h_volume_bound_array_f, &box_min, &box_max, false, true);

        delete[] bound_volume;
    }

    //
    checkCudaErrors(cudaMemcpyToSymbolAsync(density_tex, &h_density_tex, sizeof(density_tex)));
    checkCudaErrors(cudaMemcpyToSymbolAsync(
        density_bound_tex, &h_density_bound_tex, sizeof(density_bound_tex)));
}

extern "C" void set_texture_filter_mode(bool bLinearFilter)
{
    if (bLinearFilter != linear_interp)
    {
        linear_interp = bLinearFilter;
        if (quantized)
        {
            h_density_tex =
                create_cuda_texture(h_volumeArray, &box_min, &box_max, bLinearFilter, true);
        }
        else
        {
            h_density_tex =
                create_cuda_texture(h_volumeArray_f, &box_min, &box_max, bLinearFilter, true);
        }
        checkCudaErrors(cudaMemcpyToSymbolAsync(density_tex, &h_density_tex, sizeof(density_tex)));
    }
}

extern "C" void free_cuda_buffers()
{
    free_cuda_array(h_volumeArray);
    free_cuda_array(h_volume_bound_array);
    free_cuda_array(h_volumeArray_f);
    free_cuda_array(h_volume_bound_array_f);
    free_cuda_texture(h_density_tex);
    free_cuda_texture(h_density_bound_tex);
}

#else

__constant__ float3 c_box_min;
__constant__ float3 c_box_max;
__constant__ float3 c_world_to_normalized;

cudaArray*                                          d_volumeArray = 0;
texture<VolumeType, 3, cudaReadModeNormalizedFloat> density_tex;  // 3D texture

#if USE_OPENVDB
cudaArray*                                          d_volume_bound_array = 0;
texture<uchar2, 3, cudaReadModeNormalizedFloat>     density_bound_tex;

// find volume density bound (min, max) around each voxel
static void compute_volume_bound(const uchar*      volume,
                                 const cudaExtent& extent,
                                 const float3&     boxmin,
                                 const float3&     boxmax,
                                 const float3&     world_to_normalized)
{
    auto bound_volume = compute_volume_value_bound(volume, extent, search_radius);

    {
        // create 3D array
        cudaChannelFormatDesc channelDesc = cudaCreateChannelDesc<uchar2>();
        checkCudaErrors(cudaMalloc3DArray(&d_volume_bound_array, &channelDesc, extent));

        // copy data to 3D array
        cudaMemcpy3DParms copyParams = {0};
        copyParams.srcPtr            = make_cudaPitchedPtr(
            bound_volume, extent.width * sizeof(uchar2), extent.width, extent.height);
        copyParams.dstArray = d_volume_bound_array;
        copyParams.extent   = extent;
        copyParams.kind     = cudaMemcpyHostToDevice;
        checkCudaErrors(cudaMemcpy3D(&copyParams));

        // set texture parameters
        density_bound_tex.normalized     = true;  // access with normalized texture coordinates
        density_bound_tex.filterMode     = cudaFilterModePoint;   // linear interpolation
        density_bound_tex.addressMode[0] = cudaAddressModeClamp;  // clamp texture coordinates
        density_bound_tex.addressMode[1] = cudaAddressModeClamp;
        density_bound_tex.addressMode[2] = cudaAddressModeClamp;

        // bind array to 3D texture
        checkCudaErrors(
            cudaBindTextureToArray(density_bound_tex, d_volume_bound_array, channelDesc));
    }

    delete[] bound_volume;
}
#endif

extern "C" void init_cuda(void*         h_volume,
                          cudaExtent    volumeSize,
                          bool          quantized,
                          const float3* boxmin,
                          const float3* boxmax)
{
    if (h_volume)
    {
        // create 3D array
        cudaChannelFormatDesc channelDesc = cudaCreateChannelDesc<VolumeType>();
        checkCudaErrors(cudaMalloc3DArray(&d_volumeArray, &channelDesc, volumeSize));

        // copy data to 3D array
        cudaMemcpy3DParms copyParams = {0};
        copyParams.srcPtr            = make_cudaPitchedPtr(
            h_volume, volumeSize.width * sizeof(VolumeType), volumeSize.width, volumeSize.height);
        copyParams.dstArray = d_volumeArray;
        copyParams.extent   = volumeSize;
        copyParams.kind     = cudaMemcpyHostToDevice;
        checkCudaErrors(cudaMemcpy3D(&copyParams));

        // set texture parameters
        density_tex.normalized     = true;  // access with normalized texture coordinates
        density_tex.filterMode     = cudaFilterModeLinear;  // linear interpolation
        density_tex.addressMode[0] = cudaAddressModeClamp;  // clamp texture coordinates
        density_tex.addressMode[1] = cudaAddressModeClamp;
        density_tex.addressMode[2] = cudaAddressModeClamp;

        // bind array to 3D texture
        checkCudaErrors(cudaBindTextureToArray(density_tex, d_volumeArray, channelDesc));
    }

    //
    float3 box_min;
    float3 box_max;
    if (boxmin && boxmax)
    {
        box_min = *boxmin;
        box_max = *boxmax;
    }
    else
    {
        box_min = make_float3(-1.0f,
                              -(float)volumeSize.height / (float)volumeSize.width,
                              -(float)volumeSize.depth / (float)volumeSize.width);
        box_max = make_float3(1.0f,
                              (float)volumeSize.height / (float)volumeSize.width,
                              (float)volumeSize.depth / (float)volumeSize.width);
    }

    float3 world_to_normalized = make_float3(1.0f,
                                             (float)volumeSize.width / (float)volumeSize.height,
                                             (float)volumeSize.width / (float)volumeSize.depth);
    checkCudaErrors(cudaMemcpyToSymbolAsync(c_box_min, &box_min, sizeof(float3)));
    checkCudaErrors(cudaMemcpyToSymbolAsync(c_box_max, &box_max, sizeof(float3)));
    checkCudaErrors(
        cudaMemcpyToSymbolAsync(c_world_to_normalized, &world_to_normalized, sizeof(float3)));

#if USE_OPENVDB
    compute_volume_bound(
        reinterpret_cast<uchar*>(h_volume), volumeSize, box_min, box_max, world_to_normalized);
#endif
}

extern "C" void set_texture_filter_mode(bool bLinearFilter)
{
    density_tex.filterMode = bLinearFilter ? cudaFilterModeLinear : cudaFilterModePoint;
}

extern "C" void free_cuda_buffers()
{
    checkCudaErrors(cudaFreeArray(d_volumeArray));
#if USE_OPENVDB
    checkCudaErrors(cudaFreeArray(d_volume_bound_array));
#endif
}
#endif

}  // namespace TextureVolume

CudaRng* cuda_rng = nullptr;

class Frame
{
    float3 n, t, b;  // normal, tangent, bitangent

public:
    __device__ Frame(const float3& normal)
    {
        n        = (normal);
        float3 a = fabs(n.x) > 0.1 ? make_float3(0, 1, 0) : make_float3(1, 0, 0);
        t        = normalize(cross(a, n));
        b        = cross(n, t);
    }
    __device__ float3 toWorld(const float3& c) const { return t * c.x + b * c.y + n * c.z; }
    __device__ const float3& normal() const { return n; }
    __device__ const float3& tangent() const { return t; }
    __device__ const float3& bitangent() const { return b; }
};

class HGPhaseFunction
{
    float g;

    // perfect inversion, pdf matches evaluation exactly
    __device__ float3 sample(float rnd0, float rnd1) const
    {
        float cos_theta;
        if (fabs(g) > 1e-6f)
        {
            float s   = 2.0f * rnd0 - 1.0f;
            float f   = (1.0f - g * g) / (1.0f + g * s);
            cos_theta = (0.5f / g) * (1.0f + g * g - f * f);
            cos_theta = fmaxf(0.0f, fminf(1.0f, cos_theta));
        }
        else
        {
            cos_theta = 2.0f * rnd0 - 1.0f;
        }
        float  sin_theta = sqrt(1.0f - cos_theta * cos_theta);
        float  phi       = 2.0f * M_PI * rnd1;
        float3 ret       = make_float3(cos(phi) * sin_theta, sin(phi) * sin_theta, cos_theta);
        return ret;
    }

    __device__ float evaluate(float cos_theta) const
    {
        return (1.0f - g * g) / (4.0f * M_PI * pow(1.0f + g * g - 2 * g * cos_theta, 1.5f));
    }

public:
    __device__ HGPhaseFunction(float g) : g(g) {}

    __device__ float3 sample(const Frame& frame, float rnd0, float rnd1) const
    {
        float3 s = sample(rnd0, rnd1);
        return frame.toWorld(s);
    }

    __device__ float evaluate(const Frame& frame, const float3& dir) const
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

__constant__ float3x4 c_invModelMatrix;  // inverse model matrix

struct Ray
{
    float3 o;  // origin
    float3 d;  // direction
};

// transform vector by matrix (no translation)
__device__ float3 mul(const float3x4& M, const float3& v)
{
    float3 r;
    r.x = dot(v, make_float3(M.m[0]));
    r.y = dot(v, make_float3(M.m[1]));
    r.z = dot(v, make_float3(M.m[2]));
    return r;
}

// transform vector by matrix with translation
__device__ float4 mul(const float3x4& M, const float4& v)
{
    float4 r;
    r.x = dot(v, M.m[0]);
    r.y = dot(v, M.m[1]);
    r.z = dot(v, M.m[2]);
    r.w = 1.0f;
    return r;
}

// intersect ray with a box
// http://www.siggraph.org/education/materials/HyperGraph/raytrace/rtinter3.htm

__device__ int intersectBox(
    Ray r, const float3& boxmin, const float3& boxmax, float* tnear, float* tfar)
{
    // compute intersection of ray with all six bbox planes
#if USE_MODEL_TRANSFORM
    float3 invR = make_float3(1.0f) / mul(c_invModelMatrix, r.d);
    float3 tbot = invR * (boxmin - make_float3(mul(c_invModelMatrix, make_float4(r.o, 1.0f))));
    float3 ttop = invR * (boxmax - make_float3(mul(c_invModelMatrix, make_float4(r.o, 1.0f))));
#else
    float3        invR = make_float3(1.0f) / r.d;
    float3        tbot = invR * (boxmin - r.o);
    float3        ttop = invR * (boxmax - r.o);
#endif

    // re-order intersections to find smallest and largest on each axis
    float3 tmin = fminf(ttop, tbot);
    float3 tmax = fmaxf(ttop, tbot);

    // find the largest tmin and the smallest tmax
    float largest_tmin  = max_of(tmin);
    float smallest_tmax = min_of(tmax);

    *tnear = largest_tmin;
    *tfar  = smallest_tmax;

    return smallest_tmax > largest_tmin && smallest_tmax >= 1e-3f;
}

__device__ float vol_sigma_t(const float3& pos_, float density)
{
#if USE_MODEL_TRANSFORM
    float3 pos = make_float3(mul(c_invModelMatrix, make_float4(pos_, 1.0f)));
#else
    const float3& pos  = pos_;
#endif

#if USE_OPENVDB
#if DYNAMIC_TEXTURE
    // use world position for access
    float t = TextureVolume::density_tex.fetch_gpu<float>(pos);
    // t           = clamp(t, 0.0f, 1.0f) * density;
    t *= density;
    return t;
#else
    // remap position to [0, 1] coordinates
    float3 posn = pos * TextureVolume::c_world_to_normalized;
    float  t    = tex3D(TextureVolume::density_tex,
                    posn.x * 0.5f + 0.5f,
                    posn.y * 0.5f + 0.5f,
                    posn.z * 0.5f + 0.5f);
    // t           = clamp(t, 0.0f, 1.0f) * density;
    t *= density;
    return t;
#endif
#elif 0
    float         x    = pos.x * 0.5f + 0.5f;
    float         y    = pos.y * 0.5f + 0.5f;
    float         z    = pos.z * 0.5f + 0.5f;
    int           xi   = (int)ceil(5.0 * x);
    int           yi   = (int)ceil(5.0 * y);
    int           zi   = (int)ceil(5.0 * z);
    return float((xi + yi + zi) & 0x01) * density;
#else
    FractalJuliaSet fract;
    return fract.density(pos * TextureVolume::c_world_to_normalized) * density;
#endif
}

// transmittance estimation by delta tracking
__device__ float Tr(const float3& boxMin,
                    const float3& boxMax,
                    const float3& start_point,
                    const float3& end_point,
                    float         inv_sigma,
                    float         density,
                    CudaRng&      rng)
{
    Ray shadow_ray;
    shadow_ray.o = start_point;
    shadow_ray.d = normalize(end_point - start_point);

    float t_near, t_far;
    bool  shade_vol = intersectBox(shadow_ray, boxMin, boxMax, &t_near, &t_far);
    if (!shade_vol)
    {
        return 1.0f;
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

    float3 w = make_float3(1.0f);  // transmittance

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

namespace Envmap
{
#define MULT_PDF 0
#define PRE_WARP 1

__constant__ int                            HDRwidth;
__constant__ int                            HDRheight;
__constant__ float                          HDRpdfnorm;
__constant__ float                          HDRpdfnormAlt;
texture<float4, 2, cudaReadModeElementType> HDRtexture;
cudaArray_t                                 HDRtexture_  = nullptr;
cudaArray_t                                 HDRtexture_t = nullptr;
texture<float, 1, cudaReadModeElementType>  EnvmapPdfY;
cudaArray_t                                 EnvmapPdfY_  = nullptr;
cudaArray_t                                 EnvmapPdfY_t = nullptr;
texture<float, 1, cudaReadModeElementType>  EnvmapCdfY;
cudaArray_t                                 EnvmapCdfY_  = nullptr;
cudaArray_t                                 EnvmapCdfY_t = nullptr;
texture<float, 2, cudaReadModeElementType>  EnvmapPdfX;
cudaArray_t                                 EnvmapPdfX_  = nullptr;
cudaArray_t                                 EnvmapPdfX_t = nullptr;
texture<float, 2, cudaReadModeElementType>  EnvmapCdfX;
cudaArray_t                                 EnvmapCdfX_  = nullptr;
cudaArray_t                                 EnvmapCdfX_t = nullptr;

int  envmap_width  = 0;
int  envmap_height = 0;
bool initialized   = 0;

__device__ inline float dir_to_theta(const vec3& dir)
{
    float theta = atanf(dir[2] / dir[0]) + M_PI_2;
    if (dir[0] < 0) theta += M_PI;
    return theta;
}

__device__ inline void dir_to_uv(const vec3& dir, float& u, float& v)
{
    float phi   = acosf(dir.y);
    float theta = dir_to_theta(dir);
    u           = theta * M_1_TWOPI;
    v           = phi * M_1_PI;
}

__device__ inline vec3 uv_to_dir(float u, float v)
{
    float theta = u * TWO_PI;
    float phi   = v * M_PI;
    return vec3(sin(phi) * sin(theta), cos(phi), sin(phi) * -cos(theta));
}

__device__ int sample_y(float r)
{
    int begin = 0;
    int end   = HDRheight - 1;
    while (end > begin)
    {
        int   mid = begin + (end - begin) / 2;
        float c   = tex1D<float>(EnvmapCdfY, mid + 0.5f);
        if (c >= r)
        {
            end = mid;
        }
        else
        {
            begin = mid + 1;
        }
    }

    return begin;
}

__device__ int sample_x(int y, float r)
{
    int begin = 0;
    int end   = HDRwidth - 1;
    while (end > begin)
    {
        int   mid = begin + (end - begin) / 2;
        float c   = tex2D<float>(EnvmapCdfX, mid + 0.5f, y + 0.5f);
        if (c >= r)
        {
            end = mid;
        }
        else
        {
            begin = mid + 1;
        }
    }

    return begin;
}

__host__ __device__ float luminance(const float4& c)
{
    return c.x * 0.2126 + c.y * 0.7152 + c.z * 0.0722;
}

__host__ __device__ float luminance(const vec3& c)
{
    return c.x * 0.2126 + c.y * 0.7152 + c.z * 0.0722;
}

__device__ vec3 eval_envmap(const vec3& raydir)
{
#if 0
    float longlatX = ((raydir.x > 0 ? atanf(raydir.z / raydir.x) : atanf(raydir.z / raydir.x) + M_PI) + M_PI * 0.5);
    float longlatY = acosf(raydir.y);  // add RotateMap at some point, see Fragmentarium

    // map theta and phi to u and v texturecoordinates in [0,1] x [0,1] range
    float offsetY = 0.5f;
    float u       = longlatX / TWO_PI;  // +offsetY;
    float v       = longlatY / M_PI;
#else
    float u, v;
    dir_to_uv(raydir, u, v);
#endif

    float4 HDRcol = tex2D<float4>(HDRtexture, u, v);  // fetch from texture
    return vec3(HDRcol);
}

// importance sample envmap
// input as randfs
// output as x-y coordinates
// returns pdf
__device__ float sample_envmap(float& u, float& v, vec3& c)
{
    int iy = sample_y(v);
    int ix = sample_x(iy, u);

    // TODO : hash uv as randfs for in-pixel offset
    u = ((float)ix + 0.5f) / (float)HDRwidth;
    v = ((float)iy + 0.5f) / (float)HDRheight;

    c = vec3(tex2D<float4>(HDRtexture, u, v));

#if MULT_PDF
    // any warp available
    float pdf = tex1D(EnvmapPdfY, iy + 0.5f) * tex2D(EnvmapPdfX, ix + 0.5f, iy + 0.5f);
    pdf *= HDRpdfnorm;
    float phi = v * M_PI;
    pdf /= sin(phi);
#else
    // consistent with sine warp
    float pdf = luminance(c) * HDRpdfnormAlt;
    float phi = v * M_PI;
#if !(PRE_WARP)
    pdf /= sin(phi);  // cancelled with the same factor in luminance of the original envmap
#endif
#endif

    return pdf;
}

__device__ float pdf_envmap(const vec3& dir, const vec3& dir_color)
{
    float u, v;
    dir_to_uv(dir, u, v);

#if MULT_PDF
    int ix = u * HDRwidth;
    int iy = v * HDRheight;
    // any warp available
    float pdf = tex1D(EnvmapPdfY, iy + 0.5f) * tex2D(EnvmapPdfX, ix + 0.5f, iy + 0.5f);
    pdf *= HDRpdfnorm;
    float phi = v * M_PI;
    pdf /= sin(phi);
#else
    // consistent with sine warp

    // float pdf = luminance(tex2D(HDRtexture, u, v)) * HDRpdfnormAlt;
    float pdf = luminance(dir_color) * HDRpdfnormAlt;

    float phi = v * M_PI;
#if !(PRE_WARP)
    pdf /= sin(phi);  // cancelled with the same factor in luminance of the original envmap
#endif
#endif

    return pdf;
}

static float build_cdf_1d(const float* f, float* pdf, float* cdf, int size)
{
    if (size < 1) return 0;

    float sum = 0.0f;
    for (int i = 0; i < size; i++) sum += f[i];
    float norm = 1.0f / sum;

    float I = 0.0f;
    for (int i = 0; i < size; i++)
    {
        float p = f[i] * norm;
        I += p;
        pdf[i] = p;
        cdf[i] = I;
    }
    cdf[size - 1] = 1.0f;

    return sum;
}

static float build_cdf_2d(
    const float* f, float* pdfY, float* cdfY, float* pdfX, float* cdfX, int width, int height)
{
    if (width < 1 || height < 1) return 0;

    stl_vector<float> row_sum(height);

    for (int y = 0; y < height; y++)
    {
        row_sum[y] = build_cdf_1d(f + y * width, pdfX + y * width, cdfX + y * width, width);
    }
    float sum = build_cdf_1d(row_sum.data(), pdfY, cdfY, height);
    return sum;
}

extern "C" void init_envmap(const float4* HDRmap, int width, int height)
{
    bool                  resized     = false;
    cudaChannelFormatDesc desc_float4 = cudaCreateChannelDesc<float4>();
    cudaChannelFormatDesc desc_float  = cudaCreateChannelDesc<float>();

    if (width != envmap_width || height != envmap_height)
    {
        // delay free array until texture rebound
        HDRtexture_t = HDRtexture_;
        EnvmapPdfY_t = EnvmapPdfY_;
        EnvmapCdfY_t = EnvmapCdfY_;
        EnvmapPdfX_t = EnvmapPdfX_;
        EnvmapCdfX_t = EnvmapCdfX_;

        envmap_width  = width;
        envmap_height = height;

        checkCudaErrors(cudaMemcpyToSymbolAsync(HDRwidth, &width, sizeof(int)));
        checkCudaErrors(cudaMemcpyToSymbolAsync(HDRheight, &height, sizeof(int)));

        // float pdfnorm = (float)width * (float)height * M_1_TWO_PI_PI;
        float pdfnorm = (float)width * (float)height / (M_PI * TWO_PI);
        checkCudaErrors(cudaMemcpyToSymbolAsync(HDRpdfnorm, &pdfnorm, sizeof(float)));

        // envmap color tex
        checkCudaErrors(cudaMallocArray(&HDRtexture_, &desc_float4, width, height));
        HDRtexture.filterMode = cudaFilterModePoint;
        HDRtexture.normalized = 1;
        checkCudaErrors(cudaBindTextureToArray(&HDRtexture, HDRtexture_, &desc_float4));

        // pdfY
        checkCudaErrors(cudaMallocArray(&EnvmapPdfY_, &desc_float, height));
        EnvmapPdfY.filterMode = cudaFilterModePoint;
        EnvmapPdfY.normalized = 0;
        checkCudaErrors(cudaBindTextureToArray(&EnvmapPdfY, EnvmapPdfY_, &desc_float));

        // cdfY
        checkCudaErrors(cudaMallocArray(&EnvmapCdfY_, &desc_float, height));
        EnvmapCdfY.filterMode = cudaFilterModePoint;
        EnvmapCdfY.normalized = 0;
        checkCudaErrors(cudaBindTextureToArray(&EnvmapCdfY, EnvmapCdfY_, &desc_float));

        // pdfX
        checkCudaErrors(cudaMallocArray(&EnvmapPdfX_, &desc_float, width, height));
        EnvmapPdfX.filterMode = cudaFilterModePoint;
        EnvmapPdfX.normalized = 0;
        checkCudaErrors(cudaBindTextureToArray(&EnvmapPdfX, EnvmapPdfX_, &desc_float));

        // cdfX
        checkCudaErrors(cudaMallocArray(&EnvmapCdfX_, &desc_float, width, height));
        EnvmapCdfX.filterMode = cudaFilterModePoint;
        EnvmapCdfX.normalized = 0;
        checkCudaErrors(cudaBindTextureToArray(&EnvmapCdfX, EnvmapCdfX_, &desc_float));

        resized = true;

        std::cout << "gpu envmap resized to " << width << " x " << height << std::endl;
    }

    // envmap color tex
    {
        checkCudaErrors(cudaMemcpy2DToArrayAsync(HDRtexture_,
                                                 0,
                                                 0,
                                                 HDRmap,
                                                 sizeof(float4) * width,
                                                 sizeof(float4) * width,
                                                 height,
                                                 cudaMemcpyHostToDevice));
    }

#if BUNDLE_TEST || (!(PASSIVE_ENVMAP))
    // cdf texture
    size_t            total = size_t(width) * size_t(height);
    stl_vector<float> lum(total);
    for (size_t i = 0; i < total; i++)
    {
        lum[i] = luminance(HDRmap[i]);
    }

#if PRE_WARP
    for (int y = 0; y < height; y++)
    {
        for (int x = 0; x < width; x++)
        {
            float phi = M_PI * (y + 0.5f) / height;
            lum[x + y * width] *= sin(phi);
        }
    }
#endif

    float lumsum = 0.0f;
    for (const auto& l : lum) lumsum += l;
    float pdfnormalt = (float)width * (float)height * M_1_TWO_PI_PI / lumsum;
    checkCudaErrors(cudaMemcpyToSymbolAsync(HDRpdfnormAlt, &pdfnormalt, sizeof(float)));

    stl_vector<float> pdfY(height);
    stl_vector<float> cdfY(height);
    stl_vector<float> pdfX(total);
    stl_vector<float> cdfX(total);
    build_cdf_2d(lum.data(), pdfY.data(), cdfY.data(), pdfX.data(), cdfX.data(), width, height);

    {
        // pdfY
        checkCudaErrors(cudaMemcpyToArray(
            EnvmapPdfY_, 0, 0, pdfY.data(), sizeof(float) * height, cudaMemcpyHostToDevice));
    }

    {
        // cdfY
        checkCudaErrors(cudaMemcpyToArray(
            EnvmapCdfY_, 0, 0, cdfY.data(), sizeof(float) * height, cudaMemcpyHostToDevice));
    }

    {
        // pdfX
        checkCudaErrors(cudaMemcpy2DToArray(EnvmapPdfX_,
                                            0,
                                            0,
                                            pdfX.data(),
                                            sizeof(float) * width,
                                            sizeof(float) * width,
                                            height,
                                            cudaMemcpyHostToDevice));
    }

    {
        // cdfX
        checkCudaErrors(cudaMemcpy2DToArray(EnvmapCdfX_,
                                            0,
                                            0,
                                            cdfX.data(),
                                            sizeof(float) * width,
                                            sizeof(float) * width,
                                            height,
                                            cudaMemcpyHostToDevice));
    }
#endif

    if (resized)
    {
        checkCudaErrors(cudaFreeArray(HDRtexture_t));
        checkCudaErrors(cudaFreeArray(EnvmapPdfY_t));
        checkCudaErrors(cudaFreeArray(EnvmapCdfY_t));
        checkCudaErrors(cudaFreeArray(EnvmapPdfX_t));
        checkCudaErrors(cudaFreeArray(EnvmapCdfX_t));
        HDRtexture_t = nullptr;
        EnvmapPdfY_t = nullptr;
        EnvmapCdfY_t = nullptr;
        EnvmapPdfX_t = nullptr;
        EnvmapCdfX_t = nullptr;
    }

    // checkCudaErrors(cudaDeviceSynchronize());
    initialized = true;
    // std::cout << "gpu envmap initialized" << std::endl;
}

extern "C" void free_envmap()
{
    if (!initialized) return;

    // checkCudaErrors(cudaUnbindTexture(&HDRtexture));
    // checkCudaErrors(cudaUnbindTexture(&EnvmapPdfY));
    // checkCudaErrors(cudaUnbindTexture(&EnvmapCdfY));
    // checkCudaErrors(cudaUnbindTexture(&EnvmapPdfX));
    // checkCudaErrors(cudaUnbindTexture(&EnvmapCdfX));

    checkCudaErrors(cudaFreeArray(HDRtexture_));
    checkCudaErrors(cudaFreeArray(EnvmapPdfY_));
    checkCudaErrors(cudaFreeArray(EnvmapCdfY_));
    checkCudaErrors(cudaFreeArray(EnvmapPdfX_));
    checkCudaErrors(cudaFreeArray(EnvmapCdfX_));

    // checkCudaErrors(cudaDeviceSynchronize());
    initialized = false;
    std::cout << "gpu envmap freed" << std::endl;
}

}  // namespace Envmap

__constant__ float3 sun_light_dir;
__constant__ float3 sun_light_power;
__constant__ float3 sun_light_power_original;

__device__ __forceinline__ float3 background(const float3& dir, int depth)
{
    // return make_float4(0.15f, 0.20f, 0.25f) * 0.5f * (dir.y + 0.5);
    // return (dir.y > -0.1f) ? make_float3(0.03, 0.07, 0.23) : make_float3(0.03, 0.03, 0.03);
#if SUN_LIGHT
    if (depth == 0 && (dot(dir, sun_light_dir) > 94.0f / sqrtf(94.0f * 94.0f + 0.45f * 0.45f)))
        return sun_light_power_original;
#endif
    return Envmap::eval_envmap(dir);
}

extern "C" void set_sun(float* sun_dir, float* sun_power)
{
    checkCudaErrors(cudaMemcpyToSymbolAsync(sun_light_power_original, sun_power, sizeof(float3)));

    // convert from disk light to directional light
    // sun's solid angle is its projected area on unit sphere
    float3 p = make_float3(sun_power[0], sun_power[1], sun_power[2]);
    float  r = 0.45 / 94.0f;
    p *= M_PI * (r * r);

    // const float3 light_dir   = make_float3(0.5826, 0.7660, 0.2717);
    // const float3 light_power = make_float3(2.6, 2.5, 2.3);
    checkCudaErrors(cudaMemcpyToSymbolAsync(sun_light_dir, sun_dir, sizeof(float3)));
    checkCudaErrors(cudaMemcpyToSymbolAsync(sun_light_power, &p.x, sizeof(float3)));
}

__global__ void __d_render(float4* d_output, CudaRng* rngs, const Param P)
{
    const float  density    = P.density;
    const float  brightness = P.brightness;
    const float3 albedo     = P.albedo;

#if DYNAMIC_TEXTURE
    const float3& boxMin = TextureVolume::density_tex.min;
    const float3& boxMax = TextureVolume::density_tex.max;
#else
    const float3& boxMin = TextureVolume::c_box_min;
    const float3& boxMax = TextureVolume::c_box_max;
#endif

    uint x = blockIdx.x * blockDim.x + threadIdx.x;
    uint y = blockIdx.y * blockDim.y + threadIdx.y;

    if ((x >= P.width) || (y >= P.height)) return;

#if !(FAST_RNG)
    CudaRng& rng = rngs[x + y * P.width];
#else
    CudaRng       rng;
    rng.init(x, y, (int)rngs);
#endif

    // float    u   = (x * 2.0f - P.width) / P.height;
    // float    v   = (y * 2.0f - P.height) / P.height;
    float u = (x * 2.0f - P.width) / P.width;
    float v = (y * 2.0f - P.height) / P.width;

    // calculate eye ray in world space
    float fovx = 54.43;

    Ray cr;
    cr.o = make_float3(mul(c_invViewMatrix, make_float4(0.0f, 0.0f, 0.0f, 1.0f)));
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
    float  max_sigma_t      = max_of(sigma_t_spectral.x, sigma_t_spectral.y, sigma_t_spectral.z);
#else
    float sigma_t = density;
#endif

    int i;
    for (i = 0; i < 10000; i++)
    {
        // find intersection with box
        float t_near, t_far;
        int   hit = intersectBox(cr, boxMin, boxMax, &t_near, &t_far);

        if (!hit)
        {
#if PASSIVE_ENVMAP
            radiance += background(cr.d, i) * throughput;
#else
            if (0 == i) radiance += background(cr.d, i) * throughput;
#endif
            break;
        }

        if (t_near < 0.0f)
        {
            t_near = 0.0f;  // clamp to near plane
        }

        /// woodcock tracking / delta tracking
        float3 pos  = cr.o + cr.d * t_near;  // current position
        float  dist = t_near;

        // hyperion trick
        float s = fmaxf(0.0f, fminf(1.0f, (i - 5) * 0.066666666666666666667f));
        float g = (1 - s) * P.g;
#if SPECTRAL_TRACKING
        float  density_prime      = (1 - s) * density + s * density * (1 - P.g);
        float  sigma_t_prime      = max_sigma_t * density_prime;  // max volume density is 1
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
                through = true;  // transmitted through the volume, probability is
                                 // 1-exp(-optical_thickness)
                break;
            }

#if SPECTRAL_TRACKING
            float  den            = vol_sigma_t(pos, density_prime);
            float3 sigma_t_den    = sigma_t_spectral * den;
            float3 sigma_s_den    = sigma_s_spectral * den;
            float3 sigma_a_den    = sigma_a_spectral * den;
            float3 sigma_null_den = make_float3(sigma_t_prime) - sigma_t_den;

            // history aware avg
            // float Pa = sum_of(fabsf(sigma_a_den.x * throughput.x),
            //                  fabsf(sigma_a_den.y * throughput.y),
            //                  fabsf(sigma_a_den.z * throughput.z));
            // float Ps = sum_of(fabsf(sigma_s_den.x * throughput.x),
            //                  fabsf(sigma_s_den.y * throughput.y),
            //                  fabsf(sigma_s_den.z * throughput.z));
            // float Pn = sum_of(fabsf(sigma_null_den.x * throughput.x),
            //                  fabsf(sigma_null_den.y * throughput.y),
            //                  fabsf(sigma_null_den.z * throughput.z));
            float Pa = 0;
            float Ps = sum_of(fabsf(sigma_t_den.x * throughput.x),
                              fabsf(sigma_t_den.y * throughput.y),
                              fabsf(sigma_t_den.z * throughput.z));
            float Pn = sum_of(fabsf(sigma_null_den.x * throughput.x),
                              fabsf(sigma_null_den.y * throughput.y),
                              fabsf(sigma_null_den.z * throughput.z));

            // history aware max
            // float Pa = max_of(fabsf(sigma_a_den.x * throughput.x),
            //                  fabsf(sigma_a_den.y * throughput.y),
            //                  fabsf(sigma_a_den.z * throughput.z));
            // float Ps = max_of(fabsf(sigma_s_den.x * throughput.x),
            //                  fabsf(sigma_s_den.y * throughput.y),
            //                  fabsf(sigma_s_den.z * throughput.z));
            // float Pn = max_of(fabsf(sigma_null_den.x * throughput.x),
            //                  fabsf(sigma_null_den.y * throughput.y),
            //                  fabsf(sigma_null_den.z * throughput.z));

            // probability normalizer
            float c = (Pa + Ps + Pn);

            // using reduced termination rates
            float e = rng.next() * c;
            if (e < Pa + Ps)
            {
                throughput *= sigma_s_den * (inv_sigma * c / (Pa + Ps));
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
#if PASSIVE_ENVMAP
            radiance += background(cr.d, i) * throughput;
#else
            if (0 == i) radiance += background(cr.d, i) * throughput;
#endif
            break;
        }

#if !(SPECTRAL_TRACKING)
        throughput *= albedo;
#endif

        Frame frame(cr.d);

        //
        // direct lighting
        //
        {
            // to match passive result
            float s = fmaxf(0.0f, fminf(1.0f, (i - 4) * 0.066666666666666666667f));  // (i + 1) - 5
            float g = (1 - s) * P.g;
#if SPECTRAL_TRACKING
            float  density_prime      = (1 - s) * density + s * density * (1 - P.g);
            float  sigma_t_prime      = max_sigma_t * density_prime;  // max volume density is 1
            float3 inv_sigma_spectral = make_float3(1.0f) / (sigma_t_spectral * density_prime);
#else
            float sigma_t_prime = (1 - s) * sigma_t + s * sigma_t * (1 - P.g);
#endif
            float inv_sigma = 1.0f / sigma_t_prime;  // max volume density is 1

#if SUN_LIGHT
#if SPECTRAL_TRACKING
            // reuse path and estimate for all channels (recommended)
            float3 a = Tr_spectral(boxMin,
                                   boxMax,
                                   pos,
                                   sun_light_dir * 1e10f,
                                   inv_sigma,
                                   density_prime,
                                   sigma_t_spectral,
                                   rng);
            radiance += sun_light_power * (throughput * phase.evaluate(frame, sun_light_dir) * a);
#else
            float a = Tr(boxMin, boxMax, pos, sun_light_dir * 1e10f, inv_sigma, sigma_t_prime, rng);
            radiance += light_power * (throughput * phase.evaluate(frame, sun_light_dir) * a);
#endif
#endif

#if !(PASSIVE_ENVMAP)
            //
            // one sample MIS
            //
            constexpr float P_phase  = 0.5f;
            constexpr float P_envmap = 1.0f - P_phase;
            // sample by phase function
            if (rng.next() < P_phase)
            {
                float u        = rng.next();
                float v        = rng.next();
                vec3  brdf_dir = phase.sample(frame, u, v);
                // brdf_dir      = normalize(brdf_dir);

                vec3  envc            = Envmap::eval_envmap(brdf_dir);
                float pdf_brdf        = phase.evaluate(frame, brdf_dir);  // perfect cancellation
                float pdf_env_virtual = Envmap::pdf_envmap(brdf_dir, envc);
                // float weight          = MIS_power_heuristic(pdf_brdf, pdf_env_virtual);
                float weight =
                    MIS_balance_heuristic(pdf_brdf * P_phase, pdf_env_virtual * P_envmap) / P_phase;

#if SPECTRAL_TRACKING
                // reuse path and estimate for all channels
                float3 a = Tr_spectral(boxMin,
                                       boxMax,
                                       pos,
                                       brdf_dir * 1e10f,
                                       inv_sigma,
                                       density_prime,
                                       sigma_t_spectral,
                                       rng);
                radiance += envc * (throughput /** phase.evaluate(frame, brdf_dir) / pdf_brdf*/ *
                                    weight * a);
#else
                float a = Tr(boxMin, boxMax, pos, brdf_dir * 1e10f, inv_sigma, sigma_t_prime, rng);
                radiance += envc * (throughput /** phase.evaluate(frame, brdf_dir) / pdf_brdf*/ *
                                    weight * a);
#endif
            }
            // sample by envmap pdf
            else
            {
                float u = rng.next();
                float v = rng.next();
                vec3  envc;
                float pdf_env = Envmap::sample_envmap(u, v, envc);
                if (pdf_env <= 0.0f) continue;

                vec3 envmap_dir = Envmap::uv_to_dir(u, v);
                // envmap_dir      = normalize(envmap_dir);

                float pdf_brdf_virtual = phase.evaluate(frame, envmap_dir);
                // float weight           = MIS_power_heuristic(pdf_env, pdf_brdf_virtual);
                float weight =
                    MIS_balance_heuristic(pdf_env * P_envmap, pdf_brdf_virtual * P_phase) /
                    P_envmap;

#if SPECTRAL_TRACKING
                // reuse path and estimate for all channels
                float3 a = Tr_spectral(boxMin,
                                       boxMax,
                                       pos,
                                       envmap_dir * 1e10f,
                                       inv_sigma,
                                       density_prime,
                                       sigma_t_spectral,
                                       rng);
                radiance +=
                    envc * (throughput * phase.evaluate(frame, envmap_dir) / pdf_env * weight * a);
#else
                float a =
                    Tr(boxMin, boxMax, pos, envmap_dir * 1e10f, inv_sigma, sigma_t_prime, rng);
                radiance +=
                    envc * (throughput * phase.evaluate(frame, envmap_dir) / pdf_env * weight * a);
#endif
            }
#endif
        }

        // scattered direction
        float3 new_dir = normalize(phase.sample(frame, rng.next(), rng.next()));
        cr.o           = pos;
        cr.d           = new_dir;
    }

    radiance *= brightness;

    // write output color
    float heat = i * 0.001;
#if MULTI_CHANNEL
    float4 color               = make_float4(0.0f, 0.0f, 0.0f, heat);
    GetChannel(color, channel) = fmaxf(GetChannel(radiance, channel), 0.0f) * 3.0f;
    d_output[x + y * P.width] += color;
#else
    d_output[x + y * P.width] += make_float4(
        fmaxf(radiance.x, 0.0f), fmaxf(radiance.y, 0.0f), fmaxf(radiance.z, 0.0f), heat);
#endif
}

__device__ float vol_bound(const float3& pos_)
{
#if USE_MODEL_TRANSFORM
    float3 pos = make_float3(mul(c_invModelMatrix, make_float4(pos_, 1.0f)));
#else
    const float3& pos = pos_;
#endif

#if USE_OPENVDB
#if DYNAMIC_TEXTURE
    // use world position for access
    float t = TextureVolume::density_bound_tex.fetch_gpu<float2>(pos).x;
    return t;
#else
    // remap position to [0, 1] coordinates
    float3 posn = pos * TextureVolume::c_world_to_normalized;
    float  t    = tex3D(TextureVolume::density_bound_tex,
                    posn.x * 0.5f + 0.5f,
                    posn.y * 0.5f + 0.5f,
                    posn.z * 0.5f + 0.5f)
                  .x;
    return t;
#endif
#else
    return 1.0f;
#endif
}

__device__ float2 vol_bound_minmax(const float3& pos_)
{
#if USE_MODEL_TRANSFORM
    float3 pos = make_float3(mul(c_invModelMatrix, make_float4(pos_, 1.0f)));
#else
    const float3& pos = pos_;
#endif

#if USE_OPENVDB
#if DYNAMIC_TEXTURE
    // use world position for access
    return TextureVolume::density_bound_tex.fetch_gpu<float2>(pos);
#else
    // remap position to [0, 1] coordinates
    float3 posn = pos * TextureVolume::c_world_to_normalized;
    return tex3D(TextureVolume::density_bound_tex,
                 posn.x * 0.5f + 0.5f,
                 posn.y * 0.5f + 0.5f,
                 posn.z * 0.5f + 0.5f);
#endif
#else
    return make_float2(1.0f, 0.0f);  // max, min
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
#if USE_MODEL_TRANSFORM
    float3 invR = make_float3(1.0f) / mul(c_invModelMatrix, r.d);
    float3 tbot = invR * (boxmin - make_float3(mul(c_invModelMatrix, make_float4(r.o, 1.0f))));
    float3 ttop = invR * (boxmax - make_float3(mul(c_invModelMatrix, make_float4(r.o, 1.0f))));
#else
    float3        invR   = make_float3(1.0f) / r.d;
    float3        tbot   = invR * (boxmin - r.o);
    float3        ttop   = invR * (boxmax - r.o);
#endif

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
__global__ void __d_render_bounded(float4* d_output, CudaRng* rngs, const Param P)
{
    const float  density    = P.density;
    const float  brightness = P.brightness;
    const float3 albedo     = P.albedo;

#if DYNAMIC_TEXTURE
    const float3& boxMin = TextureVolume::density_tex.min;
    const float3& boxMax = TextureVolume::density_tex.max;
#else
    const float3& boxMin = TextureVolume::c_box_min;
    const float3& boxMax = TextureVolume::c_box_max;
#endif

    uint x = blockIdx.x * blockDim.x + threadIdx.x;
    uint y = blockIdx.y * blockDim.y + threadIdx.y;

    if ((x >= P.width) || (y >= P.height)) return;

#if !(FAST_RNG)
    CudaRng& rng = rngs[x + y * P.width];
#else
    CudaRng       rng;
    rng.init(x, y, (int)rngs);
#endif

    // float    u   = (x * 2.0f - P.width) / P.height;
    // float    v   = (y * 2.0f - P.height) / P.height;
    float u = (x * 2.0f - P.width) / P.width;
    float v = (y * 2.0f - P.height) / P.width;

    // calculate eye ray in world space
    float fovx = 54.43;

    Ray cr;
    cr.o = make_float3(mul(c_invViewMatrix, make_float4(0.0f, 0.0f, 0.0f, 1.0f)));
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
    float  max_sigma_t      = max_of(sigma_t_spectral.x, sigma_t_spectral.y, sigma_t_spectral.z);
#else
    float sigma_t = density;
#endif

    int num_scatters = 0;

    int i;
    for (i = 0; i < 10000; i++)
    {
        // find intersection with box
        float t_near, t_far;
        float d_min, d_max;
        int   hit = intersectSuperVolume(cr, boxMin, boxMax, &t_near, &t_far, &d_min, &d_max);

        if (!hit)
        {
#if PASSIVE_ENVMAP
            radiance += background(cr.d, num_scatters) * throughput;
#else
            if (0 == num_scatters) radiance += background(cr.d, num_scatters) * throughput;
#endif
            break;
        }

        /// woodcock tracking / delta tracking
        float3 pos  = cr.o + cr.d * t_near;  // current position
        float  dist = t_near;

        // hyperion trick
        float s = fmaxf(0.0f, fminf(1.0f, (num_scatters - 5) * 0.066666666666666666667f));
        float g = (1 - s) * P.g;
        float reduction_factor = (1 - s) + s * (1 - P.g);
#if SPECTRAL_TRACKING
        float  density_prime = reduction_factor * density;
        float  sigma_t_prime = max_sigma_t * density_prime * d_max;  // max volume density is d_max
        float3 inv_sigma_spectral = make_float3(1.0f) / (sigma_t_spectral * density_prime);
#else
        float sigma_t_prime = reduction_factor * sigma_t;
#endif

        HGPhaseFunction phase(g);

        float inv_sigma = 1.0f / sigma_t_prime;  // max volume density is d_max

        bool through = false;
        // delta tracking scattering event sampling
        for (;;)
        {
            dist += -log(rng.next()) * inv_sigma;
            pos = cr.o + cr.d * dist;
            if (dist >= t_far)
            {
                through = true;  // transmitted through the volume, probability
                                 // is 1-exp(-optical_thickness)
                break;
            }

#if SPECTRAL_TRACKING
            float  den            = vol_sigma_t(pos, density_prime);
            float3 sigma_t_den    = sigma_t_spectral * den;
            float3 sigma_s_den    = sigma_s_spectral * den;
            float3 sigma_a_den    = sigma_a_spectral * den;
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
                throughput *= sigma_s_den * (inv_sigma * c / (Ps));
                ++num_scatters;
                break;
            }
            else
            {
                throughput *= sigma_null_den * (inv_sigma * c / Pn);
            }
#else
            if (rng.next() < vol_sigma_t(pos, sigma_t_prime) * inv_sigma)
            {
                ++num_scatters;
                break;
            }
#endif
        }

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

        //
        // direct lighting
        //
        {
            // to match passive result (note that num_scatters is already incremented)
            float s = fmaxf(0.0f, fminf(1.0f, (num_scatters - 5) * 0.066666666666666666667f));
            float g = (1 - s) * P.g;
            float reduction_factor = (1 - s) + s * (1 - P.g);
#if SPECTRAL_TRACKING
            float density_prime = reduction_factor * density;
            float sigma_t_prime =
                max_sigma_t * density_prime * d_max;  // max volume density is d_max
            float3 inv_sigma_spectral = make_float3(1.0f) / (sigma_t_spectral * density_prime);
#else
            float sigma_t_prime = reduction_factor * sigma_t;
#endif
            float inv_sigma = 1.0f / sigma_t_prime;  // max volume density is d_max

#if SUN_LIGHT
#if SPECTRAL_TRACKING
            // reuse path and estimate for all channels (recommended)
            float3 a = Tr_spectral(boxMin,
                                   boxMax,
                                   pos,
                                   sun_light_dir * 1e10f,
                                   inv_sigma,
                                   density_prime,
                                   sigma_t_spectral,
                                   rng);
            radiance += sun_light_power * (throughput * phase.evaluate(frame, sun_light_dir) * a);
#else
            float a = Tr(boxMin, boxMax, pos, sun_light_dir * 1e10f, inv_sigma, sigma_t_prime, rng);
            radiance += light_power * (throughput * phase.evaluate(frame, sun_light_dir) * a);
#endif
#endif

#if !(PASSIVE_ENVMAP)
            //
            // one sample MIS
            //
            constexpr float P_phase  = 0.5f;
            constexpr float P_envmap = 1.0f - P_phase;
            // sample by phase function
            if (rng.next() < P_phase)
            {
                float u        = rng.next();
                float v        = rng.next();
                vec3  brdf_dir = phase.sample(frame, u, v);
                // brdf_dir      = normalize(brdf_dir);

                vec3  envc            = Envmap::eval_envmap(brdf_dir);
                float pdf_brdf        = phase.evaluate(frame, brdf_dir);  // perfect cancellation
                float pdf_env_virtual = Envmap::pdf_envmap(brdf_dir, envc);
                // float weight          = MIS_power_heuristic(pdf_brdf, pdf_env_virtual);
                float weight =
                    MIS_balance_heuristic(pdf_brdf * P_phase, pdf_env_virtual * P_envmap) / P_phase;

#if SPECTRAL_TRACKING
                // reuse path and estimate for all channels
                float3 a = Tr_spectral(boxMin,
                                       boxMax,
                                       pos,
                                       brdf_dir * 1e10f,
                                       inv_sigma,
                                       density_prime,
                                       sigma_t_spectral,
                                       rng);
                radiance += envc * (throughput /** phase.evaluate(frame, brdf_dir) / pdf_brdf*/ *
                                    weight * a);
#else
                float a = Tr(boxMin, boxMax, pos, brdf_dir * 1e10f, inv_sigma, sigma_t_prime, rng);
                radiance += envc * (throughput /** phase.evaluate(frame, brdf_dir) / pdf_brdf*/ *
                                    weight * a);
#endif
            }
            // sample by envmap pdf
            else
            {
                float u = rng.next();
                float v = rng.next();
                vec3  envc;
                float pdf_env = Envmap::sample_envmap(u, v, envc);
                if (pdf_env <= 0.0f) continue;

                vec3 envmap_dir = Envmap::uv_to_dir(u, v);
                // envmap_dir      = normalize(envmap_dir);

                float pdf_brdf_virtual = phase.evaluate(frame, envmap_dir);
                // float weight           = MIS_power_heuristic(pdf_env, pdf_brdf_virtual);
                float weight =
                    MIS_balance_heuristic(pdf_env * P_envmap, pdf_brdf_virtual * P_phase) /
                    P_envmap;

#if SPECTRAL_TRACKING
                // reuse path and estimate for all channels
                float3 a = Tr_spectral(boxMin,
                                       boxMax,
                                       pos,
                                       envmap_dir * 1e10f,
                                       inv_sigma,
                                       density_prime,
                                       sigma_t_spectral,
                                       rng);
                radiance +=
                    envc * (throughput * phase.evaluate(frame, envmap_dir) / pdf_env * weight * a);
#else
                float a =
                    Tr(boxMin, boxMax, pos, envmap_dir * 1e10f, inv_sigma, sigma_t_prime, rng);
                radiance +=
                    envc * (throughput * phase.evaluate(frame, envmap_dir) / pdf_env * weight * a);
#endif
            }
#endif
        }

        // scattered direction
        float3 new_dir = normalize(phase.sample(frame, rng.next(), rng.next()));
        cr.o           = pos;
        cr.d           = new_dir;
    }

    radiance *= brightness;

    // write output color
    float heat = i * 0.001;
#if MULTI_CHANNEL
    float4 color               = make_float4(0.0f, 0.0f, 0.0f, heat);
    GetChannel(color, channel) = fmaxf(GetChannel(radiance, channel), 0.0f) * 3.0f;
    d_output[x + y * P.width] += color;
#else
    d_output[x + y * P.width] += make_float4(
        fmaxf(radiance.x, 0.0f), fmaxf(radiance.y, 0.0f), fmaxf(radiance.z, 0.0f), heat);
#endif
}

// using tracking restart (similar to regular tracking)
// if each TextureVolume::search_radius track length is exceeded
// the tracker is restarted with modified ray origin only
// and for each track a local max density is used to boost the mean free path
__global__ void __d_render_bounded_decomp(float4* d_output, CudaRng* rngs, const Param P)
{
    const float  density    = P.density;
    const float  brightness = P.brightness;
    const float3 albedo     = P.albedo;

#if DYNAMIC_TEXTURE
    const float3& boxMin = TextureVolume::density_tex.min;
    const float3& boxMax = TextureVolume::density_tex.max;
#else
    const float3& boxMin = TextureVolume::c_box_min;
    const float3& boxMax = TextureVolume::c_box_max;
#endif

    uint x = blockIdx.x * blockDim.x + threadIdx.x;
    uint y = blockIdx.y * blockDim.y + threadIdx.y;

    if ((x >= P.width) || (y >= P.height)) return;

#if !(FAST_RNG)
    CudaRng& rng = rngs[x + y * P.width];
#else
    CudaRng       rng;
    rng.init(x, y, (int)rngs);
#endif

    // float    u   = (x * 2.0f - P.width) / P.height;
    // float    v   = (y * 2.0f - P.height) / P.height;
    float u = (x * 2.0f - P.width) / P.width;
    float v = (y * 2.0f - P.height) / P.width;

    // calculate eye ray in world space
    float fovx = 54.43;

    Ray cr;
    cr.o = make_float3(mul(c_invViewMatrix, make_float4(0.0f, 0.0f, 0.0f, 1.0f)));
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
        int   hit = intersectSuperVolume(cr, boxMin, boxMax, &t_near, &t_far, &d_min, &d_max);
        bool  use_decomposition = d_min > 0.0f;  // faster than always using decomp
        // bool use_decomposition = false;

        if (!hit)
        {
#if PASSIVE_ENVMAP
            radiance += background(cr.d, num_scatters) * throughput;
#else
            if (0 == num_scatters) radiance += background(cr.d, num_scatters) * throughput;
#endif
            break;
        }

        /// woodcock tracking / delta tracking
        float3 pos  = cr.o + cr.d * t_near;  // current position
        float  dist = t_near;

        // hyperion trick
        float s = fmaxf(0.0f, fminf(1.0f, (num_scatters - 5) * 0.066666666666666666667f));
        float g = (1 - s) * P.g;
        float reduction_factor = (1 - s) + s * (1 - P.g);
#if SPECTRAL_TRACKING
        float  density_prime = reduction_factor * density;
        float  sigma_t_prime = max_sigma_t * density_prime * d_max;  // max volume density is d_max
        float3 inv_sigma_spectral = make_float3(1.0f) / (sigma_t_spectral * density_prime);

        // analog decomposition tracking
        if (use_decomposition)
        {
            sigma_c_prime    = min_sigma_t * density_prime * d_min;
            distc            = dist - log(rng.next()) / fmaxf(sigma_c_prime, 1e-20f);
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
        float inv_sigma     = 1.0f / sigma_t_prime;  // max volume density is d_max
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

        //
        // direct lighting
        //
        {
            // to match passive result (note that num_scatters is already incremented)
            float s = fmaxf(0.0f, fminf(1.0f, (num_scatters - 5) * 0.066666666666666666667f));
            float g = (1 - s) * P.g;
            float reduction_factor = (1 - s) + s * (1 - P.g);
#if SPECTRAL_TRACKING
            float density_prime = reduction_factor * density;
            float sigma_t_prime =
                max_sigma_t * density_prime * d_max;  // max volume density is d_max
#else
            float sigma_t_prime = reduction_factor * sigma_t;
#endif
            float inv_sigma = 1.0f / sigma_t_prime;  // max volume density is d_max

#if SUN_LIGHT
#if SPECTRAL_TRACKING
            // reuse path and estimate for all channels
            float3 a = Tr_spectral(boxMin,
                                   boxMax,
                                   pos,
                                   sun_light_dir * 1e10f,
                                   inv_sigma,
                                   density_prime,
                                   sigma_t_spectral,
                                   rng);
            radiance += sun_light_power * (throughput * phase.evaluate(frame, sun_light_dir) * a);
#else
            float a = Tr(boxMin, boxMax, pos, sun_light_dir * 1e10f, inv_sigma, sigma_t_prime, rng);
            radiance += light_power * (throughput * phase.evaluate(frame, sun_light_dir) * a);
#endif
#endif

#if !(PASSIVE_ENVMAP)
            //
            // one sample MIS
            //
            constexpr float P_phase  = 0.5f;
            constexpr float P_envmap = 1.0f - P_phase;
            // sample by phase function
            if (rng.next() < P_phase)
            {
                float u        = rng.next();
                float v        = rng.next();
                vec3  brdf_dir = phase.sample(frame, u, v);
                // brdf_dir      = normalize(brdf_dir);

                vec3  envc            = Envmap::eval_envmap(brdf_dir);
                float pdf_brdf        = phase.evaluate(frame, brdf_dir);  // perfect cancellation
                float pdf_env_virtual = Envmap::pdf_envmap(brdf_dir, envc);
                // float weight          = MIS_power_heuristic(pdf_brdf, pdf_env_virtual) / P_phase;
                float weight =
                    MIS_balance_heuristic(pdf_brdf * P_phase, pdf_env_virtual * P_envmap) / P_phase;

#if SPECTRAL_TRACKING
                // reuse path and estimate for all channels
                float3 a = Tr_spectral(boxMin,
                                       boxMax,
                                       pos,
                                       brdf_dir * 1e10f,
                                       inv_sigma,
                                       density_prime,
                                       sigma_t_spectral,
                                       rng);
                radiance += envc * (throughput /** phase.evaluate(frame, brdf_dir) / pdf_brdf*/ *
                                    weight * a);
#else
                float a = Tr(boxMin, boxMax, pos, brdf_dir * 1e10f, inv_sigma, sigma_t_prime, rng);
                radiance += envc * (throughput /** phase.evaluate(frame, brdf_dir) / pdf_brdf*/ *
                                    weight * a);
#endif
            }
            // sample by envmap pdf
            else
            {
                float u = rng.next();
                float v = rng.next();
                vec3  envc;
                float pdf_env = Envmap::sample_envmap(u, v, envc);
                if (pdf_env <= 0.0f) continue;

                vec3 envmap_dir = Envmap::uv_to_dir(u, v);
                // envmap_dir      = normalize(envmap_dir);

                float pdf_brdf_virtual = phase.evaluate(frame, envmap_dir);
                // float weight           = MIS_power_heuristic(pdf_env, pdf_brdf_virtual) / (1.0f -
                // P_phase);
                float weight =
                    MIS_balance_heuristic(pdf_env * P_envmap, pdf_brdf_virtual * P_phase) /
                    P_envmap;

#if SPECTRAL_TRACKING
                // reuse path and estimate for all channels
                float3 a = Tr_spectral(boxMin,
                                       boxMax,
                                       pos,
                                       envmap_dir * 1e10f,
                                       inv_sigma,
                                       density_prime,
                                       sigma_t_spectral,
                                       rng);
                radiance +=
                    envc * (throughput * phase.evaluate(frame, envmap_dir) / pdf_env * weight * a);
#else
                float a =
                    Tr(boxMin, boxMax, pos, envmap_dir * 1e10f, inv_sigma, sigma_t_prime, rng);
                radiance +=
                    envc * (throughput * phase.evaluate(frame, envmap_dir) / pdf_env * weight * a);
#endif
            }
#endif
        }

        // scattered direction
        float3 new_dir = normalize(phase.sample(frame, rng.next(), rng.next()));
        cr.o           = pos;
        cr.d           = new_dir;
    }

    radiance *= brightness;

    // write output color
    float heat = num_scatters * 0.001;
#if MULTI_CHANNEL
    float4 color               = make_float4(0.0f, 0.0f, 0.0f, heat);
    GetChannel(color, channel) = fmaxf(GetChannel(radiance, channel), 0.0f) * 3.0f;
    d_output[x + y * P.width] += color;
#else
    d_output[x + y * P.width] += make_float4(
        fmaxf(radiance.x, 0.0f), fmaxf(radiance.y, 0.0f), fmaxf(radiance.z, 0.0f), heat);
#endif
}

extern "C" void copy_inv_view_matrix(float* invViewMatrix, size_t sizeofMatrix)
{
    checkCudaErrors(cudaMemcpyToSymbolAsync(c_invViewMatrix, invViewMatrix, sizeofMatrix));
}

extern "C" void copy_inv_model_matrix(float* invModelMatrix, size_t sizeofMatrix)
{
    checkCudaErrors(cudaMemcpyToSymbolAsync(c_invModelMatrix, invModelMatrix, sizeofMatrix));
}

#if !(FAST_RNG)
namespace XORShift
{
// XOR shift PRNG
unsigned int        x = 123456789;
unsigned int        y = 362436069;
unsigned int        z = 521288629;
unsigned int        w = 88675123;
inline unsigned int frand()
{
    unsigned int t;
    t = x ^ (x << 11);
    x = y;
    y = z;
    z = w;
    return (w = (w ^ (w >> 19)) ^ (t ^ (t >> 8)));
}
}  // namespace XORShift

__global__ void __init_rng(CudaRng* rng, int width, int height, unsigned int* seeds)
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

extern "C" void init_rng(dim3 gridSize, dim3 blockSize, int width, int height)
{
    cout << "init cuda rng to " << width << " x " << height << endl;
    unsigned int* seeds;
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
    __init_rng<<<gridSize, blockSize>>>(cuda_rng, width, height, seeds);
    checkCudaErrors(cudaFree(seeds));
}

extern "C" void free_rng()
{
    cout << "free cuda rng" << endl;
    checkCudaErrors(cudaFree(cuda_rng));
}
#else
extern "C" void init_rng(dim3 gridSize, dim3 blockSize, int width, int height) {}
extern "C" void free_rng() {}
#endif

__global__ void __scale(float4* dst, float4* src, int size, float scale)
{
    int idx = threadIdx.x + blockIdx.x * blockDim.x;
    if (idx >= size)
    {
        return;
    }
    dst[idx] = src[idx] * scale;
}

extern "C" void scale(float4* dst, float4* src, int size, float scale)
{
    __scale<<<(size + 256 - 1) / 256, 256>>>(dst, src, size, scale);
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

extern "C" void render_kernel(
    dim3 gridSize, dim3 blockSize, float4* d_output, int spp, const Param& p)
{
#if FAST_RNG
    cuda_rng = (CudaRng*)spp;
#endif

    //__d_render<<<gridSize, blockSize>>>(d_output, cuda_rng, p);
    //__d_render_bounded<<<gridSize, blockSize>>>(d_output, cuda_rng, p);
    __d_render_bounded_decomp<<<gridSize, blockSize>>>(d_output, cuda_rng, p);
}
