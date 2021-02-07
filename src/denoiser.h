#pragma once

#include <cuda_runtime.h>
#include <optix.h>
#include <optix_stubs.h>
#include <memory>

namespace osc
{
struct CUDABuffer;
}
class CudaDenoiser
{
public:
    /*! @{ CUDA device context and stream that optix pipeline will run
    on, as well as device properties for this device */
    static CUcontext      cudaContext;
    static CUstream       stream;
    static cudaDeviceProp deviceProps;

    //! the optix context that our pipeline will run in.
    static OptixDeviceContext optixContext;

    bool accumulate = true;
    bool active     = true;

    int width, height;

    /*! the color buffer we use during _rendering_, which is a bit
        larger than the actual displayed frame buffer (to account for
        the border), and in float4 format (the denoiser requires
        floats) */
    std::unique_ptr<osc::CUDABuffer> renderBuffer;
    /*! the actual final color buffer used for display, in rgba8 */
    std::unique_ptr<osc::CUDABuffer> denoisedBuffer;

    OptixDenoiser                    denoiser = nullptr;
    std::unique_ptr<osc::CUDABuffer> denoiserScratch;
    std::unique_ptr<osc::CUDABuffer> denoiserState;

    static void context_log_cb(unsigned int level, const char* tag, const char* message, void*);

    /*! creates and configures a optix device context (in this simple
     example, only for the primary GPU device) */
    static void create_context();

    static void free_context();

    ~CudaDenoiser();

    CudaDenoiser(int width, int height);

    void denoise(int spp, float4* color_buffer);
};
