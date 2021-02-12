#include "denoiser.h"

#include "cuda_helpers.h"
#include <optix_function_table_definition.h>  // include once in the project

#include <cassert>
#include <iostream>
#include <vector>

#define checkOptixErrors(call)                                                                    \
    {                                                                                             \
        OptixResult res = call;                                                                   \
        if (res != OPTIX_SUCCESS)                                                                 \
        {                                                                                         \
            fprintf(                                                                              \
                stderr, "Optix call (%s) failed with code %d (line %d)\n", #call, res, __LINE__); \
            exit(2);                                                                              \
        }                                                                                         \
    }

/*! \namespace osc - Optix Siggraph Course */
namespace osc
{
/*! simple wrapper for creating, and managing a device-side CUDA
    buffer */
struct CUDABuffer
{
    inline CUdeviceptr d_pointer() const { return (CUdeviceptr)d_ptr; }

    //! re-size buffer to given number of bytes
    void resize(size_t size)
    {
        if (d_ptr) free();
        alloc(size);
    }

    //! allocate to given number of bytes
    void alloc(size_t size)
    {
        assert(d_ptr == nullptr);
        this->sizeInBytes = size;
        checkCudaErrors(cudaMalloc((void**)&d_ptr, sizeInBytes));
    }

    //! free allocated memory
    void free()
    {
        checkCudaErrors(cudaFree(d_ptr));
        d_ptr       = nullptr;
        sizeInBytes = 0;
    }

    template <typename T>
    void alloc_and_upload(const std::vector<T>& vt)
    {
        alloc(vt.size() * sizeof(T));
        upload((const T*)vt.data(), vt.size());
    }

    template <typename T>
    void upload(const T* t, size_t count)
    {
        assert(d_ptr != nullptr);
        assert(sizeInBytes == count * sizeof(T));
        checkCudaErrors(cudaMemcpy(d_ptr, (void*)t, count * sizeof(T), cudaMemcpyHostToDevice));
    }

    template <typename T>
    void download(T* t, size_t count)
    {
        assert(d_ptr != nullptr);
        assert(sizeInBytes == count * sizeof(T));
        checkCudaErrors(cudaMemcpy((void*)t, d_ptr, count * sizeof(T), cudaMemcpyDeviceToHost));
    }

    inline size_t size() const { return sizeInBytes; }
    size_t        sizeInBytes{0};
    void*         d_ptr{nullptr};
};
}  // namespace osc

void CudaDenoiser::context_log_cb(unsigned int level, const char* tag, const char* message, void*)
{
    fprintf(stderr, "[%2d][%12s]: %s\n", (int)level, tag, message);
}

/*! creates and configures a optix device context (in this simple
     example, only for the primary GPU device) */

void CudaDenoiser::create_context()
{
    checkOptixErrors(optixInit());

    // for this sample, do everything on one device
    const int deviceID = 0;
    checkCudaErrors(cudaSetDevice(deviceID));
    checkCudaErrors(cudaStreamCreate(&stream));

    cudaGetDeviceProperties(&deviceProps, deviceID);
    std::cout << "#osc: running on device: " << deviceProps.name << std::endl;

    CUresult cuRes = cuCtxGetCurrent(&cudaContext);
    if (cuRes != CUDA_SUCCESS)
        fprintf(stderr, "Error querying current context: error code %d\n", cuRes);

    checkOptixErrors(optixDeviceContextCreate(cudaContext, 0, &optixContext));
    checkOptixErrors(optixDeviceContextSetLogCallback(optixContext, context_log_cb, nullptr, 4));
}

void CudaDenoiser::free_context()
{
    checkCudaErrors(cudaStreamDestroy(stream));
    checkOptixErrors(optixDeviceContextDestroy(optixContext));
}

CudaDenoiser::~CudaDenoiser() { checkOptixErrors(optixDenoiserDestroy(denoiser)); }

CudaDenoiser::CudaDenoiser(int width, int height) : width(width), height(height)
{
    renderBuffer    = std::make_unique<osc::CUDABuffer>();
    denoisedBuffer  = std::make_unique<osc::CUDABuffer>();
    denoiserScratch = std::make_unique<osc::CUDABuffer>();
    denoiserState   = std::make_unique<osc::CUDABuffer>();

    // ------------------------------------------------------------------
    // create the denoiser:
    OptixDenoiserOptions denoiserOptions = {};

    denoiserOptions.inputKind = OPTIX_DENOISER_INPUT_RGB;

#if OPTIX_VERSION < 70100
    // these only exist in 7.0, not 7.1
    denoiserOptions.pixelFormat = OPTIX_PIXEL_FORMAT_FLOAT4;
#endif

    checkOptixErrors(optixDenoiserCreate(optixContext, &denoiserOptions, &denoiser));
    checkOptixErrors(optixDenoiserSetModel(denoiser, OPTIX_DENOISER_MODEL_KIND_LDR, NULL, 0));

    // .. then compute and allocate memory resources for the denoiser
    OptixDenoiserSizes denoiserReturnSizes;
    checkOptixErrors(
        optixDenoiserComputeMemoryResources(denoiser, width, height, &denoiserReturnSizes));

#if OPTIX_VERSION < 70100
    denoiserScratch->resize(denoiserReturnSizes.recommendedScratchSizeInBytes);
#else
    denoiserScratch->resize(std::max(denoiserReturnSizes.withOverlapScratchSizeInBytes,
                                     denoiserReturnSizes.withoutOverlapScratchSizeInBytes));
#endif
    denoiserState->resize(denoiserReturnSizes.stateSizeInBytes);

    // ------------------------------------------------------------------
    // resize our cuda frame buffer
    denoisedBuffer->resize(width * height * sizeof(float4));
    renderBuffer->resize(width * height * sizeof(float4));

    // ------------------------------------------------------------------
    checkOptixErrors(optixDenoiserSetup(denoiser,
                                        0,
                                        width,
                                        height,
                                        denoiserState->d_pointer(),
                                        denoiserState->size(),
                                        denoiserScratch->d_pointer(),
                                        denoiserScratch->size()));
}

void CudaDenoiser::denoise(int spp, float4* color_buffer)
{
    OptixDenoiserParams denoiserParams;
    denoiserParams.denoiseAlpha = 1;
    denoiserParams.hdrIntensity = (CUdeviceptr)0;
    denoiserParams.blendFactor  = accumulate ? 1.f / (spp) : 1.0f;

    // -------------------------------------------------------
    OptixImage2D inputLayer;
    inputLayer.data = (CUdeviceptr)color_buffer;
    /// Width of the image (in pixels)
    inputLayer.width = width;
    /// Height of the image (in pixels)
    inputLayer.height = height;
    /// Stride between subsequent rows of the image (in bytes).
    inputLayer.rowStrideInBytes = width * sizeof(float4);
    /// Stride between subsequent pixels of the image (in bytes).
    /// For now, only 0 or the value that corresponds to a dense packing of
    /// pixels (no gaps) is supported.
    inputLayer.pixelStrideInBytes = sizeof(float4);
    /// Pixel format.
    inputLayer.format = OPTIX_PIXEL_FORMAT_FLOAT4;

    // -------------------------------------------------------
    OptixImage2D outputLayer;
    outputLayer.data = denoisedBuffer->d_pointer();
    /// Width of the image (in pixels)
    outputLayer.width = width;
    /// Height of the image (in pixels)
    outputLayer.height = height;
    /// Stride between subsequent rows of the image (in bytes).
    outputLayer.rowStrideInBytes = width * sizeof(float4);
    /// Stride between subsequent pixels of the image (in bytes).
    /// For now, only 0 or the value that corresponds to a dense packing of
    /// pixels (no gaps) is supported.
    outputLayer.pixelStrideInBytes = sizeof(float4);
    /// Pixel format.
    outputLayer.format = OPTIX_PIXEL_FORMAT_FLOAT4;

    // -------------------------------------------------------
    if (active)
    {
        checkOptixErrors(optixDenoiserInvoke(denoiser,
                                             /*stream*/ 0,
                                             &denoiserParams,
                                             denoiserState->d_pointer(),
                                             denoiserState->size(),
                                             &inputLayer,
                                             1,
                                             /*inputOffsetX*/ 0,
                                             /*inputOffsetY*/ 0,
                                             &outputLayer,
                                             denoiserScratch->d_pointer(),
                                             denoiserScratch->size()));
    }
    else
    {
        cudaMemcpy((void*)outputLayer.data,
                   (void*)inputLayer.data,
                   outputLayer.width * outputLayer.height * sizeof(float4),
                   cudaMemcpyDeviceToDevice);
    }

    cudaMemcpy((void*)color_buffer,
               (void*)outputLayer.data,
               outputLayer.width * outputLayer.height * sizeof(float4),
               cudaMemcpyDeviceToDevice);

    // sync - make sure the frame is rendered before we download and
    // display (obviously, for a high-performance application you
    // want to use streams and double-buffering, but for this simple
    // example, this will have to do)
    checkCudaErrors(cudaDeviceSynchronize());
}

CUcontext          CudaDenoiser::cudaContext;
CUstream           CudaDenoiser::stream;
cudaDeviceProp     CudaDenoiser::deviceProps;
OptixDeviceContext CudaDenoiser::optixContext;
