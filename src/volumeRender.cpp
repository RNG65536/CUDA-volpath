#include <iostream>
#include <algorithm>
#include <glm/glm.hpp>
#include <glm/gtc/type_ptr.hpp>
#include <glm/gtc/matrix_transform.hpp>
using std::cout;
using std::endl;

#include <GL/glew.h>
#include <GL/freeglut.h>
#include <cuda_runtime.h>
#include <cuda_gl_interop.h>
#include <vector_types.h>
#include <vector_functions.h>
#include <helper_cuda.h>
#include <helper_timer.h>
#include <random>
#include <chrono>

#include "param.h"

#if USE_OPENVDB
#include <load_vdb.h>
#endif

std::vector<Param> params;

bool linearFiltering = false;

typedef unsigned int uint;
typedef unsigned char uchar;

static std::default_random_engine rng;
static std::uniform_real_distribution<float> dist;
static float randf()
{
    return dist(rng);
}

class Timer
{
private:
    std::chrono::high_resolution_clock::time_point last;

public:
    Timer() { record(); }

    void record() { last = std::chrono::high_resolution_clock::now(); }

    float elapsed() const
    {
        return std::chrono::duration_cast<std::chrono::microseconds>(
                   std::chrono::high_resolution_clock::now() - last)
            .count();
    }
};

const char* sSDKsample = "CUDA 3D Volume Render";

typedef unsigned char VolumeType;

dim3 blockSize(8, 8);
dim3 gridSize;

GLuint pbo = 0;     // OpenGL pixel buffer object
GLuint tex = 0;     // OpenGL texture object

//float3 viewRotation = make_float3(20, -20, 0);
//float3 viewTranslation = make_float3(0.0, 0.0, -4.0f);
float3 viewRotation    = make_float3(-12, -90, 0);
float3 viewTranslation = make_float3(0.03, -0.05, -4.0);

extern "C" void render_kernel(dim3 gridSize, dim3 blockSize, float4 *d_output, int spp, const Param& p);
extern "C" void copy_inv_view_matrix(float *invViewMatrix, size_t sizeofMatrix);
extern "C" void init_rng(dim3 gridSize, dim3 blockSize, int width, int height);
extern "C" void free_rng();
extern "C" void scale(float4 *dst, float4 *src, int size, float scale);
extern "C" void gamma_correct(float4 *dst, float4 *src, int size, float scale, float gamma);

namespace TextureVolume
{
extern "C" void init_cuda(void *h_volume, cudaExtent volumeSize);
extern "C" void set_texture_filter_mode(bool bLinearFilter);
extern "C" void free_cuda_buffers();
}

void initPixelBuffer();

class SdkTimer
{
    StopWatchInterface *timer = 0;

public:
    SdkTimer()
    {
        sdkCreateTimer(&timer);


    }
    ~SdkTimer()
    {
        sdkDeleteTimer(&timer);
    }

    void start()
    {
        sdkStartTimer(&timer);
    }

    void stop()
    {
        sdkStopTimer(&timer);
    }

    float fps()
    {
        return 1.0f / (sdkGetAverageTimerValue(&timer) / 1000.0f);
    }

    void reset()
    {
        sdkResetTimer(&timer);
    }
};

template <typename T>
class PboResource
{
    struct cudaGraphicsResource *cuda_pbo_resource; // CUDA Graphics Resource (to transfer PBO)
    bool mapped = false;

public:
    PboResource(unsigned int pbo)
    {
        // register this buffer object with CUDA
        checkCudaErrors(cudaGraphicsGLRegisterBuffer(&cuda_pbo_resource, pbo, cudaGraphicsMapFlagsWriteDiscard));
    }
    ~PboResource()
    {
        if (mapped)
        {
            unmap();
        }
        // unregister this buffer object from CUDA C
        checkCudaErrors(cudaGraphicsUnregisterResource(cuda_pbo_resource));
    }
    T *map()
    {
        T *d_output;
        // map PBO to get CUDA device pointer
        checkCudaErrors(cudaGraphicsMapResources(1, &cuda_pbo_resource, 0));
        size_t num_bytes;
        checkCudaErrors(cudaGraphicsResourceGetMappedPointer((void **)&d_output, &num_bytes,
            cuda_pbo_resource));
        //printf("CUDA mapped PBO: May access %ld bytes\n", num_bytes);
        mapped = true;
        return d_output;
    }

    void unmap()
    {
        checkCudaErrors(cudaGraphicsUnmapResources(1, &cuda_pbo_resource, 0));
        mapped = false;
    }
};
//////////////////////////////////////////////////////////////////////////

SdkTimer timer;
PboResource<float4> *resource = nullptr;
Param P;

class CudaFrameBuffer
{
    int width, height;
    int spp = 0;
    float4 *sum_buffer;

public:
    CudaFrameBuffer(int w, int h)
    {
        width = w;
        height = h;
        checkCudaErrors(cudaMalloc((void**)&sum_buffer, width * height * sizeof(float4)));
        // clear image
        checkCudaErrors(cudaMemset(sum_buffer, 0, width * height * sizeof(float4)));
    }
    ~CudaFrameBuffer()
    {
        checkCudaErrors(cudaFree(sum_buffer));
    }
    void reset()
    {
        spp = 0;
        // clear image
        checkCudaErrors(cudaMemset(sum_buffer, 0, width * height * sizeof(float4)));
    }
    float4 *ptr()
    {
        return sum_buffer;
    }
    void incrementSPP()
    {
        spp++;
    }
    void scaledOutput(float4 *dst)
    {
        scale(dst, sum_buffer, width * height, 1.0f / spp);
    }
    void gammaCorrectedOutput(float4 *dst, float gamma)
    {
        gamma_correct(dst, sum_buffer, width * height, 1.0f / spp, gamma);
    }
    int samplesPerPixel() const
    {
        return spp;
    }
};

CudaFrameBuffer *fb = nullptr;

int fpsCount = 0;        // FPS count for averaging
int fpsLimit = 1;        // FPS limit for sampling
unsigned int frameCount = 0;

void computeFPS()
{
    frameCount++;
    fpsCount++;

    if (fpsCount == fpsLimit)
    {
        char fps[256];
        float ifps = timer.fps();
        sprintf(fps, "Volume Render: %3.1f fps @ %d spp", ifps, fb->samplesPerPixel());

        glutSetWindowTitle(fps);
        fpsCount = 0;

        fpsLimit = (int)std::max(1.0f, ifps);
        timer.reset();
    }
}


// render image using CUDA
void cuda_volpath()
{
    // build inverse view matrix and convert to row major
    glm::mat4 model = glm::mat4(1.0f);
    model = glm::rotate(model, glm::radians(-viewRotation.y), glm::vec3(0.0, 1.0, 0.0));
    model = glm::rotate(model, glm::radians(-viewRotation.x), glm::vec3(1.0, 0.0, 0.0));
    model = glm::translate(model, glm::vec3(-viewTranslation.x, -viewTranslation.y, -viewTranslation.z));
    model = glm::transpose(model);

    copy_inv_view_matrix(glm::value_ptr(model), sizeof(float4)*3);

    // map PBO to get CUDA device pointer
    float4 *d_output = resource->map();

    // call CUDA kernel, writing results to PBO
    Timer timer;
    render_kernel(gridSize, blockSize, fb->ptr(), fb->samplesPerPixel(), P);
    checkCudaErrors(cudaDeviceSynchronize());
    float elapsed = timer.elapsed();
    printf("%f M samples / s, %d x %d, %f\n", (float)P.width * (float)P.height / elapsed, P.width, P.height, elapsed);

    fb->incrementSPP();
    //fb->scaledOutput(d_output);
    fb->gammaCorrectedOutput(d_output, 2.2f);

    getLastCudaError("kernel failed");

    resource->unmap();
}

// display results using OpenGL (called by GLUT)
void display()
{
    timer.start();

    cuda_volpath();

    // display results
    glClear(GL_COLOR_BUFFER_BIT);
    glDisable(GL_DEPTH_TEST);

    // draw image from PBO
    glPixelStorei(GL_UNPACK_ALIGNMENT, 1);

    // draw using texture
    // copy from pbo to texture
    glBindBuffer(GL_PIXEL_UNPACK_BUFFER, pbo);
    glBindTexture(GL_TEXTURE_2D, tex);
    glTexSubImage2D(GL_TEXTURE_2D, 0, 0, 0, P.width, P.height, GL_RGBA, GL_FLOAT, 0);
    glBindBuffer(GL_PIXEL_UNPACK_BUFFER, 0);

    // draw textured quad
    glEnable(GL_TEXTURE_2D);
    glBegin(GL_QUADS);
    glTexCoord2f(0, 0);
    glVertex2f(0, 0);
    glTexCoord2f(1, 0);
    glVertex2f(1, 0);
    glTexCoord2f(1, 1);
    glVertex2f(1, 1);
    glTexCoord2f(0, 1);
    glVertex2f(0, 1);
    glEnd();

    glDisable(GL_TEXTURE_2D);
    glBindTexture(GL_TEXTURE_2D, 0);

    glutSwapBuffers();
    glutReportErrors();

    timer.stop();

    computeFPS();
}

void idle()
{
    glutPostRedisplay();
}

void keyboard(unsigned char key, int x, int y)
{
    switch (key)
    {
        case 27:
        case 'q':
            glutDestroyWindow(glutGetWindow());
            return;

        case 'f':
            linearFiltering = !linearFiltering;
            TextureVolume::set_texture_filter_mode(linearFiltering);
            break;

        case '+':
        case '=':
            P.density += 1;
            break;

        case '-':
            P.density -= 1;
            P.density = std::max(P.density, 0.0f);
            break;

        case ']':
            P.brightness += 0.1f;
            break;

        case '[':
            P.brightness -= 0.1f;
            break;

        case 'x':
            P.albedo.x = std::max(0.0f, std::min(P.albedo.x + 0.01f, 1.0f));
            P.albedo.y = std::max(0.0f, std::min(P.albedo.y + 0.01f, 1.0f));
            P.albedo.z = std::max(0.0f, std::min(P.albedo.z + 0.01f, 1.0f));
            break;

        case 'z':
            P.albedo.x = std::max(0.0f, std::min(P.albedo.x - 0.01f, 1.0f));
            P.albedo.y = std::max(0.0f, std::min(P.albedo.y - 0.01f, 1.0f));
            P.albedo.z = std::max(0.0f, std::min(P.albedo.z - 0.01f, 1.0f));
            break;

        case 's':
            P.g += 0.01;
            P.g = std::max(-1.0f, std::min(P.g, 1.0f));
            break;

        case 'a':
            P.g -= 0.01;
            P.g = std::max(-1.0f, std::min(P.g, 1.0f));
            break;

        case ' ':
            P = params[rand() % params.size()];
            break;

        default:
            break;
    }

    printf("density = %.2f, brightness = %.2f, ", P.density, P.brightness);
    printf("albedo = %.2f, %.2f, %.2f, g = %.2f\n", P.albedo.x, P.albedo.y, P.albedo.z, P.g);
    glutPostRedisplay();

    fb->reset();
}

int ox, oy;
int buttonState = 0;

void mouse(int button, int state, int x, int y)
{
    if (state == GLUT_DOWN)
    {
        buttonState  |= 1<<button;
    }
    else if (state == GLUT_UP)
    {
        buttonState = 0;
    }

    ox = x;
    oy = y;
    glutPostRedisplay();
}

void motion(int x, int y)
{
    if (x == ox && y == oy) return;

    float dx, dy;
    dx = (float)(x - ox);
    dy = (float)(y - oy);

    if (buttonState == 4)
    {
        // middle = translate
        viewTranslation.x += dx / 100.0f;
        viewTranslation.y -= dy / 100.0f;
    }
    else if (buttonState == 2)
    {
        // right = zoom
        viewTranslation.z += dy / 100.0f;
    }
    else if (buttonState == 1)
    {
        // left = rotate
        viewRotation.x += dy / 5.0f;
        viewRotation.y += dx / 5.0f;
    }

    ox = x;
    oy = y;
    glutPostRedisplay();

    fb->reset();
}

int divideUp(int a, int b)
{
    return (a + b - 1) / b;
}

void reshape(int w, int h)
{
    P.width = w;
    P.height = h;
    initPixelBuffer();

    // calculate new grid size
    gridSize = dim3(divideUp(P.width, blockSize.x), divideUp(P.height, blockSize.y));
    free_rng();
    init_rng(gridSize, blockSize, P.width, P.height);

    glViewport(0, 0, w, h);

    glMatrixMode(GL_MODELVIEW);
    glLoadIdentity();

    glMatrixMode(GL_PROJECTION);
    glLoadIdentity();
    glOrtho(0.0, 1.0, 0.0, 1.0, 0.0, 1.0);

    delete fb;
    fb = new CudaFrameBuffer(P.width, P.height);
}

void cleanup()
{
    TextureVolume::free_cuda_buffers();
    free_rng();

    if (pbo)
    {
        delete resource;
        glDeleteBuffers(1, &pbo);
        glDeleteTextures(1, &tex);

        delete fb;
    }

    // cudaDeviceReset causes the driver to clean up all state. While
    // not mandatory in normal operation, it is good practice.  It is also
    // needed to ensure correct operation when the application is being
    // profiled. Calling cudaDeviceReset causes all profile data to be
    // flushed before the application exits
    cudaDeviceReset();
}

void initPixelBuffer()
{
    if (pbo)
    {
        delete resource;

        // delete old buffer
        glDeleteBuffers(1, &pbo);
        glDeleteTextures(1, &tex);

        delete fb;
    }

    // create pixel buffer object for display
    glGenBuffers(1, &pbo);
    glBindBuffer(GL_PIXEL_UNPACK_BUFFER, pbo);
    glBufferData(GL_PIXEL_UNPACK_BUFFER, P.width * P.height * sizeof(GLfloat) * 4, 0, GL_STREAM_DRAW);
    glBindBuffer(GL_PIXEL_UNPACK_BUFFER, 0);

    resource = new PboResource<float4>(pbo);
    fb = new CudaFrameBuffer(P.width, P.height);

    // create texture for display
    glGenTextures(1, &tex);
    glBindTexture(GL_TEXTURE_2D, tex);
    glTexImage2D(GL_TEXTURE_2D, 0, GL_RGBA32F, P.width, P.height, 0, GL_RGBA, GL_FLOAT, NULL);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_NEAREST);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_NEAREST);
    glBindTexture(GL_TEXTURE_2D, 0);
}

// Load raw data from disk
void *loadRawFile(char *filename, size_t size)
{
    FILE *fp = fopen(filename, "rb");

    if (!fp)
    {
        fprintf(stderr, "Error opening file '%s'\n", filename);
        return nullptr;
    }

    void *data = malloc(size);
    size_t read = fread(data, 1, size, fp);
    fclose(fp);

    printf("Read '%s', %zu bytes\n", filename, read);

    return data;
}

void* loadBinaryFile(char* filename, int &width, int &height, int &depth)
{
    FILE *fp = fopen(filename, "rb");
    if (!fp)
    {
        fclose(fp);
        fprintf(stderr, "Error opening file '%s'\n", filename);
        return nullptr;
    }

    fread(&width, sizeof(int), 1, fp);
    fread(&height, sizeof(int), 1, fp);
    fread(&depth, sizeof(int), 1, fp);

    if (width < 0 || height < 0 || depth < 0)
    {
        fclose(fp);
        fprintf(stderr, "Invalid resolution of file '%s'\n", filename);
        return nullptr;
    }

    size_t total = size_t(width) * size_t(height) * size_t(depth);
    if (total > 1llu << 33)
    {
        fclose(fp);
        fprintf(stderr, "Resolution too large of file '%s'\n", filename);
        return nullptr;
    }

    float* dataf = reinterpret_cast<float*>(malloc(sizeof(float) * total));
    size_t read = fread(dataf, sizeof(float), width * height * depth, fp);
    fclose(fp);
    printf("Read '%s', %zu bytes\n", filename, read);

    // normalize the volume values
    VolumeType* data = reinterpret_cast<VolumeType*>(malloc(total));
    for (size_t i = 0; i < total; i++)
    {
        data[i] = VolumeType(std::max(0.0f, std::min(dataf[i], 1.0f)) * 255.0f);
    }
    free(dataf);

    return data;
}

#if USE_OPENVDB
void* loadVdbFile(char* filename, int& width, int& height, int& depth)
{
    FILE* f = fopen(filename, "r");
    if (!f)
    {
        fprintf(stderr, "cannot open file: %s\n", filename);
        exit(1);
    }
    fclose(f);

    float min_value, max_value;
    auto dataf = load_vdb(filename, width, height, depth, min_value, max_value);
    max_value = std::max(max_value, 0.0001f);

    if (!dataf)
    {
        fprintf(stderr, "Error opening file '%s'\n", filename);
        free(dataf);
        return nullptr;
    }

    if (width < 0 || height < 0 || depth < 0)
    {
        fprintf(stderr, "Invalid resolution of file '%s'\n", filename);
        free(dataf);
        return nullptr;
    }

    size_t total = size_t(width) * size_t(height) * size_t(depth);
    if (total > 1llu << 33)
    {
        fprintf(stderr, "Resolution too large of file '%s'\n", filename);
        free(dataf);
        return nullptr;
    }

    // normalize the volume values
    VolumeType* data = reinterpret_cast<VolumeType*>(malloc(total));
    for (size_t i = 0; i < total; i++)
    {
        //data[i] = VolumeType(std::max(0.0f, std::min(dataf[i], 1.0f)) * 255.0f);
        data[i] = VolumeType(std::max(0.0f, dataf[i]) / max_value * 255.0f);
    }
    free(dataf);

    return data;
}

template <typename T, int MaxSize>
class CircularBuffer
{
private:
    T      _data[MaxSize];
    size_t _begin = 0;
    size_t _size  = 0;

public:
    void push_back(const T& v)
    {
        if (_size < MaxSize)
        {
            size_t _end = (_begin + _size) % MaxSize;
            _data[_end] = v;
            ++_size;
        }
    }
    void pop_back()
    {
        if (_size > 0)
        {
            --_size;
        }
    }
    void pop_front()
    {
        if (_size > 0)
        {
            _begin = (_begin + 1) % MaxSize;
            --_size;
        }
    }
    // void push_front();

    void clear()
    {
        _begin = 0;
        _size  = 0;
    }
    bool empty() const { return _size == 0; }

    const T& front() const { return _data[_begin]; }
    const T& back() const
    {
        size_t _end = (_begin + _size - 1) % MaxSize;
        return _data[_end];
    }

    size_t size() const { return _size; }
};

template <typename Data, typename Data2>
Data2 make(const Data& a, const Data& b);

template <>
uchar2 make<unsigned char, uchar2>(const unsigned char& x, const unsigned char& y)
{
    return make_uchar2(x, y);
}

template <>
float2 make<float, float2>(const float& x, const float& y)
{
    return make_float2(x, y);
}

// find volume density bound (min, max) around each voxel
template <typename Data, typename Data2>
Data2* compute_volume_value_bound_(const Data*       volume,
                                   const cudaExtent& extent,
                                   float             search_radius)
{
    size_t size           = extent.width * extent.height * extent.depth;
    Data2* bound_volume   = new Data2[size];
    Data2* bound_volume_2 = new Data2[size];

    float cell_size = 2.0f / extent.width;

    // to ensure the diagonal is covered
    int diffusion_iters = ceil(search_radius / cell_size);
    printf_s("diffusion iters = %d\n", diffusion_iters);

    constexpr size_t buffer_size = 256; // must be larger than diffusion_iters * 2 + 1

    int nx  = extent.width;
    int ny  = extent.height;
    int nz  = extent.depth;
    int nxy = nx * ny;

    for (size_t i = 0; i < size; i++)
    {
        bound_volume[i] = make<Data, Data2>(volume[i], volume[i]);
    }

    int window_size = diffusion_iters * 2 + 1;

    Timer timer;
    float elapsed;

    for (int sweep_dir = 0; sweep_dir < 3; sweep_dir++)
    {
        switch (sweep_dir)
        {
        default:
            break;
        case 0:
            timer.record();

#pragma omp parallel for
            for (int k = 0; k < nz; k++)
            for (int j = 0; j < ny; j++)
            {
                CircularBuffer<int, buffer_size> max_window;
                CircularBuffer<int, buffer_size> min_window;
                size_t offset = j * nx + k * nxy;
                for (int i = 0; i < nx + diffusion_iters + 1; i++)
                {
                    if (i > diffusion_iters)
                    {
                        Data dmax = bound_volume[max_window.front() + offset].x;
                        Data dmin = bound_volume[min_window.front() + offset].y;
                        bound_volume_2[(i - diffusion_iters - 1) + offset] = make<Data, Data2>(dmax, dmin);
                    }
                    if (i < nx)
                    {
                        Data2 d = bound_volume[i + offset];
                        while (!max_window.empty() && d.x > bound_volume[max_window.back() + offset].x) max_window.pop_back();
                        while (!min_window.empty() && d.y < bound_volume[min_window.back() + offset].y) min_window.pop_back();
                    }
                    if (!max_window.empty() && max_window.front() <= i - (window_size)) max_window.pop_front();
                    if (!min_window.empty() && min_window.front() <= i - (window_size)) min_window.pop_front();
                    if (i < nx)
                    {
                        max_window.push_back(i);
                        min_window.push_back(i);
                    }
                }
            }

            elapsed = timer.elapsed();
            printf_s("sweeping x took %f us\n", elapsed);
            break;
        case 1:
            timer.record();

#pragma omp parallel for
            for (int k = 0; k < nz; k++)
            for (int i = 0; i < nx; i++)
            {
                CircularBuffer<int, buffer_size> max_window;
                CircularBuffer<int, buffer_size> min_window;
                size_t offset = i + k * nxy;
                for (int j = 0; j < ny + diffusion_iters + 1; j++)
                {
                    if (j > diffusion_iters)
                    {
                        Data dmax = bound_volume[max_window.front() * nx + offset].x;
                        Data dmin = bound_volume[min_window.front() * nx + offset].y;
                        bound_volume_2[(j - diffusion_iters - 1) * nx + offset] = make<Data, Data2>(dmax, dmin);
                    }
                    if (j < ny)
                    {
                        Data2 d = bound_volume[j * nx + offset];
                        while (!max_window.empty() && d.x > bound_volume[max_window.back() * nx + offset].x) max_window.pop_back();
                        while (!min_window.empty() && d.y < bound_volume[min_window.back() * nx + offset].y) min_window.pop_back();
                    }
                    if (!max_window.empty() && max_window.front() <= j - (window_size)) max_window.pop_front();
                    if (!min_window.empty() && min_window.front() <= j - (window_size)) min_window.pop_front();
                    if (j < ny)
                    {
                        max_window.push_back(j);
                        min_window.push_back(j);
                    }
                }
            }

            elapsed = timer.elapsed();
            printf_s("sweeping y took %f us\n", elapsed);
            break;
        case 2:
            timer.record();

#pragma omp parallel for
            for (int j = 0; j < ny; j++)
            for (int i = 0; i < nx; i++)
            {
                CircularBuffer<int, buffer_size> max_window;
                CircularBuffer<int, buffer_size> min_window;
                size_t offset = i + j * nx;
                for (int k = 0; k < nz + diffusion_iters + 1; k++)
                {
                    if (k > diffusion_iters)
                    {
                        Data dmax = bound_volume[max_window.front() * nxy + offset].x;
                        Data dmin = bound_volume[min_window.front() * nxy + offset].y;
                        bound_volume_2[(k - diffusion_iters - 1) * nxy + offset] = make<Data, Data2>(dmax, dmin);
                    }
                    if (k < nz)
                    {
                        Data2 d = bound_volume[k * nxy + offset];
                        while (!max_window.empty() && d.x > bound_volume[max_window.back() * nxy + offset].x) max_window.pop_back();
                        while (!min_window.empty() && d.y < bound_volume[min_window.back() * nxy + offset].y) min_window.pop_back();
                    }
                    if (!max_window.empty() && max_window.front() <= k - (window_size)) max_window.pop_front();
                    if (!min_window.empty() && min_window.front() <= k - (window_size)) min_window.pop_front();
                    if (k < nz)
                    {
                        max_window.push_back(k);
                        min_window.push_back(k);
                    }
                }
            }

            elapsed = timer.elapsed();
            printf_s("sweeping z took %f us\n", elapsed);
            break;
        }

        std::swap(bound_volume, bound_volume_2);
    }
    
    delete[] bound_volume_2;

    return bound_volume;
}

uchar2* compute_volume_value_bound(const unsigned char* volume,
                                   const cudaExtent&    extent,
                                   float                search_radius)
{
    return compute_volume_value_bound_<unsigned char, uchar2>(
        volume, extent, search_radius);
}
float2* compute_volume_value_bound(const float*      volume,
                                   const cudaExtent& extent,
                                   float             search_radius)
{
    return compute_volume_value_bound_<float, float2>(
        volume, extent, search_radius);
}
#endif

void Mat(Param& P, float X, float Y, float Z, float R, float G, float B)
{
    P.sigma_t.x = ((X) + (R));
    P.sigma_t.y = ((Y) + (G));
    P.sigma_t.z = ((Z) + (B));
    P.albedo.x  = (X) / P.sigma_t.x;
    P.albedo.y  = (Y) / P.sigma_t.y;
    P.albedo.z  = (Z) / P.sigma_t.z;
    float f     = std::max(std::max(P.sigma_t.x, P.sigma_t.y), P.sigma_t.z);
    P.sigma_t.x /= f;
    P.sigma_t.y /= f;
    P.sigma_t.z /= f;
    params.push_back(P);
}

int main(int argc, char** argv)
{
    P.brightness = 1.0f;
    P.width      = 960;
    P.height     = 512;
    P.albedo     = make_float3(1.0f, 1.0f, 1.0f);
    P.g          = 0.877f;
    P.density    = 800;
    P.sigma_t    = make_float3(1.0f, 1.0f, 1.0f);

    if (1)
    {
        Mat(P, 2.29f, 2.39f, 1.97f, 0.0030f, 0.0034f, 0.046f);
        Mat(P, 0.15f, 0.21f, 0.38f, 0.015f, 0.077f, 0.19f);
        Mat(P, 0.19f, 0.25f, 0.32f, 0.018f, 0.088f, 0.20f);
        Mat(P, 7.38f, 5.47f, 3.15f, 0.0002f, 0.0028f, 0.0163f);
        Mat(P, 0.18f, 0.07f, 0.03f, 0.061f, 0.97f, 1.45f);
        Mat(P, 2.19f, 2.62f, 3.00f, 0.0021f, 0.0041f, 0.0071f);
        Mat(P, 0.68f, 0.70f, 0.55f, 0.0024f, 0.0090f, 0.12f);
        Mat(P, 0.70f, 1.22f, 1.90f, 0.0014f, 0.0025f, 0.0142f);
        Mat(P, 0.74f, 0.88f, 1.01f, 0.032f, 0.17f, 0.48f);
        Mat(P, 1.09f, 1.59f, 1.79f, 0.013f, 0.070f, 0.145f);
        Mat(P, 11.6f, 20.4f, 14.9f, 0.0f, 0.0f, 0.0f);
        Mat(P, 2.55f, 3.21f, 3.77f, 0.0011f, 0.0024f, 0.014f);
        Mat(P, 1.0f, 1.0f, 1.0f, 0.0f, 0.0f, 0.0f);
    }

    //start logs
    printf("%s Starting...\n\n", sSDKsample);

    // First initialize OpenGL context, so we can properly set the GL for CUDA.
    // This is necessary in order to achieve optimal performance with OpenGL/CUDA interop.
    // initialize GLUT callback functions
    glutInit(&argc, argv);
    glutInitDisplayMode(GLUT_RGB | GLUT_DOUBLE);
    glutInitWindowSize(P.width, P.height);
    glutInitWindowPosition(100, 100);

    glutCreateWindow("CUDA volume rendering");
    glewInit();
    if (!glewIsSupported("GL_VERSION_2_0 GL_ARB_pixel_buffer_object"))
    {
        printf("Required OpenGL extensions missing.");
        exit(EXIT_SUCCESS);
    }

    int        width, height, depth;
    #if USE_OPENVDB
    char* vdb_name = "../wdas_cloud_quarter.vdb";
    void* h_volume = loadVdbFile(vdb_name, width, height, depth);
    cudaExtent volumeSize = make_cudaExtent(width, height, depth);
    TextureVolume::init_cuda(h_volume, volumeSize);
    free(h_volume);
    TextureVolume::set_texture_filter_mode(linearFiltering);
    #else
    cudaExtent volumeSize = make_cudaExtent(32, 32, 32);
    TextureVolume::init_cuda(nullptr, volumeSize);
    #endif

    printf("Press '+' and '-' to change density (0.01 increments)\n"
           "      ']' and '[' to change brightness\n");

    // calculate new grid size
    gridSize = dim3(divideUp(P.width, blockSize.x), divideUp(P.height, blockSize.y));

    // This is the normal rendering path for VolumeRender
    glutDisplayFunc(display);
    glutKeyboardFunc(keyboard);
    glutMouseFunc(mouse);
    glutMotionFunc(motion);
    glutReshapeFunc(reshape);
    glutIdleFunc(idle);

    initPixelBuffer();
    init_rng(gridSize, blockSize, P.width, P.height);

    glutCloseFunc(cleanup);
    glutMainLoop();
}
