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

bool linearFiltering = true;

GLuint pbo = 0;     // OpenGL pixel buffer object
GLuint tex = 0;     // OpenGL texture object

//float3 viewRotation = make_float3(20, -20, 0);
//float3 viewTranslation = make_float3(0.0, 0.0, -4.0f);
float3 viewRotation    = make_float3(-12, -90, 0);
float3 viewTranslation = make_float3(0.03, -0.05, -4.0);

extern "C" void render_kernel(dim3 gridSize, dim3 blockSize, float4 *d_output, const Param& p);
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
    // build view matrix
    glm::mat4 model = glm::mat4(1.0f);
    model = glm::rotate(model, glm::radians(-viewRotation.y), glm::vec3(0.0, 1.0, 0.0));
    model = glm::rotate(model, glm::radians(-viewRotation.x), glm::vec3(1.0, 0.0, 0.0));
    model = glm::translate(model, glm::vec3(-viewTranslation.x, -viewTranslation.y, -viewTranslation.z));
    model = glm::transpose(model); // build row major

    copy_inv_view_matrix(glm::value_ptr(model), sizeof(float4)*3);

    // map PBO to get CUDA device pointer
    float4 *d_output = resource->map();

    // call CUDA kernel, writing results to PBO
    Timer timer;
    render_kernel(gridSize, blockSize, fb->ptr(), P);
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
#endif

#define Mat(sigma_t, albedo, X, Y, Z, R, G, B) \
do {                                           \
    (sigma_t).x = X;                           \
    (sigma_t).y = Y;                           \
    (sigma_t).z = Z;                           \
    (albedo).x = (X) / ((X) + (R));            \
    (albedo).y = (Y) / ((Y) + (G));            \
    (albedo).z = (Z) / ((Z) + (B));            \
} while (0)

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
        Mat(P.sigma_t, P.albedo, 2.29f, 2.39f, 1.97f, 0.0030f, 0.0034f, 0.046f);
        Mat(P.sigma_t, P.albedo, 0.15f, 0.21f, 0.38f, 0.015f, 0.077f, 0.19f);
        Mat(P.sigma_t, P.albedo, 0.19f, 0.25f, 0.32f, 0.018f, 0.088f, 0.20f);
        Mat(P.sigma_t, P.albedo, 7.38f, 5.47f, 3.15f, 0.0002f, 0.0028f, 0.0163f);
        Mat(P.sigma_t, P.albedo, 0.18f, 0.07f, 0.03f, 0.061f, 0.97f, 1.45f);
        Mat(P.sigma_t, P.albedo, 2.19f, 2.62f, 3.00f, 0.0021f, 0.0041f, 0.0071f);
        Mat(P.sigma_t, P.albedo, 0.68f, 0.70f, 0.55f, 0.0024f, 0.0090f, 0.12f);
        Mat(P.sigma_t, P.albedo, 0.70f, 1.22f, 1.90f, 0.0014f, 0.0025f, 0.0142f);
        Mat(P.sigma_t, P.albedo, 0.74f, 0.88f, 1.01f, 0.032f, 0.17f, 0.48f);
        Mat(P.sigma_t, P.albedo, 1.09f, 1.59f, 1.79f, 0.013f, 0.070f, 0.145f);
        Mat(P.sigma_t, P.albedo, 11.6f, 20.4f, 14.9f, 0.0f, 0.0f, 0.0f);
        Mat(P.sigma_t, P.albedo, 2.55f, 3.21f, 3.77f, 0.0011f, 0.0024f, 0.014f);
        Mat(P.sigma_t, P.albedo, 1.0f, 1.0f, 1.0f, 0.0f, 0.0f, 0.0f);

        float f = std::max(std::max(P.sigma_t.x, P.sigma_t.y), P.sigma_t.z);
        P.sigma_t.x /= f;
        P.sigma_t.y /= f;
        P.sigma_t.z /= f;
    }

    //start logs
    printf("%s Starting...\n\n", sSDKsample);

    // First initialize OpenGL context, so we can properly set the GL for CUDA.
    // This is necessary in order to achieve optimal performance with OpenGL/CUDA interop.
    // initialize GLUT callback functions
    glutInit(&argc, argv);
    glutInitDisplayMode(GLUT_RGB | GLUT_DOUBLE);
    glutInitWindowSize(P.width, P.height);

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
    TextureVolume::set_texture_filter_mode(false);
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
