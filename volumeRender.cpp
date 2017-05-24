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

#include "param.h"

typedef unsigned int uint;
typedef unsigned char uchar;

const char *sSDKsample = "CUDA 3D Volume Render";

cudaExtent volumeSize = make_cudaExtent(32, 32, 32);
typedef unsigned char VolumeType;

dim3 blockSize(16, 16);
dim3 gridSize;

bool linearFiltering = true;

GLuint pbo = 0;     // OpenGL pixel buffer object
GLuint tex = 0;     // OpenGL texture object

float3 viewRotation = make_float3(20, -20, 0);
float3 viewTranslation = make_float3(0.0, 0.0, -4.0f);

extern "C" void set_texture_filter_mode(bool bLinearFilter);
extern "C" void init_cuda(void *h_volume, cudaExtent volumeSize);
extern "C" void free_cuda_buffers();
extern "C" void render_kernel(dim3 gridSize, dim3 blockSize, float4 *d_output, const Param& p);
extern "C" void copy_inv_view_matrix(float *invViewMatrix, size_t sizeofMatrix);
extern "C" void init_rng(dim3 gridSize, dim3 blockSize, int width, int height);
extern "C" void free_rng();
extern "C" void scale(float4 *dst, float4 *src, int size, float scale);

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
        checkCudaErrors(cudaMallocManaged((void**)&sum_buffer, width * height * sizeof(float4)));
        // clear image
        checkCudaErrors(cudaMemset(sum_buffer, 0, width * height * sizeof(float4)));
    }
    ~CudaFrameBuffer()
    {
        checkCudaErrors(cudaFree(sum_buffer));
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
//     model = glm::inverse(model);
    model = glm::transpose(model);

    copy_inv_view_matrix(glm::value_ptr(model), sizeof(float4)*3);

    // map PBO to get CUDA device pointer
    float4 *d_output = resource->map();

    // call CUDA kernel, writing results to PBO
    render_kernel(gridSize, blockSize, fb->ptr(), P);
    fb->incrementSPP();
    fb->scaledOutput(d_output);

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
            set_texture_filter_mode(linearFiltering);
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
            P.albedo += 0.01;
            P.albedo = std::max(0.0f, std::min(P.albedo, 1.0f));
            break;

        case 'z':
            P.albedo -= 0.01;
            P.albedo = std::max(0.0f, std::min(P.albedo, 1.0f));
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
    printf("albedo = %.2f, g = %.2f\n", P.albedo, P.g);
    glutPostRedisplay();

    delete fb;
    fb = new CudaFrameBuffer(P.width, P.height);
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
    float dx, dy;
    dx = (float)(x - ox);
    dy = (float)(y - oy);

    if (buttonState == 4)
    {
        // right = zoom
        viewTranslation.z += dy / 100.0f;
    }
    else if (buttonState == 2)
    {
        // middle = translate
        viewTranslation.x += dx / 100.0f;
        viewTranslation.y -= dy / 100.0f;
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

    delete fb;
    fb = new CudaFrameBuffer(P.width, P.height);
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
    glOrtho(0.5 - float(P.width) / float(P.height) / 2,
            0.5 + float(P.width) / float(P.height) / 2, 0.0, 1.0, 0.0, 1.0);
}

void cleanup()
{
    free_cuda_buffers();
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
        return 0;
    }

    void *data = malloc(size);
    size_t read = fread(data, 1, size, fp);
    fclose(fp);

    printf("Read '%s', %zu bytes\n", filename, read);

    return data;
}

int main(int argc, char **argv)
{
    P.density = 100.0f * 2;
    P.brightness = 3.0f;
    P.width = 512;
    P.height = 512;
    P.albedo = 1.0f;
    P.g = 0; //  0.877f;

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

    void *h_volume = new VolumeType[volumeSize.width * volumeSize.height * volumeSize.depth];

    init_cuda(h_volume, volumeSize);
    free(h_volume);


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
