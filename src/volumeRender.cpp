#include <algorithm>
#include <glm/glm.hpp>
#include <glm/gtc/matrix_transform.hpp>
#include <glm/gtc/type_ptr.hpp>
#include <iostream>
using std::cout;
using std::endl;

#include <GL/glew.h>
//
#include <GL/freeglut.h>
#include <cuda_gl_interop.h>
#include <cuda_runtime.h>
#include <vector_functions.h>
#include <vector_types.h>

#include <chrono>
#include <deque>
#include <random>
#include <thread>

#include "cuda_helpers.h"
using namespace std::chrono_literals;

#include "hdr/HDRloader.h"
#include "image.h"
#include "param.h"

#if USE_OPENVDB
#include <load_vdb.h>
#endif

#include "denoiser.h"
bool g_denoise = false;

#include "sunsky/sunsky.h"
bool g_set_sunsky = true;

bool linearFiltering = true;

std::vector<Param> params;

// sigma_s, sigma_a
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

typedef unsigned int  uint;
typedef unsigned char uchar;

static bool exists(const char* filename)
{
    FILE* f = fopen(filename, "r");
    if (!f)
        return false;
    else
    {
        fclose(f);
        return true;
    }
}

static std::default_random_engine            rng;
static std::uniform_real_distribution<float> dist;
static float                                 randf() { return dist(rng); }

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

// float3 viewRotation = make_float3(20, -20, 0);
// float3 viewTranslation = make_float3(0.0, 0.0, -4.0f);
float3 viewRotation    = make_float3(-12, -90, 0);
float3 viewTranslation = make_float3(0.03, -0.05, -4.0);

glm::vec3 cam_position   = glm::vec3(3.922986, -0.782739, 0.030000);
glm::vec3 cam_forward    = glm::vec3(-0.978148, 0.207912, 0.000000);
glm::vec3 cam_right      = glm::vec3(-0.000000, 0.000000, -1.000000);
glm::vec3 cam_up         = glm::vec3(0.207912, 0.978148, -0.000000);
float     cam_focus_dist = 4.0f;
glm::mat4 cam_view;
glm::mat4 cam_inv_view;
bool      cam_update = true;

extern "C" void render_kernel(
    dim3 gridSize, dim3 blockSize, float4* d_output, int spp, const Param& p);
extern "C" void copy_inv_view_matrix(float* invViewMatrix, size_t sizeofMatrix);
extern "C" void copy_inv_model_matrix(float* invModelMatrix, size_t sizeofMatrix);
extern "C" void init_rng(dim3 gridSize, dim3 blockSize, int width, int height);
extern "C" void free_rng();
extern "C" void scale(float4* dst, float4* src, int size, float scale);
extern "C" void gamma_correct(float4* dst, float4* src, int size, float scale, float gamma);
extern "C" void init_envmap(const float4* data, int width, int height);
extern "C" void free_envmap();
extern "C" void set_sun(float* sun_dir, float* sun_power);
extern "C" void precompute_opacity(const float* light_dir);

class SdkTimer
{
    StopWatchInterface* timer = 0;

public:
    SdkTimer() { sdkCreateTimer(&timer); }
    ~SdkTimer() { sdkDeleteTimer(&timer); }

    void start() { sdkStartTimer(&timer); }

    void stop() { sdkStopTimer(&timer); }

    float fps() { return 1.0f / (sdkGetAverageTimerValue(&timer) / 1000.0f); }

    void reset() { sdkResetTimer(&timer); }
};

template <typename T>
class PboResource
{
    struct cudaGraphicsResource* cuda_pbo_resource = nullptr;
    bool                         mapped            = false;

public:
    PboResource(unsigned int pbo)
    {
        checkCudaErrors(cudaGraphicsGLRegisterBuffer(
            &cuda_pbo_resource, pbo, cudaGraphicsRegisterFlagsWriteDiscard));
    }
    ~PboResource()
    {
        if (mapped)
        {
            unmap();
        }
        checkCudaErrors(cudaGraphicsUnregisterResource(cuda_pbo_resource));
    }
    T* map()
    {
        T* d_output;
        checkCudaErrors(cudaGraphicsMapResources(1, &cuda_pbo_resource, 0));
        size_t num_bytes;
        checkCudaErrors(
            cudaGraphicsResourceGetMappedPointer((void**)&d_output, &num_bytes, cuda_pbo_resource));
        mapped = true;
        return d_output;
    }

    void unmap()
    {
        checkCudaErrors(cudaGraphicsUnmapResources(1, &cuda_pbo_resource, 0));
        mapped = false;
    }
};
template <typename T>
class Tex2DResource
{
    struct cudaGraphicsResource* cuda_tex_resource = nullptr;
    bool                         mapped            = false;

public:
    Tex2DResource(unsigned int tex)
    {
        checkCudaErrors(cudaGraphicsGLRegisterImage(
            &cuda_tex_resource, tex, GL_TEXTURE_2D, cudaGraphicsRegisterFlagsWriteDiscard));
    }
    ~Tex2DResource()
    {
        if (mapped)
        {
            unmap();
        }
        checkCudaErrors(cudaGraphicsUnregisterResource(cuda_tex_resource));
    }
    cudaArray_t map()
    {
        cudaArray_t d_output;
        checkCudaErrors(cudaGraphicsMapResources(1, &cuda_tex_resource, 0));
        checkCudaErrors(cudaGraphicsSubResourceGetMappedArray(&d_output, cuda_tex_resource, 0, 0));
        mapped = true;
        return d_output;
    }

    void unmap()
    {
        checkCudaErrors(cudaGraphicsUnmapResources(1, &cuda_tex_resource, 0));
        mapped = false;
    }
};

class EnvMapLoader
{
public:
    EnvMapLoader(const char* HDRfile, float k_brightness)
    {
        HDRImage HDRresult;

        if (HDRLoader::load(HDRfile, HDRresult))
            printf("HDR environment map loaded. Width: %d Height: %d\n",
                   HDRresult.width,
                   HDRresult.height);
        else
            printf("HDR environment map not found\n");

        int                HDRwidth  = HDRresult.width;
        int                HDRheight = HDRresult.height;
        stl_vector<float4> cpuHDRenv(HDRwidth * HDRheight);

        for (int i = 0; i < HDRwidth; i++)
        {
            for (int j = 0; j < HDRheight; j++)
            {
                int idx         = 3 * (HDRwidth * j + i);
                int idx2        = HDRwidth * (j) + i;
                cpuHDRenv[idx2] = make_float4(k_brightness * HDRresult.colors[idx],
                                              k_brightness * HDRresult.colors[idx + 1],
                                              k_brightness * HDRresult.colors[idx + 2],
                                              0.0f);
            }
        }

        init_envmap(cpuHDRenv.data(), HDRwidth, HDRheight);
    }

    EnvMapLoader(const float4* data, int width, int height) { init_envmap(data, width, height); }

    ~EnvMapLoader() {}
};

std::unique_ptr<EnvMapLoader> envmap;

float sunsky_x      = 0;
float sunsky_y      = 0;
float sunsky_dirty  = false;
float opacity_dirty = false;
// SkyModel<PreethamSunSky> s;
SkyModel<Tungsten::Skydome> s;

void setup_sunsky(float x, float y)
{
    sunsky_x      = x;
    sunsky_y      = y;
    sunsky_dirty  = true;
    opacity_dirty = true;
}

void update_sunsky(int spp, bool baked = false)
{
    if (sunsky_dirty)
    {
        float x = sunsky_x;
        float y = sunsky_y;
        y *= 0.5f;
        y = clamp(y, 0.0f, 0.49999f);

        int                 hdrwidth = 512 * 2, hdrheight = 256 * 2;
        std::vector<float4> hdrimage(hdrwidth * hdrheight);

        s.setSunPhi(x * M_PI * 2);
        s.setSunTheta(y * M_PI);

        bool            bake_sun     = false;  // if include sun in the envmap
        constexpr float sunsky_scale = 0.02;

        auto sun_dir   = s.getSunDir();
        auto sun_power = s.sunColor() * sunsky_scale;

        if (baked)
        {
#pragma omp parallel for
            for (int i = 0; i < hdrwidth; i++)
            {
                for (int j = 0; j < hdrheight; j++)
                {
                    if (j < hdrheight / 2)
                    {
                        float phi   = float(i) / hdrwidth * 2 * M_PI;
                        float theta = (float(j) / hdrheight) * M_PI;
                        // must match Envmap::uv_to_dir
                        float3 d = make_float3(
                            sinf(theta) * sinf(phi), cosf(theta), sinf(theta) * -cosf(phi));
                        float3 c = s.skyColor(d, bake_sun);

                        hdrimage[i + j * hdrwidth] = make_float4(c, 1.0f) * sunsky_scale;
                    }
                    else
                    {
                        float3 ground_albedo      = make_float3(0.01f);
                        float3 reflected_radiance = ground_albedo * sun_dir.y * sun_power *
                                                    (M_PI * (0.45 / 94.0f * 0.45 / 94.0f));
                        hdrimage[i + j * hdrwidth] = make_float4(reflected_radiance, 1.0f);
                    }
                }
            }

            envmap = std::make_unique<EnvMapLoader>(hdrimage.data(), hdrwidth, hdrheight);
        }

        if (bake_sun) sun_power = make_float3(0.0f);
        printf("sun power = %f, %f, %f\n", sun_power.x, sun_power.y, sun_power.z);
        set_sun(&sun_dir.x, &sun_power.x);

        sunsky_dirty = false;
    }

    // precomputed opacity use condition
    if (spp > 10)
    {
        if (opacity_dirty)
        {
            auto sun_dir = s.getSunDir();
            precompute_opacity(&sun_dir.x);
            opacity_dirty = false;
        }
    }
}

namespace TextureVolume
{
extern "C" void init_cuda(void*         h_volume,
                          cudaExtent    volumeSize,
                          bool          quantized,
                          const float3* boxmin,
                          const float3* boxmax);
extern "C" void set_texture_filter_mode(bool bLinearFilter);
extern "C" void free_cuda_buffers();
}  // namespace TextureVolume

class CudaFrameBuffer
{
protected:
    int     _width, _height;
    int     spp = 0;
    float4* sum_buffer;

public:
    CudaFrameBuffer(int w, int h)
    {
        _width  = w;
        _height = h;
        checkCudaErrors(cudaMalloc((void**)&sum_buffer, _width * _height * sizeof(float4)));
        this->reset();
    }
    ~CudaFrameBuffer() { checkCudaErrors(cudaFree(sum_buffer)); }
    void reset()
    {
        spp = 0;
        checkCudaErrors(cudaMemset(sum_buffer, 0, _width * _height * sizeof(float4)));
    }
    float4* ptr() { return sum_buffer; }
    void    incrementSPP() { spp++; }
    void    scaledOutput(float4* dst) { scale(dst, sum_buffer, _width * _height, 1.0f / spp); }
    void    gammaCorrectedOutput(float4* dst, float gamma)
    {
        gamma_correct(dst, sum_buffer, _width * _height, 1.0f / spp, gamma);
    }
    int samplesPerPixel() const { return spp; }
    int width() const { return _width; }
    int height() const { return _height; }
};

class FrameBuffer : public CudaFrameBuffer
{
    void bind()
    {
        if (use_pbo)
        {
            mapped_buffer = resource->map();
        }
        else
        {
            mapped_array = resource_tex->map();
        }
    }
    void unbind()
    {
        if (use_pbo)
        {
            resource->unmap();
        }
        else
        {
            resource_tex->unmap();
        }
    }

public:
    GLuint                                 pbo = 0;
    GLuint                                 tex = 0;
    std::unique_ptr<PboResource<float4>>   resource;
    std::unique_ptr<Tex2DResource<float4>> resource_tex;
    float4*                                mapped_buffer = nullptr;
    cudaArray_t                            mapped_array  = nullptr;
    std::unique_ptr<CudaDenoiser>          dn;

    bool use_pbo = true;

    FrameBuffer(int width, int height) : CudaFrameBuffer(width, height)
    {
        // create texture for display
        glGenTextures(1, &tex);
        glActiveTexture(GL_TEXTURE0);
        glBindTexture(GL_TEXTURE_2D, tex);
        glTexImage2D(GL_TEXTURE_2D, 0, GL_RGBA32F, width, height, 0, GL_RGBA, GL_FLOAT, NULL);
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_NEAREST);
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_NEAREST);
        glBindTexture(GL_TEXTURE_2D, 0);

        if (use_pbo)
        {
            // create pixel buffer object for display
            glGenBuffers(1, &pbo);
            glBindBuffer(GL_PIXEL_UNPACK_BUFFER, pbo);
            glBufferData(
                GL_PIXEL_UNPACK_BUFFER, width * height * sizeof(GLfloat) * 4, 0, GL_STREAM_DRAW);
            glBindBuffer(GL_PIXEL_UNPACK_BUFFER, 0);

            resource = std::make_unique<PboResource<float4>>(pbo);
        }
        else
        {
            resource_tex = std::make_unique<Tex2DResource<float4>>(tex);
            checkCudaErrors(cudaMalloc((void**)&mapped_buffer, sizeof(float4) * width * height));
        }

        dn = std::make_unique<CudaDenoiser>(width, height);
    }

    ~FrameBuffer()
    {
        mapped_buffer = nullptr;
        mapped_array  = nullptr;

        if (use_pbo)
        {
            resource.reset();
            glDeleteBuffers(1, &pbo);
        }
        else
        {
            resource_tex.reset();
            checkCudaErrors(cudaFree(mapped_buffer));
        }

        glDeleteTextures(1, &tex);
    }

    void finalize_gamma(float gamma)
    {
        this->bind();

        this->gammaCorrectedOutput(mapped_buffer, 2.2f);
        if (!use_pbo)
        {
            checkCudaErrors(cudaMemcpy2DToArrayAsync(mapped_array,
                                                     0,
                                                     0,
                                                     mapped_buffer,
                                                     sizeof(float4) * _width,
                                                     sizeof(float4) * _width,
                                                     _height,
                                                     cudaMemcpyDeviceToDevice));
        }

        this->unbind();
    }

    void finalize_denoised()
    {
        this->bind();

        this->scaledOutput(mapped_buffer);
        dn->denoise(this->samplesPerPixel(), mapped_buffer);
        gamma_correct(mapped_buffer, mapped_buffer, _width * _height, 1.0f, 2.2f);

        this->unbind();
    }

    void extract(void* data, bool hdr)
    {
        this->bind();

        if (hdr) this->scaledOutput(mapped_buffer);
        checkCudaErrors(cudaMemcpyAsync(
            data, mapped_buffer, sizeof(float4) * _width * _height, cudaMemcpyDeviceToHost));

        this->unbind();
    }

    void draw()
    {
        glBindTexture(GL_TEXTURE_2D, tex);

        if (use_pbo)
        {
            // draw image from PBO
            glPixelStorei(GL_UNPACK_ALIGNMENT, 1);

            // draw using texture
            // copy from pbo to texture
            glBindBuffer(GL_PIXEL_UNPACK_BUFFER, pbo);
            glTexSubImage2D(GL_TEXTURE_2D, 0, 0, 0, _width, _height, GL_RGBA, GL_FLOAT, 0);
            glBindBuffer(GL_PIXEL_UNPACK_BUFFER, 0);
        }

        // display results
        glClear(GL_COLOR_BUFFER_BIT);
        glDisable(GL_DEPTH_TEST);

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
    }
};

std::unique_ptr<FrameBuffer> fb;

SdkTimer timer;
Param    P;

int          fpsCount   = 0;  // FPS count for averaging
int          fpsLimit   = 1;  // FPS limit for sampling
unsigned int frameCount = 0;

void computeFPS()
{
    frameCount++;
    fpsCount++;

    if (fpsCount == fpsLimit)
    {
        char  fps[256];
        float ifps = timer.fps();
        sprintf(fps, "Volume Render: %3.1f fps @ %d spp", ifps, fb->samplesPerPixel());

        glutSetWindowTitle(fps);
        fpsCount = 0;

        fpsLimit = (int)std::max(1.0f, ifps);
        timer.reset();
    }
}

void capture(bool hdr = false)
{
    static int  i   = 0;
    std::string ext = hdr ? ".hdr" : ".ppm";
    std::string filename;
    do
    {
        filename = "output" + std::to_string(i) + ext;
        ++i;
    } while (exists(filename.c_str()));

    Image image(fb->width(), fb->height());

    fb->extract(image.buffer(), hdr);

    // image.flip_updown();
    if (hdr)
    {
        image.dump_hdr(filename.c_str());
    }
    else
    {
        // image.tonemap_gamma(2.2f);
        image.dump_ppm(filename.c_str());
    }
}

// render image using CUDA
void cuda_volpath()
{
    update_sunsky(fb->samplesPerPixel(), true);

    if (cam_update)
    {
        cam_view = glm::lookAt(cam_position, cam_position + cam_forward * cam_focus_dist, cam_up);
        cam_inv_view    = glm::inverse(cam_view);
        glm::mat4 model = glm::transpose(cam_inv_view);  // convert to row major

        copy_inv_view_matrix(glm::value_ptr(model), sizeof(float4) * 3);
        cam_update = false;
    }

    for (int s = 0; s < 1; s++)
    {
        // call CUDA kernel, writing results to PBO
        Timer timer;
        render_kernel(gridSize, blockSize, fb->ptr(), fb->samplesPerPixel(), P);
        checkCudaErrors(cudaDeviceSynchronize());
        float elapsed = timer.elapsed();
        printf("%f M samples / s, %d x %d, %f\n",
               (float)P.width * (float)P.height / elapsed,
               P.width,
               P.height,
               elapsed);

        fb->incrementSPP();
    }

    if (g_denoise)
    {
        fb->finalize_denoised();
    }
    else
    {
        fb->finalize_gamma(2.2f);
    }

    getLastCudaError("kernel failed");
}

// display results using OpenGL (called by GLUT)
void display()
{
    timer.start();

    cuda_volpath();

    fb->draw();

    glutSwapBuffers();
    glutReportErrors();

    timer.stop();

    computeFPS();
}

void idle() { glutPostRedisplay(); }

void keyboard(unsigned char key, int x, int y)
{
    bool need_reset = false;

    switch (key)
    {
        case 27:
        case 'q':
            glutDestroyWindow(glutGetWindow());
            return;

        case 'f':
            linearFiltering = !linearFiltering;
            TextureVolume::set_texture_filter_mode(linearFiltering);
            need_reset = true;
            break;

        case '+':
        case '=':
            P.density += 1;
            need_reset = true;
            break;

        case '-':
            P.density -= 1;
            P.density  = std::max(P.density, 0.0f);
            need_reset = true;
            break;

        case ']':
            P.brightness += 0.1f;
            need_reset = true;
            break;

        case '[':
            P.brightness -= 0.1f;
            need_reset = true;
            break;

        case 'x':
            P.albedo.x = std::max(0.0f, std::min(P.albedo.x + 0.01f, 1.0f));
            P.albedo.y = std::max(0.0f, std::min(P.albedo.y + 0.01f, 1.0f));
            P.albedo.z = std::max(0.0f, std::min(P.albedo.z + 0.01f, 1.0f));
            need_reset = true;
            break;

        case 'z':
            P.albedo.x = std::max(0.0f, std::min(P.albedo.x - 0.01f, 1.0f));
            P.albedo.y = std::max(0.0f, std::min(P.albedo.y - 0.01f, 1.0f));
            P.albedo.z = std::max(0.0f, std::min(P.albedo.z - 0.01f, 1.0f));
            need_reset = true;
            break;

        case 's':
            P.g += 0.01;
            P.g        = std::max(-1.0f, std::min(P.g, 1.0f));
            need_reset = true;
            break;

        case 'a':
            P.g -= 0.01;
            P.g        = std::max(-1.0f, std::min(P.g, 1.0f));
            need_reset = true;
            break;

        case ' ':
            P          = params[rand() % params.size()];
            need_reset = true;
            break;

        case 'r':
            Mat(P, randf(), randf(), randf(), randf(), randf(), randf());
            need_reset = true;
            break;

        case 'c':
            capture();
            break;

        case 'n':
            g_denoise = !g_denoise;
            break;

        case 'k':
            g_set_sunsky = !g_set_sunsky;
            break;

        default:
            break;
    }

    printf("density = %.2f, brightness = %.2f, ", P.density, P.brightness);
    printf("albedo = %.2f, %.2f, %.2f, g = %.2f\n", P.albedo.x, P.albedo.y, P.albedo.z, P.g);
    glutPostRedisplay();

    if (need_reset) fb->reset();
}

int ox, oy;
int buttonState = 0;

void mouse(int button, int state, int x, int y)
{
    if (state == GLUT_DOWN)
    {
        buttonState |= 1 << button;
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
        // right = translate
        cam_position -= cam_right * dx / 1000.0f * cam_focus_dist;
        cam_position += cam_up * dy / 1000.0f * cam_focus_dist;
    }
    else if (buttonState == 2)
    {
        if (g_set_sunsky)
        {
            setup_sunsky((x + 0.5f) / P.width, (y + 0.5f) / P.height);
        }
        else
        {
            // middle = zoom
            glm::vec3 center = cam_position + cam_forward * cam_focus_dist;
            cam_focus_dist += dy / 100.0f;
            cam_position = center - cam_forward * cam_focus_dist;
        }
    }
    else if (buttonState == 1)
    {
        // left = rotate
        glm::vec3 center = cam_position + cam_forward * cam_focus_dist;

        glm::mat4 R = glm::mat4(1.0f);
        R           = glm::rotate(R, glm::radians(-dx / 5), cam_up);
        R           = glm::rotate(R, glm::radians(-dy / 5), cam_right);
        glm::mat3 r = glm::mat3(R);

        cam_forward = r * cam_forward;
        cam_right   = r * cam_right;
        cam_up      = r * cam_up;

        cam_position = center - cam_forward * cam_focus_dist;
    }
    cam_update = true;

    ox = x;
    oy = y;
    glutPostRedisplay();

    fb->reset();
}

void wheel(int button, int dir, int x, int y)
{
    glm::vec3 center = cam_position + cam_forward * cam_focus_dist;
    cam_focus_dist += -dir / 10.0f;
    cam_position = center - cam_forward * cam_focus_dist;
    cam_update   = true;

    glutPostRedisplay();
    fb->reset();
}

void reshape(int w, int h)
{
    P.width  = w;
    P.height = h;
    fb       = std::make_unique<FrameBuffer>(P.width, P.height);

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
}

void cleanup()
{
    TextureVolume::free_cuda_buffers();
    free_rng();

    fb.reset();

    envmap.reset();

    free_envmap();

    // cudaDeviceReset causes the driver to clean up all state. While
    // not mandatory in normal operation, it is good practice.  It is also
    // needed to ensure correct operation when the application is being
    // profiled. Calling cudaDeviceReset causes all profile data to be
    // flushed before the application exits
    cudaDeviceReset();
}

// Load raw data from disk
void* loadRawFile(char* filename, size_t size)
{
    FILE* fp = fopen(filename, "rb");

    if (!fp)
    {
        fprintf(stderr, "Error opening file '%s'\n", filename);
        return nullptr;
    }

    void*  data = malloc(size);
    size_t read = fread(data, 1, size, fp);
    fclose(fp);

    printf("Read '%s', %zu bytes\n", filename, read);

    return data;
}

void* loadBinaryFile(char* filename, int& width, int& height, int& depth, bool quantized = true)
{
    FILE* fp = fopen(filename, "rb");
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
    size_t read  = fread(dataf, sizeof(float), width * height * depth, fp);
    fclose(fp);
    printf("Read '%s', %zu bytes\n", filename, read);

    // normalize the volume values
    if (quantized)
    {
        VolumeType* data = reinterpret_cast<VolumeType*>(malloc(total));
        for (size_t i = 0; i < total; i++)
        {
            data[i] = VolumeType(std::max(0.0f, std::min(dataf[i], 1.0f)) * 255.0f);
        }
        free(dataf);

        return data;
    }
    else
    {
        return dataf;
    }
}

#if USE_OPENVDB
void* loadVdbFile(char* filename, int& width, int& height, int& depth, bool quantized = true)
{
    if (!exists(filename))
    {
        fprintf(stderr, "Error opening file '%s'\n", filename);
        return nullptr;
    }

    float min_value, max_value;
    auto  dataf = load_vdb(filename, width, height, depth, min_value, max_value);
    max_value   = std::max(max_value, 0.0001f);

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
    if (quantized)
    {
        VolumeType* data = reinterpret_cast<VolumeType*>(malloc(total));
        for (size_t i = 0; i < total; i++)
        {
            // data[i] = VolumeType(std::max(0.0f, std::min(dataf[i], 1.0f)) * 255.0f);
            data[i] = VolumeType(std::max(0.0f, dataf[i]) / max_value * 255.0f);
        }
        free(dataf);

        return data;
    }
    else
    {
        return dataf;
    }
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

    constexpr size_t buffer_size = 256;  // must be larger than diffusion_iters * 2 + 1

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
                        size_t                           offset = j * nx + k * nxy;
                        for (int i = 0; i < nx + diffusion_iters + 1; i++)
                        {
                            if (i > diffusion_iters)
                            {
                                Data dmax = bound_volume[max_window.front() + offset].x;
                                Data dmin = bound_volume[min_window.front() + offset].y;
                                bound_volume_2[(i - diffusion_iters - 1) + offset] =
                                    make<Data, Data2>(dmax, dmin);
                            }
                            if (i < nx)
                            {
                                Data2 d = bound_volume[i + offset];
                                while (!max_window.empty() &&
                                       d.x > bound_volume[max_window.back() + offset].x)
                                    max_window.pop_back();
                                while (!min_window.empty() &&
                                       d.y < bound_volume[min_window.back() + offset].y)
                                    min_window.pop_back();
                            }
                            if (!max_window.empty() && max_window.front() <= i - (window_size))
                                max_window.pop_front();
                            if (!min_window.empty() && min_window.front() <= i - (window_size))
                                min_window.pop_front();
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
                        size_t                           offset = i + k * nxy;
                        for (int j = 0; j < ny + diffusion_iters + 1; j++)
                        {
                            if (j > diffusion_iters)
                            {
                                Data dmax = bound_volume[max_window.front() * nx + offset].x;
                                Data dmin = bound_volume[min_window.front() * nx + offset].y;
                                bound_volume_2[(j - diffusion_iters - 1) * nx + offset] =
                                    make<Data, Data2>(dmax, dmin);
                            }
                            if (j < ny)
                            {
                                Data2 d = bound_volume[j * nx + offset];
                                while (!max_window.empty() &&
                                       d.x > bound_volume[max_window.back() * nx + offset].x)
                                    max_window.pop_back();
                                while (!min_window.empty() &&
                                       d.y < bound_volume[min_window.back() * nx + offset].y)
                                    min_window.pop_back();
                            }
                            if (!max_window.empty() && max_window.front() <= j - (window_size))
                                max_window.pop_front();
                            if (!min_window.empty() && min_window.front() <= j - (window_size))
                                min_window.pop_front();
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
                        size_t                           offset = i + j * nx;
                        for (int k = 0; k < nz + diffusion_iters + 1; k++)
                        {
                            if (k > diffusion_iters)
                            {
                                Data dmax = bound_volume[max_window.front() * nxy + offset].x;
                                Data dmin = bound_volume[min_window.front() * nxy + offset].y;
                                bound_volume_2[(k - diffusion_iters - 1) * nxy + offset] =
                                    make<Data, Data2>(dmax, dmin);
                            }
                            if (k < nz)
                            {
                                Data2 d = bound_volume[k * nxy + offset];
                                while (!max_window.empty() &&
                                       d.x > bound_volume[max_window.back() * nxy + offset].x)
                                    max_window.pop_back();
                                while (!min_window.empty() &&
                                       d.y < bound_volume[min_window.back() * nxy + offset].y)
                                    min_window.pop_back();
                            }
                            if (!max_window.empty() && max_window.front() <= k - (window_size))
                                max_window.pop_front();
                            if (!min_window.empty() && min_window.front() <= k - (window_size))
                                min_window.pop_front();
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
    return compute_volume_value_bound_<unsigned char, uchar2>(volume, extent, search_radius);
}
float2* compute_volume_value_bound(const float*      volume,
                                   const cudaExtent& extent,
                                   float             search_radius)
{
    return compute_volume_value_bound_<float, float2>(volume, extent, search_radius);
}

#endif

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

    // start logs
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

    int width, height, depth;
#if USE_OPENVDB
    bool  quantized = true;  // true for uchar texture
    char* vdb_name  = "../wdas_cloud_eighth.vdb";
    void* h_volume  = loadVdbFile(vdb_name, width, height, depth, quantized);

    float3 scale = make_float3(width, height, depth);
    scale /= width;
    float3 box_min = make_float3(-1.0f) * scale;
    float3 box_max = make_float3(1.0f) * scale;

    cudaExtent volumeSize = make_cudaExtent(width, height, depth);
    TextureVolume::init_cuda(h_volume, volumeSize, quantized, &box_min, &box_max);
    free(h_volume);
    TextureVolume::set_texture_filter_mode(linearFiltering);
#else
    cudaExtent volumeSize = make_cudaExtent(32, 32, 32);
    TextureVolume::init_cuda(nullptr, volumeSize, false, nullptr, nullptr);
#endif

    glm::mat4 model = glm::mat4(1.0f);
    model           = glm::inverse(model);
    model           = glm::transpose(model);
    copy_inv_model_matrix(glm::value_ptr(model), sizeof(float4) * 3);

    // calculate new grid size
    gridSize = dim3(divideUp(P.width, blockSize.x), divideUp(P.height, blockSize.y));

    // This is the normal rendering path for VolumeRender
    glutDisplayFunc(display);
    glutKeyboardFunc(keyboard);
    glutMouseFunc(mouse);
    glutMotionFunc(motion);
    glutReshapeFunc(reshape);
    glutMouseWheelFunc(wheel);
    glutIdleFunc(idle);

    CudaDenoiser::create_context();
    if (0)
    {
        envmap = std::make_unique<EnvMapLoader>("../envmap.hdr", 1.0f);
    }
    else if (0)
    {
        int                 hdrwidth = 16, hdrheight = 8;
        std::vector<float4> hdrimage(hdrwidth * hdrheight);
        for (int j = 0; j < hdrheight; j++)
        {
            for (int i = 0; i < hdrwidth; i++)
            {
                hdrimage[i + j * hdrwidth] =
                    j < 5 ? make_float4(0.03, 0.07, 0.23, 1) : make_float4(0.03, 0.03, 0.03, 1);
            }
        }
        envmap = std::make_unique<EnvMapLoader>(hdrimage.data(), hdrwidth, hdrheight);
    }
    else
    {
        float x = 0.5;
        float y = 0.2;
        setup_sunsky(x, y);
    }

    fb = std::make_unique<FrameBuffer>(P.width, P.height);
    init_rng(gridSize, blockSize, P.width, P.height);

    glutCloseFunc(cleanup);

    glutMainLoop();

    CudaDenoiser::free_context();

    return 0;
}
