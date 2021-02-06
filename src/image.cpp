#include "image.h"

#include <algorithm>
#include <cmath>
#include <fstream>
#include <iostream>
#include <limits>

static float luminance(const Pixel& rgb)
{
    // return 0.212671f * rgb.x + 0.715160f * rgb.y + 0.072169f * rgb.z;
    return 0.299f * rgb.x + 0.587f * rgb.y + 0.114f * rgb.z;
}

static float clampf(float x, float a, float b)
{
    return x < a ? a : x > b ? b : x;
}

void Image::dump_ppm(const char* filename)
{
    std::ofstream ofs(filename, std::ios::out | std::ios::binary);
    if (!ofs.is_open())
    {
        std::cout << "cannot write to " << filename << std::endl;
        return;
    }

    ofs << "P6\n" << m_width << " " << m_height << "\n255\n";
    for (int j = m_height - 1; j >= 0; --j)
    {
        for (int i = 0; i < m_width; ++i)
        {
            Pixel p = m_buffer[i + j * m_width];
            ofs << (unsigned char)(std::min(1.0f, p.x) * 255)
                << (unsigned char)(std::min(1.0f, p.y) * 255)
                << (unsigned char)(std::min(1.0f, p.z) * 255);
        }
    }
    ofs.close();
}

struct HDRPixel
{
    uint8_t r, g, b, e;
    HDRPixel() = default;
    HDRPixel(uint8_t r, uint8_t g, uint8_t b, uint8_t e)
        : r(r), g(g), b(b), e(e)
    {
    }
    uint8_t operator[](int i) const { return (&r)[i]; }
};

HDRPixel toRGBE(const Pixel& c)
{
    float d = std::max(std::max(c.x, c.y), c.z);
    if (d <= 1e-32)
    {
        return HDRPixel(0, 0, 0, 0);
    }
    int   e;
    float m = frexp(d, &e);  // d = m * 2^e
    d       = m * 256.0f / d;
    return HDRPixel(static_cast<uint8_t>(c.x * d),
                    static_cast<uint8_t>(c.y * d),
                    static_cast<uint8_t>(c.z * d),
                    static_cast<uint8_t>(e + 128));
}

void Image::dump_hdr(const char* filename)
{
    std::ofstream ofs(filename, std::ios::out | std::ios::binary);
    if (!ofs.is_open())
    {
        std::cout << "cannot write to " << filename << std::endl;
    }

    ofs << "#?RADIANCE" << std::endl;
    ofs << "# Made with custom writer" << std::endl;
    ofs << "FORMAT=32-bit_rle_rgbe" << std::endl;
    ofs << "EXPOSURE=1.0" << std::endl;
    ofs << std::endl;

    ofs << "-Y " << m_height << " +X " << m_width << std::endl;

    for (int j = m_height - 1; j >= 0; --j)
    {
        std::vector<HDRPixel> line(m_width);
        for (int i = 0; i < m_width; i++)
        {
            HDRPixel p = toRGBE(m_buffer[i + j * m_width]);
            line[i]    = p;
        }
        ofs << uint8_t(2) << uint8_t(2);
        ofs << uint8_t((m_width >> 8) & 0xFF) << uint8_t(m_width & 0xFF);
        for (int k = 0; k < 4; k++)
        {
            for (int cursor = 0; cursor < m_width;)
            {
                const int cursor_move = std::min(127, m_width - cursor);
                ofs << uint8_t(cursor_move);
                for (int i = cursor; i < cursor + cursor_move; i++)
                {
                    ofs << uint8_t(line[i][k]);
                }
                cursor += cursor_move;
            }
        }
    }
    ofs.close();
}

void Image::tonemap_reinhard()
{
    size_t pixel_count = m_buffer.size();
    float* lum         = new float[pixel_count];
    Pixel* fb          = m_buffer.data();

    float lum_eps = 1e-7f;

    for (int n = 0; n < pixel_count; n++)
    {
        lum[n] = luminance(fb[n]);
        if (lum[n] < lum_eps) lum[n] = lum_eps;
    }

    float lum_min = std::numeric_limits<float>::max();
    float lum_max = -std::numeric_limits<float>::max();
    for (int n = 0; n < pixel_count; n++)
    {
        lum_min = lum_min < lum[n] ? lum_min : lum[n];
        lum_max = lum_max > lum[n] ? lum_max : lum[n];
    }

    float l_logmean = 0;
    float l_mean    = 0;
    float r_mean    = 0;
    float g_mean    = 0;
    float b_mean    = 0;
    for (int n = 0; n < pixel_count; n++)
    {
        l_logmean += logf(lum[n]);
        l_mean += lum[n];
        r_mean += fb[n].x;
        g_mean += fb[n].y;
        b_mean += fb[n].z;
    }
    l_logmean /= pixel_count;
    l_mean /= pixel_count;
    r_mean /= pixel_count;
    g_mean /= pixel_count;
    b_mean /= pixel_count;

    float lmin = logf(lum_min);
    float lmax = logf(lum_max);
    float k    = (lmax - l_logmean) / (lmax - lmin);
    float m0   = 0.3f + 0.7f * powf(k, 1.4f);  // contrast
    m0         = 0.77f;                        // hdrsee default

    float m = m0;  // Contrast [0.3f, 1.0f]
    //     printf("contrast: %f\n", m);

    float c = 0.5;  // Chromatic Adaptation  [0.0f, 1.0f]
    float a = 0;    // Light Adaptation  [0.0f, 1.0f]
    float f = 0;  // Intensity [-35.0f, 10.0f] (void*)func = intuitiveintensity
                  // specify by log scale

    f = expf(-f);

    for (int n = 0; n < pixel_count; n++)
    {
        float r(fb[n].x), g(fb[n].y), b(fb[n].z);

        float r_lc = c * r + (1.0f - c) * lum[n];       // local adaptation
        float r_gc = c * r_mean + (1.0f - c) * l_mean;  // global adaptation
        float r_ca = a * r_lc + (1.0f - a) * r_gc;      // pixel adaptation

        float g_lc = c * g + (1.0f - c) * lum[n];       // local adaptation
        float g_gc = c * g_mean + (1.0f - c) * l_mean;  // global adaptation
        float g_ca = a * g_lc + (1.0f - a) * g_gc;      // pixel adaptation

        float b_lc = c * b + (1.0f - c) * lum[n];       // local adaptation
        float b_gc = c * b_mean + (1.0f - c) * l_mean;  // global adaptation
        float b_ca = a * b_lc + (1.0f - a) * b_gc;      // pixel adaptation

        r = r / (r + powf(f * r_ca, m));
        g = g / (g + powf(f * g_ca, m));
        b = b / (b + powf(f * b_ca, m));

        fb[n].x = r;
        fb[n].y = g;
        fb[n].z = b;
    }

    delete[] lum;
}

void Image::tonemap_gamma(float gamma)
{
    float inv_gamma = 1.0f / gamma;
    for (int n = 0; n < m_buffer.size(); n++)
    {
        Pixel p     = m_buffer[n];
        p.x         = powf(clampf(p.x, 0.0f, 1.0f), inv_gamma);
        p.y         = powf(clampf(p.y, 0.0f, 1.0f), inv_gamma);
        p.z         = powf(clampf(p.z, 0.0f, 1.0f), inv_gamma);
        m_buffer[n] = p;
    }
}

void Image::accumulate_pixel(int i, int j, const Pixel& c)
{
    if (i < 0 || i >= m_width || j < 0 || j >= m_height)
    {
        return;
    }

    Pixel p = m_buffer[i + j * m_width];
    p.x += c.x;
    p.y += c.y;
    p.z += c.z;
    m_buffer[i + j * m_width] = p;
}

void Image::accumulate_buffer(const Image& f)
{
    for (int n = 0; n < m_buffer.size(); n++)
    {
        Pixel p  = m_buffer[n];
        Pixel p2 = f.m_buffer[n];
        p.x += p2.x;
        p.y += p2.y;
        p.z += p2.z;
        m_buffer[n] = p;
    }
}

void Image::resize(int w, int h)
{
    m_width  = w;
    m_height = h;
    m_buffer.resize(m_width * m_height, Pixel(0.0f));
}

void Image::flip_updown()
{
    for (int r = 0; r < m_height / 2; r++)
    {
        for (int c = 0; c < m_width; c++)
        {
            std::swap(m_buffer[r * m_width + c],
                      m_buffer[(m_height - 1 - r) * m_width + c]);
        }
    }
}

Image::Image(int w, int h) { this->resize(w, h); }

Image::Image() : Image(0, 0) {}

Image::~Image() {}

Pixel Image::pixel(int i, int j) const { return m_buffer[i + j * m_width]; }

void Image::scale(float s)
{
    for (auto& v : m_buffer)
    {
        v *= s;
    }
}

int Image::height() const { return m_height; }

const float* Image::buffer() const
{
    return reinterpret_cast<const float*>(m_buffer.data());
}

float* Image::buffer() { return reinterpret_cast<float*>(m_buffer.data()); }

int Image::width() const { return m_width; }
