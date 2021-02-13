#pragma once

#include <memory>

#include "vecmath.h"

namespace Tungsten
{
using Vec2f = float2;
using Vec3f = float3;

class Skydome
{
    float _temperature;
    float _gammaScale;
    float _turbidity;
    float _intensity;
    bool  _doSample;

public:
    Skydome();

    void prepareForRender();

    float turbidity() const { return _turbidity; }

    float intensity() const { return _intensity; }

    Vec3f sunDirection() const
    {
        float sin_sun_theta = sinf(_theta);
        return make_float3(sinf(_phi) * sin_sun_theta, cosf(_theta), cosf(_phi) * sin_sun_theta);
    }

    float _theta;
    float _phi;
    void  setSunTheta(float sun_theta)
    {
        _theta   = sun_theta;
        prepared = false;
    }
    void setSunPhi(float sun_phi)
    {
        _phi     = sun_phi;
        prepared = false;
    }
    float3 skyColor(const float3& direction, bool CEL = false);

    float3 getSunDir() { return sunDirection(); }

    float3 sunColor();

    bool prepared = false;
};

}  // namespace Tungsten
