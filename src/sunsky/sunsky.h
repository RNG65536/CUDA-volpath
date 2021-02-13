#pragma once

#include "sky_preetham.h"
#include "sky_tungsten.h"

template <class SkyModelImpl>
class SkyModel
{
    SkyModelImpl impl;

public:
    SkyModel() {}

    float3 skyColor(const float3& dir, bool show_sun) { return impl.skyColor(dir, show_sun); }

    // azimuthal
    void setSunPhi(float phi) { impl.setSunPhi(phi); }

    // zenithal
    void setSunTheta(float theta) { impl.setSunTheta(theta); }

    float3 getSunDir() { return impl.getSunDir(); }

    float3 sunColor() { return impl.sunColor(); }
};