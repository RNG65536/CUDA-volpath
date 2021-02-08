#pragma once

#include "vecmath.h"

struct Context
{
    float3 c0;
    float3 c1;
    float3 c2;
    float3 c3;
    float3 c4;
    float3 inv_divisor_Yxy;
    float3 sky_up;
    float  overcast;
    float3 sun_direction;
    float3 sun_color;
};

/******************************************************************************\
 *
 * Minimal background shader using preetham model.  Applies constant scaling
 * factor to returned result in case post-process tonemapping is not desired
 * and only shows visible sun for 0-depth (eye) rays.
 *
\******************************************************************************/

//------------------------------------------------------------------------------
//
// Implements the Preetham analytic sun/sky model ( Preetham, SIGGRAPH 99 )
//
//------------------------------------------------------------------------------
class PreethamSunSky
{
public:
    PreethamSunSky();

    void setSunTheta(float sun_theta)
    {
        m_sun_theta = sun_theta;
        m_dirty     = true;
    }
    void setSunPhi(float sun_phi)
    {
        m_sun_phi = sun_phi;
        m_dirty   = true;
    }
    void setTurbidity(float turbidity)
    {
        m_turbidity = turbidity;
        m_dirty     = true;
    }

    void setUpDir(const float3& up)
    {
        m_up    = up;
        m_dirty = true;
    }
    void setOvercast(float overcast) { m_overcast = overcast; }

    float getSunTheta() { return m_sun_theta; }
    float getSunPhi() { return m_sun_phi; }
    float getTurbidity() { return m_turbidity; }

    float  getOvercast() { return m_overcast; }
    float3 getUpDir() { return m_up; }
    float3 getSunDir()
    {
        preprocess();
        return m_sun_dir;
    }

    // Query the sun color at current sun position and air turbidity ( kilo-cd /
    // m^2 )
    float3 sunColor();

    // Query the sky color in a given direction ( kilo-cd / m^2 )
    float3 skyColor(const float3& direction, bool CEL = false);

    // Sample the solid angle subtended by the sun at its current position
    //float3 sampleSun() const;

    // Set precomputed Preetham model variables on the given context:
    //   c[0-4]          :
    //   inv_divisor_Yxy :
    //   sun_dir         :
    //   sun_color       :
    //   overcast        :
    //   up              :
    //     void setVariables( Context& context );

private:
    void   preprocess();
    //float3 calculateSunColor();

    // Represents one entry from table 2 in the paper
    struct Datum
    {
        float wavelength;
        float sun_spectral_radiance;
        float k_o;
        float k_wa;
    };

    static const float cie_table[38][4];  // CIE spectral sensitivy curves
    static const Datum data[38];          // Table2

    // Calculate absorption for a given wavelength of direct sunlight
    static float calculateAbsorption(float sun_theta,  // Sun angle from zenith
                                     float m,  // Optical mass of atmosphere
                                     float lambda,     // light wavelength
                                     float turbidity,  // atmospheric turbidity
                                     float k_o,        // atten coeff for ozone
                                     float k_wa);  // atten coeff for h2o vapor

    // Unit conversion helpers
    static float3 XYZ2rgb(const float3& xyz);
    static float3 Yxy2XYZ(const float3& Yxy);
    static float  rad2deg(float rads);

    // Input parameters
    float  m_sun_theta;
    float  m_sun_phi;
    float  m_turbidity;
    float  m_overcast;
    float3 m_up;
    float3 m_sun_color;
    float3 m_sun_dir;

    // Precomputation results
    bool   m_dirty;
    float3 m_c0;
    float3 m_c1;
    float3 m_c2;
    float3 m_c3;
    float3 m_c4;
    float3 m_inv_divisor_Yxy;
};

