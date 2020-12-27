# CUDA-volpath

CUDA based volumetric path tracing for rendering heterogeneous and chromatic volumes. Using multiple techniques to accelerate the process, including reduced scattering coefficients[1][2] and spectral tracking[3].

The vdbloader utility requires installing the OpenVDB library and its dependencies, and can be used to read the volumetric cloud data from [WDAS](https://www.disneyanimation.com/data-sets/). The following render converges in several minutes.

![](1.jpg)

If built without OpenVDB support, a procedural Julia Set is used.

![](2.jpg)

References
----------
[1] Jensen et al., A Practical Model for Subsurface Light Transport
[2] Burley et al., The Design and Evolution of Disney's Hyperion Renderer
[3] Kutz et al., Spectral and Decomposition Tracking for Rendering Heterogeneous Volumes