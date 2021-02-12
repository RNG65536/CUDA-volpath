#pragma once

__device__ unsigned int Hash(unsigned int seed)
{
    seed = (seed ^ 61) ^ (seed >> 16);
    seed *= 9;
    seed = seed ^ (seed >> 4);
    seed *= 0x27d4eb2d;
    seed = seed ^ (seed >> 15);
    return seed;
}

__device__ unsigned int RngNext(unsigned int& seed_x, unsigned int& seed_y)
{
    unsigned int result = seed_x * 0x9e3779bb;

    seed_y ^= seed_x;
    seed_x = ((seed_x << 26) | (seed_x >> (32 - 26))) ^ seed_y ^ (seed_y << 9);
    seed_y = (seed_x << 13) | (seed_x >> (32 - 13));

    return result;
}

__device__ float Rand(unsigned int& seed_x, unsigned int& seed_y)
{
    unsigned int u = 0x3f800000 | (RngNext(seed_x, seed_y) >> 9);
    return __uint_as_float(u) - 1.0;
}

class CudaRng
{
    unsigned int seed_x, seed_y;

public:
    __device__ void init(unsigned int pixel_x, unsigned int pixel_y, unsigned int frame_idx)
    {
        unsigned int s0 = (pixel_x << 16) | pixel_y;
        unsigned int s1 = frame_idx;

        seed_x = Hash(s0);
        seed_y = Hash(s1);
        RngNext(seed_x, seed_y);
    }

    __device__ float next() { return Rand(seed_x, seed_y); }
};
