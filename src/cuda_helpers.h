#pragma once

#define NOMINMAX
#ifdef min
#undef min
#endif
#ifdef max
#undef max
#endif
#include "cuda/helper_cuda.h"
#include "cuda/helper_math.h"
#include "cuda/helper_string.h"
#include "cuda/helper_timer.h"

inline int divideUp(int a, int b) { return (a + b - 1) / b; }
