#pragma once


#include <array>
#include <cmath>
#include <vector>
#include <cstdint>
#include <fstream>
#include <concepts>
#include <iostream>
#include <optional>
#include <stdexcept>
#include <source_location>

#ifdef __CUDA_ARCH__
    #define CUDA_INLINE __device__ __forceinline__
#else
    #define CUDA_INLINE inline
#endif