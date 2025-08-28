
cmake_minimum_required(VERSION 3.30)

set(CMAKE_CXX_COMPILER /usr/bin/g++-12)
set(CMAKE_CUDA_ARCHITECTURES 70)
set(CMAKE_CUDA_COMPILER /usr/local/cuda-12.2/bin/nvcc)

project(GsyFluid LANGUAGES CUDA CXX)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CUDA_STANDARD 20)
set(CMAKE_CUDA_STANDARD_REQUIRED ON)

set(CUDA_COMPILE_FLAG PRIVATE --extended-lambda --expt-relaxed-constexpr)

add_subdirectory(utils)
add_subdirectory(core)
add_subdirectory(example)

add_executable(GsyFluidMain main.cpp)
