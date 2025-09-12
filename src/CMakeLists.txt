cmake_minimum_required(VERSION 3.30)

set(CMAKE_CUDA_ARCHITECTURES 89)
set(CMAKE_CUDA_HOST_COMPILER /usr/bin/g++)
set(CMAKE_CUDA_COMPILER /home/xusj/.local/envs/cuda/bin/nvcc)
set(CMAKE_CXX_COMPILER /usr/bin/g++)

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
