#pragma once

#include "stdafx.hpp"
#include "stdafx.cuh"

namespace cu
{
    template<typename T>
    void malloc(T* &devPtr, std::size_t n)
    {
        cudaMalloc((void**)&devPtr, sizeof(T)*n);
    }

    template<typename T>
    void free(T* &devPtr)
    {
        cudaFree((void*)devPtr);
        devPtr = nullptr;
    }

    template<typename T>
    void memcpy(T* pDst, const T* pSrc, std::size_t n)
    {
        cudaMemcpy((void*)pDst, (const void*)pSrc, sizeof(T)*n, cudaMemcpyDefault);
    }
}