#include "cu_check.cuh"
#include "cu_memory.cuh"
#include "invalid_free.hpp"

void raise_invalid_free()
{
    int x = 1;
    int *devPtr = &x;
    cu::free(devPtr);
    cu::check();
}