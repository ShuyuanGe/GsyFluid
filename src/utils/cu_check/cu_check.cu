#include "stdafx.cuh"
#include "cu_check.cuh"

namespace cu
{
    RuntimeError::RuntimeError(cudaError_t err, const std::source_location &loc) : 
        std::runtime_error(
            std::string("====Cuda Runtime Error====\n")+
            std::string("Location: [")+loc.file_name()+":"+std::to_string(loc.line())+"]\n"+
            std::string("Description: [")+cudaGetErrorString(err)+"]"
        )
    {

    }

    const char * RuntimeError::what() const noexcept
    {
        return std::runtime_error::what();
    }

    void check(const std::source_location &loc)
    {
        auto err = cudaGetLastError();
        if(err != cudaSuccess)
        {
            throw RuntimeError(err, loc);
        }
    }
}