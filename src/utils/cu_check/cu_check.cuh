#pragma once

#include "stdafx.hpp"
#include "stdafx.cuh"

namespace cu
{
    class RuntimeError : public std::runtime_error
    {
        public:
            RuntimeError(cudaError_t err, const std::source_location &loc);
            const char* what() const noexcept;
    };
    
    void check(const std::source_location &loc = std::source_location::current());
}