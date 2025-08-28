#pragma once

#include "stdafx.hpp"
#include "stdafx.cuh"
#include "cu_memory.cuh"
#include "double_ddf_buf_impl.hpp"

template<int N, typename DDF>
class DoubleDDFBufData : public DoubleDDFBufImpl<DoubleDDFBufData<N, DDF>, DDF>
{
    friend class DoubleDDFBufImpl<DoubleDDFBufData, DDF>;
    public:
        using ddf_t = DDF;
    protected:
        ddf_t* srcDDFPtr = nullptr; 
        ddf_t* dstDDFPtr = nullptr;
    public:
        DoubleDDFBufData(std::size_t n)
        {
            cu::malloc(srcDDFPtr, N*n);
            cu::malloc(dstDDFPtr, N*n);
        }

        ~DoubleDDFBufData() noexcept
        {
            cu::free(srcDDFPtr);
            cu::free(dstDDFPtr);
        }
};
