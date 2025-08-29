#pragma once

#include "stdafx.hpp"
#include "stdafx.cuh"
#include "cu_memory.cuh"
#include "single_ddf_buf_impl.hpp"

template<int N, typename DDF>
class SingleDDFBufData : public SingleDDFBufImpl<SingleDDFBufData<N, DDF>, DDF>
{
    friend class SingleDDFBufImpl<SingleDDFBufData, DDF>;
    public:
        using ddf_t = DDF;
    protected:
        ddf_t* DDFPtr = nullptr;
    public:
        SingleDDFBufData(std::size_t n)
        {
            cu::malloc(DDFPtr, N*n);
        }

        ~SingleDDFBufData() noexcept
        {
            cu::free(DDFPtr);
        }
};