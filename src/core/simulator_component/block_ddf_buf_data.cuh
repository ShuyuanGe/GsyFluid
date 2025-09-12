#pragma once

#include "stdafx.hpp"
#include "stdafx.cuh"
#include "cu_memory.cuh"
#include "block_ddf_buf_impl.hpp"

template<int N, typename DDF>
class BlockDDFBufData : public BlockDDFBufImpl<BlockDDFBufData<N, DDF>, DDF>
{
    friend class BlockDDFBufImpl<BlockDDFBufData, DDF>;
    public:
        using ddf_t = DDF;
    protected:
        ddf_t *evenBufPtr   = nullptr;
        ddf_t *oddBufPtr    = nullptr;
    public:
        BlockDDFBufData(std::size_t n)
        {
            cu::malloc(evenBufPtr, N*n);
            cu::malloc(oddBufPtr,  N*n);
        }

        ~BlockDDFBufData() noexcept
        {
            cu::free(evenBufPtr);
            cu::free(oddBufPtr );
        }
};