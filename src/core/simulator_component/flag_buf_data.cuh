#pragma once

#include "stdafx.hpp"
#include "stdafx.cuh"
#include "cu_memory.cuh"
#include "flag_buf_impl.hpp"

template<typename Flag>
class FlagBufData : public FlagBufImpl<FlagBufData<Flag>, Flag>
{
    friend class FlagBufImpl<FlagBufData, Flag>;
    public:
        using flag_t = Flag;
    protected:
        flag_t* flagPtr = nullptr;
    public:
        FlagBufData(std::size_t n)
        {
            cu::malloc(flagPtr, n);
        }
    
        ~FlagBufData() noexcept
        {
            cu::free(flagPtr);
        }
};