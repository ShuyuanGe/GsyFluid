#pragma once

#include "stdafx.hpp"
#include "stdafx.cuh"

template<typename DeriveData, typename flag_t>
class FlagBufImpl
{
    protected:
        FlagBufImpl() = default;
    public:
        flag_t *getFlagPtr()
        {
            return static_cast<DeriveData*>(this)->flagPtr;
        }

        void setDftFlag(flag_t dftFlag, std::size_t n)
        {
            thrust::device_ptr<flag_t> dFlagPtr {getFlagPtr()};
            thrust::fill_n(dFlagPtr, n, dftFlag);
        }
};