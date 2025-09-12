#pragma once

#include "stdafx.hpp"
#include "stdafx.cuh"

template<typename DeriveData, typename ddf_t>
class BlockDDFBufImpl
{
    protected:
        BlockDDFBufImpl() = default;
    public:

        ddf_t *getBlockEvenPtr()
        {
            return static_cast<DeriveData*>(this)->evenBufPtr;
        }

        ddf_t *getBlockOddPtr()
        {
            return static_cast<DeriveData*>(this)->oddBufPtr;
        }
};