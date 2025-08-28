#pragma once

#include "stdafx.hpp"

template<typename DeriveData, typename ddf_t>
class DoubleDDFBufImpl
{
    protected:
        DoubleDDFBufImpl() = default;
    public:
        ddf_t *getSrcDDFPtr()
        {
            return static_cast<DeriveData*>(this)->srcDDFPtr;
        }

        ddf_t *getDstDDFPtr()
        {
            return static_cast<DeriveData*>(this)->dstDDFPtr;
        }

        void swapDDFPtr() noexcept
        {
            std::swap(
                static_cast<DeriveData*>(this)->srcDDFPtr, 
                static_cast<DeriveData*>(this)->dstDDFPtr
            );
        }

};