#pragma once

#include "stdafx.hpp"

template<DeriveData>
class SingleDDFBufImpl
{
    protected:
        SingleDDFBufImpl() = default;
    public:
        using ddf_t = typename DeriveData::ddf_t;

        ddf_t *getDDFPtr()
        {
            return static_cast<DeriveData*>(this)->DDFPtr;
        }
};