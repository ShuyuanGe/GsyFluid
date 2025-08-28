#pragma once

#include "stdafx.hpp"

template<typename DeriveData, typename rho_t>
class DensityBufImpl
{
    protected:
        DensityBufImpl() = default;
    public:
    
        rho_t *getRhoPtr()
        {
            return static_cast<DeriveData*>(this)->rhoPtr;
        }
};