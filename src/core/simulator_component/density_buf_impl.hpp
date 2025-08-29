#pragma once

#include "stdafx.hpp"
#include "stdafx.cuh"

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

        void setDftRho(rho_t dftRho, std::size_t n)
        {
            thrust::device_ptr<rho_t> dRhoPtr {getRhoPtr()};
            thrust::fill_n(dRhoPtr, n, dftRho);
        }
};