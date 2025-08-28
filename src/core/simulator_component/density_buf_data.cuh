#pragma once

#include "stdafx.hpp"
#include "stdafx.cuh"
#include "cu_memory.cuh"
#include "density_buf_impl.hpp"

template<typename Rho>
class DensityBufData : public DensityBufImpl<DensityBufData<Rho>, Rho>
{
    friend class DensityBufImpl<DensityBufData, Rho>;
    public:
        using rho_t = Rho;
    protected:
        rho_t* rhoPtr = nullptr;
    public:
        DensityBufData(std::size_t n)
        {
            cu::malloc(rhoPtr, n);
        }

        ~DensityBufData() noexcept
        {
            cu::free(rhoPtr);
        }
};