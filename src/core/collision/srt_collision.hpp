#pragma once

#include "math.hpp"
#include "velocity_set.hpp"

namespace detail
{
    template<int I, int Q, typename T, typename Arr>
    CUDA_INLINE void calcEqu3DHelper(Arr &arr, const T& rhoi, const T& vxi_, const T& vyi_, const T& vzi_, const T& c3)
    {
        using VelSet = VelocitySet3D<Q>;
        if constexpr (I < VelSet::centIdx)
        {
            const T cDotUTerm = zeroOneDotValues<VelSet::dx[(Q-1)-I], VelSet::dy[(Q-1)-I], VelSet::dz[(Q-1)-I]>(vxi_, vyi_, vzi_);
            const T rhoi_ = VelSet::srtOmega[I]*rhoi;
            arr[      I] = fast_mul_add<T>(rhoi_, fast_mul_add<T>(0.5, fast_mul_add<T>(cDotUTerm, cDotUTerm, c3), -cDotUTerm), rhoi_);
            arr[(Q-1)-I] = fast_mul_add<T>(rhoi_, fast_mul_add<T>(0.5, fast_mul_add<T>(cDotUTerm, cDotUTerm, c3),  cDotUTerm), rhoi_);
            calcEqu3DHelper<I+1, Q, T>(arr, rhoi, vxi_, vyi_, vzi_, c3);
        }
        else
        {
            arr[      I] = VelSet::srtOmega[I]*fast_mul_add<T>(rhoi, static_cast<T>(0.5)*c3, rhoi);
        }
    }

    template<int I, int Q, typename T, T Omega, typename Arr>
    CUDA_INLINE void relaxationHelper(Arr &fni, const Arr &feqi)
    {
        if constexpr (I < Q)
        {
            fni[I] += Omega*(feqi[I]-fni[I]);
            relaxationHelper<I+1, Q, T, Omega, Arr>(fni, feqi);
        }
    }
}

namespace srt
{
    template<int Q, typename T, typename Arr>
    CUDA_INLINE void calcEqu3D(Arr &arr, const T& rhoi, const T& vxi, const T& vyi, const T& vzi)
    {
        const T c3 = static_cast<T>(-3)*(vxi*vxi+vyi*vyi+vzi*vzi);
        const T vxi_ = static_cast<T>(3)*vxi;
        const T vyi_ = static_cast<T>(3)*vyi;
        const T vzi_ = static_cast<T>(3)*vzi;
        detail::calcEqu3DHelper<0, Q, T>(arr, rhoi, vxi_, vyi_, vzi_, c3);
    }

    template<int Q, typename T, typename Arr>
    CUDA_INLINE void calcRhoU3D(T& rhoi, T& vxi, T& vyi, T& vzi, const Arr &arr)
    {
        using VelSet = VelocitySet3D<Q>;
        rhoi = calcSum<Q, Arr>(arr);
        vxi = zeroOneDotArr<Q, VelSet::dx, Arr>(arr);
        vyi = zeroOneDotArr<Q, VelSet::dy, Arr>(arr);
        vzi = zeroOneDotArr<Q, VelSet::dz, Arr>(arr);
        vxi /= rhoi;
        vyi /= rhoi;
        vzi /= rhoi;
    }

    template<int Q, typename T, T Omega, typename Arr>
    CUDA_INLINE void relaxation(Arr &fni, const Arr &feqi)
    {
        detail::relaxationHelper<0, Q, T, Omega, Arr>(fni, feqi);
    }
}