#pragma once

#include "stdafx.hpp"
#include "stdafx.cuh"

namespace detail
{
    template<int WCurr, int... WOther>
    constexpr int calcNonZeroValue()
    {
        static_assert(WCurr==0 or WCurr==-1 or WCurr==1);
        if constexpr (sizeof...(WOther) == 0)
        {
            return static_cast<int>(WCurr != 0);
        }
        else
        {
            return static_cast<int>(WCurr != 0)+calcNonZeroValue<WOther...>();
        }
    }

    template<int I, int N, std::array<int, N> W>
    constexpr int calcNonZeroArr()
    {
        static_assert(W[I]==0 or W[I]==-1 or W[I]==1);
        if constexpr (I == N-1)
        {
            return static_cast<int>(W[I] != 0);
        }
        else
        {
            return static_cast<int>(W[I] != 0) + calcNonZeroArr<I+1, N, W>();
        }
    }

    template<int I, int N, typename Arr>
    constexpr auto calcSum(const Arr &arr)
    {
        if constexpr (I==N-1)
        {
            return arr[I];
        }
        else
        {
            return arr[I]+calcSum<I+1, N, Arr>(arr);
        }
    }

    template<int I, int N, int ValidCnt, std::array<int, N> W, typename Arr>
    constexpr auto zeroOneDotHelper(const Arr& arr)
    {
        if constexpr (ValidCnt == 1 and W[I] != 0)
        {
            if constexpr (W[I] == 1)
            {
                return arr[I];
            }
            else
            {
                return -arr[I];
            }
        }
        else 
        {
            if constexpr (W[I]==1)
            {
                return zeroOneDotHelper<I+1, N, ValidCnt-1, W, Arr>(arr)+arr[I];
            }
            else if constexpr(W[I]==-1)
            {
                return zeroOneDotHelper<I+1, N, ValidCnt-1, W, Arr>(arr)-arr[I];
            }
            else
            {
                return zeroOneDotHelper<I+1, N, ValidCnt, W, Arr>(arr);
            }
        }
    }


    template<int ValidCnt, int WCurr, int... WOther>
    struct ZeroOneDotHelper
    {
        template<typename TCurr, typename... TOther>
        static constexpr TCurr getValue(TCurr valCurr, TOther... valOther)
        {
            if constexpr (ValidCnt == 1 and WCurr != 0)
            {
                if constexpr (WCurr == -1)
                {
                    return -valCurr;
                }
                else
                {
                    return valCurr;
                }
            }
            else
            {
                if constexpr (WCurr == -1)
                {
                    return ZeroOneDotHelper<ValidCnt-1, WOther...>::template getValue<TOther...>(valOther...)-valCurr;
                }
                else if constexpr (WCurr == 1)
                {
                    return ZeroOneDotHelper<ValidCnt-1, WOther...>::template getValue<TOther...>(valOther...)+valCurr;
                }
                else
                {
                    return ZeroOneDotHelper<ValidCnt, WOther...>::template getValue<TOther...>(valOther...);
                }
            }
        }
    };
}

template<int... W, typename... TArgs>
constexpr auto zeroOneDotValues(TArgs... values)
{
    return detail::ZeroOneDotHelper<detail::calcNonZeroValue<W...>(), W...>::template getValue<TArgs...>(values...);
}

template<int N, std::array<int, N> W, typename Arr>
constexpr auto zeroOneDotArr(const Arr& arr)
{
    return detail::zeroOneDotHelper<0, N, detail::calcNonZeroArr<0, N, W>(), W, Arr>(arr);
}

template<typename T>
CUDA_INLINE T fast_mul_add(T a, T b, T c);

template<>
CUDA_INLINE float fast_mul_add(float a, float b, float c)
{
#ifdef __CUDA_ARCH__
    return __fmaf_rn(a, b, c);
#else
    return std::fmaf(a, b, c);
#endif
}

template<int N, typename Arr>
constexpr auto calcSum(const Arr& arr)
{
    return detail::calcSum<0, N, Arr>(arr);
}

template<std::unsigned_integral T>
constexpr T divCeil(T num, T deno)
{
    return (num+deno-1) / deno;
}