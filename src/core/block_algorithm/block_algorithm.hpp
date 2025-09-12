#pragma once

#include "math.hpp"
#include "stdafx.hpp"

namespace block_algorithm
{
    template<std::integral T>
    constexpr bool validBlkConfig(T domSize, T blkSize, T blkIter)
    {
        if(blkSize<domSize and blkSize+1<=blkIter) return false;
        if(2*(blkSize+1-blkIter)<domSize and blkSize+2<=blkIter) return false;
        return true;
    }

    template<std::integral T>
    constexpr T calcBlkNum(T domSize, T blkSize, T blkIter)
    {
        if(domSize <= blkSize)
        {
            return 1;
        }
        if(domSize<=2*(blkSize+1-blkIter))
        {
            return 2;
        }
        return 2+divCeil<unsigned>(domSize-2*(blkSize+1-blkIter), blkSize+2-2*blkIter);
    }

    template<std::integral T>
    constexpr T calcValidPrev(T blkIdx, T blkNum, T blkSize, T blkIter, T domSize)
    {
        if(blkIdx==blkNum)
        {
            return domSize;
        }
        if(blkIdx==0)
        {
            return 0;
        }
        return (blkSize+1-blkIter)+(blkIdx-1)*(blkSize+2-2*blkIter);
    }
}