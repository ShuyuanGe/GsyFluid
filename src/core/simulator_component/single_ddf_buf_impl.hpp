#pragma once

#include "stdafx.hpp"
#include "stdafx.cuh"

template<typename DeriveData, typename ddf_t>
class SingleDDFBufImpl
{
    protected:
        SingleDDFBufImpl() = default;
    public:
        ddf_t *getDDFPtr()
        {
            return static_cast<DeriveData*>(this)->DDFPtr;
        }

        template<int N, typename Arr>
        void setDftDDF(const Arr &dftEqu, std::size_t n)
        {
            thrust::device_ptr<ddf_t> dDDFPtr {getDDFPtr()};
            #pragma unroll
            for(int i=0 ; i<N ; ++i)
            {
                thrust::fill_n(dDDFPtr+i*n, n, dftEqu[i]);
            }
        }
};