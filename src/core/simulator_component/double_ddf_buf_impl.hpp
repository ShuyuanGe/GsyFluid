#pragma once

#include "stdafx.hpp"
#include "stdafx.cuh"

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

        template<int N, typename Arr>
        void setSrcDftDDF(const Arr &dftEqu, std::size_t n)
        {
            thrust::device_ptr<ddf_t> dSrcDDFPtr {getSrcDDFPtr()};
            #pragma unroll
            for(int i=0 ; i<N ; ++i)
            {
                thrust::fill_n(dSrcDDFPtr+i*n, n, dftEqu[i]);
            }
        }

        template<int N, typename Arr>
        void setDstDftDDF(const Arr &dftEqu, std::size_t n)
        {
            thrust::device_ptr<ddf_t> dDstDDFPtr {getDstDDFPtr()};
            #pragma unroll
            for(int i=0 ; i<N ; ++i)
            {
                thrust::fill_n(dDstDDFPtr+i*n, n, dftEqu[i]);
            }
        }

};