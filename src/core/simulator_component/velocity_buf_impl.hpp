#pragma once

#include "stdafx.hpp"
#include "stdafx.cuh"

template<typename DeriveData, int Dim, typename vel_t>
class VelocityBufImpl;

template<typename DeriveData, typename vel_t>
class VelocityBufImpl<DeriveData, 1, vel_t>
{
    protected:
        VelocityBufImpl() = default;
    public:
        vel_t *getVxPtr()
        {
            return static_cast<DeriveData*>(this)->velPtr[0];
        }

        void setDftVx(vel_t dftVx, std::size_t n)
        {
            thrust::device_ptr<vel_t> dVxPtr {getVxPtr()};
            thrust::fill_n(dVxPtr, n, dftVx);
        }
};

template<typename DeriveData, typename vel_t>
class VelocityBufImpl<DeriveData, 2, vel_t> : public VelocityBufImpl<DeriveData, 1, vel_t>
{
    protected:
        VelocityBufImpl() = default;
    public:
        vel_t *getVyPtr()
        {
            return static_cast<DeriveData*>(this)->velPtr[1];
        }

        void setDftVy(vel_t dftVy, std::size_t n)
        {
            thrust::device_ptr<vel_t> dVyPtr {getVyPtr()};
            thrust::fill_n(dVyPtr, n, dftVy);
        }
};

template<typename DeriveData, typename vel_t>
class VelocityBufImpl<DeriveData, 3, vel_t> : public VelocityBufImpl<DeriveData, 2, vel_t>
{
    protected:
        VelocityBufImpl() = default;
    public:
        vel_t* getVzPtr()
        {
            return static_cast<DeriveData*>(this)->velPtr[2];
        }

        void setDftVz(vel_t dftVz, std::size_t n)
        {
            thrust::device_ptr<vel_t> dVzPtr {getVzPtr()};
            thrust::fill_n(dVzPtr, n, dftVz);
        }
};