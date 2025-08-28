#pragma once

#include "stdafx.hpp"
#include "stdafx.cuh"
#include "cu_memory.cuh"
#include "velocity_buf_impl.hpp"

template<int Dim, typename Vel>
class VelocityBufData;

template<typename Vel>
class VelocityBufData<1, Vel> : public VelocityBufImpl<VelocityBufData<1, Vel>, 1, Vel>
{
    friend class VelocityBufImpl<VelocityBufData, 1, Vel>;
    public:
        using vel_t = Vel;
    protected:
        std::array<vel_t*, 1> velPtr = {};
    public:
        VelocityBufData(std::size_t n)
        {
            cu::malloc(velPtr[0], n);
        }

        ~VelocityBufData() noexcept
        {
            cu::free(velPtr[0]);
        }
};

template<typename Vel>
class VelocityBufData<2, Vel> : public VelocityBufImpl<VelocityBufData<2, Vel>, 2, Vel>
{
    friend class VelocityBufImpl<VelocityBufData, 1, Vel>;
    friend class VelocityBufImpl<VelocityBufData, 2, Vel>;
    public:
        using vel_t = Vel;
    protected:
        std::array<vel_t*, 2> velPtr = {};
    public:
        VelocityBufData(std::size_t n)
        {
            cu::malloc(velPtr[0], n);
            cu::malloc(velPtr[1], n);
        }

        ~VelocityBufData() noexcept
        {
            cu::free(velPtr[0]);
            cu::free(velPtr[1]);
        }
};

template<typename Vel>
class VelocityBufData<3, Vel> : public VelocityBufImpl<VelocityBufData<3, Vel>, 3, Vel>
{
    friend class VelocityBufImpl<VelocityBufData, 1, Vel>;
    friend class VelocityBufImpl<VelocityBufData, 2, Vel>;
    friend class VelocityBufImpl<VelocityBufData, 3, Vel>;
    public:
        using vel_t = Vel;
    protected:
        std::array<vel_t*, 3> velPtr = {};
    public:
        VelocityBufData(std::size_t n)
        {
            cu::malloc(velPtr[0], n);
            cu::malloc(velPtr[1], n);
            cu::malloc(velPtr[2], n);
        }

        ~VelocityBufData() noexcept
        {
            cu::free(velPtr[0]);
            cu::free(velPtr[1]);
            cu::free(velPtr[2]);
        }
};
