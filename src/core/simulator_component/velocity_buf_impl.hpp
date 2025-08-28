#pragma once

#include "stdafx.hpp"

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
};