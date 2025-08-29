#pragma once

#include "stdafx.hpp"
#include "stdafx.cuh"
#include "flag_buf_data.cuh"
#include "density_buf_data.cuh"
#include "velocity_buf_data.cuh"
#include "single_ddf_buf_data.cuh"

using Real = float;
using Int  = std::int32_t;
using UInt = std::uint32_t;
using Flag = std::uint32_t;

class D3Q27SingleDDFSimulator : 
    public FlagBufData<Flag>, 
    public DensityBufData<Real>, 
    public VelocityBufData<3, Real>, 
    public SingleDDFBufData<27, Real>
{
    public:
        static constexpr Real invTau = 1.f / 0.7f;
        static constexpr Int nx = 512;
        static constexpr Int ny = 256;
        static constexpr Int nz = 256;
        static constexpr Int size = nx*ny*nz;

        static constexpr Flag LOAD_DDF_BIT      = 1 << 0;
        static constexpr Flag EQU_DDF_BIT       = 1 << 1;

        static constexpr Flag COLLIDE_BIT       = 1 << 2;
        
        static constexpr Flag STORE_DDF_BIT     = 1 << 3;
        static constexpr Flag STORE_RHO_BIT     = 1 << 4;
        static constexpr Flag STORE_U_BIT       = 1 << 5;
    protected:
        UInt                timeStep;
        Real                dftRho;
        std::array<Real, 3> dftV;
        cudaStream_t stream;
        cudaEvent_t start, end;

        static constexpr Flag FLUID_FLAG            = LOAD_DDF_BIT | COLLIDE_BIT | STORE_DDF_BIT | STORE_RHO_BIT | STORE_U_BIT;
        static constexpr Flag BOUNCE_BACK_BC_FLAG   = 0;
        static constexpr Flag EQU_BC_FLAG           = EQU_DDF_BIT | STORE_DDF_BIT;
    public:
        D3Q27SingleDDFSimulator(Real dftRho = 1.f, std::array<Real, 3> dftV = {0.f, 0.f, 0.f});
        ~D3Q27SingleDDFSimulator() noexcept;
        void setup();
        void run(UInt step);
};