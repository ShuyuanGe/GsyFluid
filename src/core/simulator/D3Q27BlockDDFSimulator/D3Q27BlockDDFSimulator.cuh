#pragma once

#include "stdafx.hpp"
#include "stdafx.cuh"
#include "flag_buf_data.cuh"
#include "block_algorithm.hpp"
#include "density_buf_data.cuh"
#include "velocity_buf_data.cuh"
#include "block_ddf_buf_data.cuh"
#include "double_ddf_buf_data.cuh"

using Real = float;
using Int  = std::int32_t;
using UInt = std::uint32_t;
using Flag = std::uint32_t;


class D3Q27BlockDDFSimulator : 
    public FlagBufData<Flag>, 
    public DensityBufData<Real>, 
    public VelocityBufData<3, Real>, 
    public DoubleDDFBufData<27, Real>, 
    public BlockDDFBufData<27, Real>
{
    public:
        static constexpr Real   invTau = 1.f / 0.7f;
        static constexpr Int    domX = 512;
        static constexpr Int    domY = 256;
        static constexpr Int    domZ = 256;
        static constexpr Int    domSize = domX*domY*domZ;

        static constexpr Int    blockX  = 64;
        static constexpr Int    blockY  = 16;
        static constexpr Int    blockZ  = 1;
        static constexpr Int    gridX   = 38;
        static constexpr Int    gridY   = 3;
        static constexpr Int    gridZ   = 1;

        static constexpr Int    blkX = blockX;
        static constexpr Int    blkY = gridY * blockY;
        static constexpr Int    blkZ = gridX;
        static constexpr Int    blkSize = blkX*blkY*blkZ;
        static constexpr Int    blkIter = 4;

        static constexpr Flag   LOAD_DDF_BIT        = 1u << 0;
        static constexpr Flag   REV_LOAD_DDF_BIT    = 1u << 1;
        static constexpr Flag   EQU_DDF_BIT         = 1u << 2;
        
        static constexpr Flag   COLLIDE_BIT         = 1u << 8;
        
        static constexpr Flag   STORE_DDF_BIT       = 1u << 16;
        static constexpr Flag   STORE_RHO_BIT       = 1u << 17;
        static constexpr Flag   STORE_U_BIT         = 1u << 18;
        
        static constexpr Flag   BLOCK_CORRECT_BIT   = 1u << 31;

        static constexpr Flag FLUID_FLAG            = LOAD_DDF_BIT | COLLIDE_BIT | STORE_DDF_BIT | STORE_RHO_BIT | STORE_U_BIT;
        static constexpr Flag BOUNCE_BACK_BC_FLAG   = REV_LOAD_DDF_BIT | STORE_DDF_BIT;
        static constexpr Flag EQU_BC_FLAG           = EQU_DDF_BIT | STORE_DDF_BIT;

        static constexpr Flag FINAL_STORE_DDF_FLAG  = STORE_DDF_BIT | BLOCK_CORRECT_BIT;
        static constexpr Flag FINAL_STORE_RHO_FLAG  = STORE_RHO_BIT | BLOCK_CORRECT_BIT;
        static constexpr Flag FINAL_STORE_U_FLAG    = STORE_U_BIT   | BLOCK_CORRECT_BIT;

        static_assert(blkIter > 0);
        static_assert(block_algorithm::validBlkConfig(domX, blkX, blkIter));
        static_assert(block_algorithm::validBlkConfig(domY, blkY, blkIter));
        static_assert(block_algorithm::validBlkConfig(domZ, blkZ, blkIter));

        static constexpr Int    blkNumX = block_algorithm::calcBlkNum(domX, blkX, blkIter);
        static constexpr Int    blkNumY = block_algorithm::calcBlkNum(domY, blkY, blkIter);
        static constexpr Int    blkNumZ = block_algorithm::calcBlkNum(domZ, blkZ, blkIter);
        static constexpr Int    blkNum  = blkNumX*blkNumY*blkNumZ;
    protected:
        UInt timeStep;
        Real dftRho;
        std::array<Real, 3> dftV;
        cudaStream_t stream;
        cudaEvent_t start, end;
    public:
        D3Q27BlockDDFSimulator(Real dftRho = 1.f, std::array<Real, 3> dftV = {0.f, 0.f, 0.f});
        ~D3Q27BlockDDFSimulator() noexcept;
        void setup();
        void run();
};