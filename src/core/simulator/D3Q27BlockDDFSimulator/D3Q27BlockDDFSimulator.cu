#include "math.hpp"
#include "stdafx.cuh"
#include "cu_check.cuh"
#include "cu_memory.cuh"
#include "srt_collision.hpp"
#include "D3Q27BlockDDFSimulator.cuh"

struct D3Q27BlockDDFKernelParam
{
    UInt blkXOff = 0, blkYOff = 0, blkZOff = 0;
    Real *srcDDF = nullptr, *dstDDF = nullptr;
    Real *blkEvenBuf = nullptr, *blkOddBuf = nullptr;
    Real *rho = nullptr;
    Real *vx = nullptr, *vy = nullptr, *vz = nullptr;
    Flag *blkFlag = nullptr;
};

__global__ void D3Q27BlockDDFKernel(const D3Q27BlockDDFKernelParam __grid_constant__ param);

D3Q27BlockDDFSimulator::D3Q27BlockDDFSimulator(
    Real dftRho, 
    std::array<Real, 3> dftV
) : 
    FlagBufData(blkNum*blkSize), 
    DensityBufData(domSize), 
    VelocityBufData(domSize), 
    DoubleDDFBufData(domSize), 
    BlockDDFBufData(blkSize), 
    timeStep(0), 
    dftRho(dftRho), 
    dftV(dftV)
{
    cudaStreamCreate(&stream);
    cudaEventCreate(&start);
    cudaEventCreate(&end);

    setDftFlag(0, blkNum*blkSize);

    setDftRho(dftRho, domSize);

    setDftVx(dftV[0], domSize);
    setDftVy(dftV[1], domSize);
    setDftVz(dftV[2], domSize);

    Real dftFeq[27];
    srt::calcEqu3D<27, Real>(dftFeq, dftRho, dftV[0], dftV[1], dftV[2]);
    setSrcDftDDF<27>(dftFeq, domSize);
    setDstDftDDF<27>(dftFeq, domSize);
}

D3Q27BlockDDFSimulator::~D3Q27BlockDDFSimulator() noexcept
{
    cudaStreamDestroy(stream);
    cudaEventDestroy(start);
    cudaEventDestroy(end);
}

void D3Q27BlockDDFSimulator::setup()
{
    std::vector<Flag> domFlagBuf (domSize);
    std::vector<Real> domRhoBuf  (domSize, dftRho);
    std::vector<Real> domVxBuf   (domSize, dftV[0]);
    std::vector<Real> domVyBuf   (domSize, dftV[1]);
    std::vector<Real> domVzBuf   (domSize, dftV[2]);

    const float ballCentX = domX / 4;
    const float ballCentY = domY / 2;
    const float ballCentZ = domZ / 2;
    const float ballR = 40;

    for(Int i=0 ; i<domSize ; ++i)
    {
        const Int x = i % domX;
        const Int y = (i / domX) % domY;
        const Int z = i / (domX * domY);
        if(y==0 or y==domY or z==0 or z==domZ-1)
        {
            domFlagBuf[i] = BOUNCE_BACK_BC_FLAG;
        }
        else if (x==0)
        {
            domFlagBuf[i] = EQU_BC_FLAG;
            domVxBuf[i] = 0.01;
        }
        else if (x==domX-1)
        {
            domFlagBuf[i] = EQU_BC_FLAG;
        }
        else if (distBall3D<Real>(x, y, z, ballCentX, ballCentY, ballCentZ) < ballR)
        {
            domFlagBuf[i] = BOUNCE_BACK_BC_FLAG;
            domRhoBuf[i] = 0;
        }
        else
        {
            domFlagBuf[i] = FLUID_FLAG;
        }
    }

    cu::memcpy(getRhoPtr(), domRhoBuf.data(), domSize);
    cu::memcpy(getVxPtr(), domVxBuf.data(), domSize);
    cu::memcpy(getVyPtr(), domVyBuf.data(), domSize);
    cu::memcpy(getVzPtr(), domVzBuf.data(), domSize);

    domRhoBuf.clear();
    domVxBuf.clear();
    domVyBuf.clear();
    domVzBuf.clear();

    std::vector<Flag> blkFlagBuf (blkNum*blkSize, 0);

    UInt istIdx = 0;
    for(UInt blkZIdx=0 ; blkZIdx<blkNumZ ; ++blkZIdx)
    {
        const Int blkZStoreStart = block_algorithm::calcValidPrev<Int>(blkZIdx  , blkNumZ, blkZ, blkIter, domZ);
        const Int blkZStoreEnd   = block_algorithm::calcValidPrev<Int>(blkZIdx+1, blkNumZ, blkZ, blkIter, domZ);
        const Int blkZLoadStart  = std::max<Int>(blkZStoreStart-(blkIter-1), 0);
        const Int blkZLoadEnd    = std::min<Int>(blkZStoreEnd  +(blkIter-1), domZ);
        for(UInt blkYIdx=0 ; blkYIdx<blkNumY ; ++blkYIdx)
        {
            const Int blkYStoreStart = block_algorithm::calcValidPrev<Int>(blkYIdx  , blkNumY, blkY, blkIter, domY);
            const Int blkYStoreEnd   = block_algorithm::calcValidPrev<Int>(blkYIdx+1, blkNumY, blkY, blkIter, domY);
            const Int blkYLoadStart  = std::max<Int>(blkYStoreStart-(blkIter-1), 0);
            const Int blkYLoadEnd    = std::min<Int>(blkYStoreEnd  +(blkIter-1), domY);
            for(UInt blkXIdx=0 ; blkXIdx<blkNumX ; ++blkXIdx)
            {
                const Int blkXStoreStart = block_algorithm::calcValidPrev<Int>(blkXIdx  , blkNumX, blkX, blkIter, domX);
                const Int blkXStoreEnd   = block_algorithm::calcValidPrev<Int>(blkXIdx+1, blkNumX, blkX, blkIter, domX);
                const Int blkXLoadStart  = std::max<Int>(blkXStoreStart-(blkIter-1), 0);
                const Int blkXLoadEnd    = std::min<Int>(blkXStoreEnd  +(blkIter-1), domX);
                for(UInt blkZOff=0, glbZOff=blkZLoadStart ; glbZOff<blkZLoadEnd ; ++blkZOff, ++glbZOff)
                {
                    for(UInt blkYOff=0, glbYOff=blkYLoadStart ; glbYOff<blkYLoadEnd ; ++blkYOff, ++glbYOff)
                    {
                        for(UInt blkXOff=0, glbXOff=blkXLoadStart ; glbXOff<blkXLoadEnd ; ++blkXOff, ++glbXOff)
                        {
                            const UInt glbOff = glbXOff+domX*(glbYOff+domY*glbZOff);
                            const UInt blkOff = blkXOff+blkX*(blkYOff+blkY*blkZOff);

                            if(
                                blkZStoreStart<=glbZOff and glbZOff<blkZStoreEnd and
                                blkYStoreStart<=glbYOff and glbYOff<blkYStoreEnd and 
                                blkXStoreStart<=glbXOff and glbXOff<blkXStoreEnd
                            )
                            {
                                blkFlagBuf[istIdx+blkOff] = domFlagBuf[glbOff] | BLOCK_CORRECT_BIT;
                            }
                            else
                            {
                                blkFlagBuf[istIdx+blkOff] = domFlagBuf[glbOff];
                            }
                        }
                    }
                }
                istIdx += blkSize;
            }
        }
    }
    
    cu::memcpy(getFlagPtr(), blkFlagBuf.data(), blkNum*blkSize);
}

void D3Q27BlockDDFSimulator::run()
{
    float ms;

    D3Q27BlockDDFKernelParam param
    {
        .blkXOff = 0,
        .blkYOff = 0, 
        .blkZOff = 0, 
        .srcDDF  = nullptr, 
        .dstDDF  = nullptr, 
        .blkEvenBuf = getBlockEvenPtr(), 
        .blkOddBuf  = getBlockOddPtr(), 
        .rho = getRhoPtr(), 
        .vx  = getVxPtr(), 
        .vy  = getVyPtr(), 
        .vz  = getVzPtr(), 
        .blkFlag = nullptr        
    };

    void* kernelArgs[1] = {(void*)&param};

    cudaEventRecord(start, stream);
    param.srcDDF = getSrcDDFPtr();
    param.dstDDF = getDstDDFPtr();
    for(Int blkIdx=0 ; blkIdx<blkNum ; ++blkIdx)
    {
        const Int blkXIdx = blkIdx % blkNumX;
        const Int blkYIdx = (blkIdx / blkNumX) % blkNumY;
        const Int blkZIdx = blkIdx / (blkNumX * blkNumY);
        const Int zOff = std::max<Int>(0, block_algorithm::calcValidPrev<Int>(blkZIdx, blkNumZ, blkZ, blkIter, domZ)-(blkIter-1));
        const Int yOff = std::max<Int>(0, block_algorithm::calcValidPrev<Int>(blkYIdx, blkNumY, blkY, blkIter, domY)-(blkIter-1));
        const Int xOff = std::max<Int>(0, block_algorithm::calcValidPrev<Int>(blkXIdx, blkNumX, blkX, blkIter, domX)-(blkIter-1));
        param.blkXOff = xOff;
        param.blkYOff = yOff;
        param.blkZOff = zOff;
        param.blkFlag = getFlagPtr()+blkIdx*blkSize;
        //printf("xOff: %3d, yOff: %3d, zOff: %3d\n", xOff, yOff, zOff);
        cudaLaunchCooperativeKernel((const void*)&D3Q27BlockDDFKernel, dim3{gridX, gridY, gridZ}, dim3{blockX, blockY, blockZ}, &kernelArgs[0], 0, stream);
    }
    swapDDFPtr();
    ++timeStep;
    cudaEventRecord(end, stream);
    cudaEventSynchronize(end);
    cudaEventElapsedTime(&ms, start, end);
    cu::check();
    const float mlups = (static_cast<float>(domSize)/(1024*1024)) / (ms/1000) * blkIter;
    printf("[Info]: D3Q27BlockDDFSimulator run, speed = %.2f (MLUPS)\n", mlups);
}

#define defBlkX     static_cast<Int>(threadIdx.x)
#define defBlkY     static_cast<Int>(blockIdx.y*D3Q27BlockDDFSimulator::blockY+threadIdx.y)
#define defBlkZ     static_cast<Int>(blockIdx.x)
#define defBlkNx    static_cast<Int>(D3Q27BlockDDFSimulator::blkX)
#define defBlkNy    static_cast<Int>(D3Q27BlockDDFSimulator::blkY)
#define defBlkNz    static_cast<Int>(D3Q27BlockDDFSimulator::blkZ)
#define defDomNx    static_cast<Int>(D3Q27BlockDDFSimulator::domX)
#define defDomNy    static_cast<Int>(D3Q27BlockDDFSimulator::domY)
#define defDomNz    static_cast<Int>(D3Q27BlockDDFSimulator::domZ)

__device__ __forceinline__ void D3Q27PullLoadDDF(Int idx, Int n, Int pdx, Int ndx, Int pdy, Int ndy, Int pdz, Int ndz, const Real *ddf, Real(&fni)[27])
{
    /*load f[x:-,y:-,z:-] from nbr[x:+,y:+,z:+]*/
    fni[ 0] = ddf[ 0*n+idx+pdx+pdy+pdz]; 
    /*load f[x:0,y:-,z:-] from nbr[x:0,y:+,z:+]*/
    fni[ 1] = ddf[ 1*n+idx+    pdy+pdz]; 
    /*load f[x:+,y:-,z:-] from nbr[x:-,y:+,z:+]*/
    fni[ 2] = ddf[ 2*n+idx+ndx+pdy+pdz];
    /*load f[x:-,y:0,z:-] from nbr[x:+,y:0,z:+]*/
    fni[ 3] = ddf[ 3*n+idx+pdx+    pdz];
    /*load f[x:0,y:0,z:-] from nbr[x:0,y:0,z:+]*/
    fni[ 4] = ddf[ 4*n+idx+        pdz];
    /*load f[x:+,y:0,z:-] from nbr[x:-,y:0,z:+]*/
    fni[ 5] = ddf[ 5*n+idx+ndx+    pdz];
    /*load f[x:-,y:+,z:-] from nbr[x:+,y:-,z:+]*/
    fni[ 6] = ddf[ 6*n+idx+pdx+ndy+pdz];
    /*load f[x:0,y:+,z:-] from nbr[x:0,y:-,z:+]*/
    fni[ 7] = ddf[ 7*n+idx+    ndy+pdz];
    /*load f[x:+,y:+,z:-] from nbr[x:-,y:-,z:+]*/
    fni[ 8] = ddf[ 8*n+idx+ndx+ndy+pdz];
                                                
    /*load f[x:-,y:-,z:0] from nbr[x:+,y:+,z:0]*/
    fni[ 9] = ddf[ 9*n+idx+pdx+pdy    ];
    /*load f[x:0,y:-,z:0] from nbr[x:0,y:+,z:0]*/
    fni[10] = ddf[10*n+idx+    pdy    ];
    /*load f[x:+,y:-,z:0] from nbr[x:-,y:+,z:0]*/
    fni[11] = ddf[11*n+idx+ndx+pdy    ];
    /*load f[x:-,y:0,z:0] from nbr[x:+,y:0,z:0]*/
    fni[12] = ddf[12*n+idx+pdx        ];
    /*load f[x:0,y:0,z:0] from nbr[x:0,y:0,z:0]*/
    fni[13] = ddf[13*n+idx            ];
    /*load f[x:+,y:0,z:0] from nbr[x:-,y:0,z:0]*/
    fni[14] = ddf[14*n+idx+ndx        ];
    /*load f[x:-,y:+,z:0] from nbr[x:+,y:-,z:0]*/
    fni[15] = ddf[15*n+idx+pdx+ndy    ];
    /*load f[x:0,y:+,z:0] from nbr[x:0,y:-,z:0]*/
    fni[16] = ddf[16*n+idx+    ndy    ];
    /*load f[x:+,y:+,z:0] from nbr[x:-,y:-,z:0]*/
    fni[17] = ddf[17*n+idx+ndx+ndy    ];
                                                
    /*load f[x:-,y:-,z:+] from nbr[x:+,y:+,z:-]*/
    fni[18] = ddf[18*n+idx+pdx+pdy+ndz];
    /*load f[x:0,y:-,z:+] from nbr[x:0,y:+,z:-]*/
    fni[19] = ddf[19*n+idx+    pdy+ndz];
    /*load f[x:+,y:-,z:+] from nbr[x:-,y:+,z:-]*/
    fni[20] = ddf[20*n+idx+ndx+pdy+ndz];
    /*load f[x:-,y:0,z:+] from nbr[x:+,y:0,z:-]*/
    fni[21] = ddf[21*n+idx+pdx+    ndz];
    /*load f[x:0,y:0,z:+] from nbr[x:0,y:0,z:-]*/
    fni[22] = ddf[22*n+idx+        ndz];
    /*load f[x:+,y:0,z:+] from nbr[x:-,y:0,z:-]*/
    fni[23] = ddf[23*n+idx+ndx+    ndz];
    /*load f[x:-,y:+,z:+] from nbr[x:+,y:-,z:-]*/
    fni[24] = ddf[24*n+idx+pdx+ndy+ndz];
    /*load f[x:0,y:+,z:+] from nbr[x:0,y:-,z:-]*/
    fni[25] = ddf[25*n+idx+    ndy+ndz];
    /*load f[x:+,y:+,z:+] from nbr[x:-,y:-,z:-]*/
    fni[26] = ddf[26*n+idx+ndx+ndy+ndz];
}

__device__ __forceinline__ void D3Q27RevPullLoadDDF(Int idx, Int n, Int pdx, Int ndx, Int pdy, Int ndy, Int pdz, Int ndz, const Real *ddf, Real(&fni)[27])
{
    /*load f[x:+,y:+,z:+] from nbr[x:+,y:+,z:+]*/
    fni[26] = ddf[ 0*n+idx+pdx+pdy+pdz];
    /*load f[x:0,y:+,z:+] from nbr[x:0,y:+,z:+]*/
    fni[25] = ddf[ 1*n+idx+    pdy+pdz];
    /*load f[x:-,y:+,z:+] from nbr[x:-,y:+,z:+]*/
    fni[24] = ddf[ 2*n+idx+ndx+pdy+pdz];
    /*load f[x:+,y:0,z:+] from nbr[x:+,y:0,z:+]*/
    fni[23] = ddf[ 3*n+idx+pdx+    pdz];
    /*load f[x:0,y:0,z:+] from nbr[x:0,y:0,z:+]*/
    fni[22] = ddf[ 4*n+idx+        pdz];
    /*load f[x:-,y:0,z:+] from nbr[x:-,y:0,z:+]*/
    fni[21] = ddf[ 5*n+idx+ndx+    pdz];
    /*load f[x:+,y:-,z:+] from nbr[x:+,y:-,z:+]*/
    fni[20] = ddf[ 6*n+idx+pdx+ndy+pdz];
    /*load f[x:0,y:-,z:+] from nbr[x:0,y:-,z:+]*/
    fni[19] = ddf[ 7*n+idx+    ndy+pdz];
    /*load f[x:-,y:-,z:+] from nbr[x:-,y:-,z:+]*/
    fni[18] = ddf[ 8*n+idx+ndx+ndy+pdz];

    /*load f[x:+,y:+,z:0] from nbr[x:+,y:+,z:0]*/
    fni[17] = ddf[ 9*n+idx+pdx+pdy    ];
    /*load f[x:0,y:+,z:0] from nbr[x:0,y:+,z:0]*/
    fni[16] = ddf[10*n+idx+    pdy    ];
    /*load f[x:-,y:+,z:0] from nbr[x:-,y:+,z:0]*/
    fni[15] = ddf[11*n+idx+ndx+pdy    ];
    /*load f[x:+,y:0,z:0] from nbr[x:+,y:0,z:0]*/
    fni[14] = ddf[12*n+idx+pdx        ];
    /*load f[x:0,y:0,z:0] from nbr[x:0,y:0,z:0]*/
    fni[13] = ddf[13*n+idx            ];
    /*load f[x:-,y:0,z:0] from nbr[x:-,y:0,z:0]*/
    fni[12] = ddf[14*n+idx+ndx        ];
    /*load f[x:+,y:-,z:0] from nbr[x:+,y:-,z:0]*/
    fni[11] = ddf[15*n+idx+pdx+ndy    ];
    /*load f[x:0,y:-,z:0] from nbr[x:0,y:-,z:0]*/
    fni[10] = ddf[16*n+idx+    ndy    ];
    /*load f[x:-,y:-,z:0] from nbr[x:-,y:-,z:0]*/
    fni[ 9] = ddf[17*n+idx+ndx+ndy    ];

    /*load f[x:+,y:+,z:-] from nbr[x:+,y:+,z:-]*/
    fni[ 8] = ddf[18*n+idx+pdx+pdy+ndz];
    /*load f[x:0,y:+,z:-] from nbr[x:0,y:+,z:-]*/
    fni[ 7] = ddf[19*n+idx+    pdy+ndz];
    /*load f[x:-,y:+,z:-] from nbr[x:-,y:+,z:-]*/
    fni[ 6] = ddf[20*n+idx+ndx+pdy+ndz];
    /*load f[x:+,y:0,z:-] from nbr[x:+,y:0,z:-]*/
    fni[ 5] = ddf[21*n+idx+pdx+    ndz];
    /*load f[x:0,y:0,z:-] from nbr[x:0,y:0,z:-]*/
    fni[ 4] = ddf[22*n+idx+        ndz];
    /*load f[x:-,y:0,z:-] from nbr[x:-,y:0,z:-]*/
    fni[ 3] = ddf[23*n+idx+ndx+    ndz];
    /*load f[x:+,y:-,z:-] from nbr[x:+,y:-,z:-]*/
    fni[ 2] = ddf[24*n+idx+pdx+ndy+ndz];
    /*load f[x:0,y:-,z:-] from nbr[x:0,y:-,z:-]*/
    fni[ 1] = ddf[25*n+idx+    ndy+ndz];
    /*load f[x:-,y:-,z:-] from nbr[x:-,y:-,z:-]*/
    fni[ 0] = ddf[26*n+idx+ndx+ndy+ndz];
}

__device__ __forceinline__ void D3Q27StoreDDF(Int idx, Int n, Real *ddf, const Real(&fni)[27])
{
    ddf[ 0*n+idx] = fni[ 0];
    ddf[ 1*n+idx] = fni[ 1];
    ddf[ 2*n+idx] = fni[ 2];
    ddf[ 3*n+idx] = fni[ 3];
    ddf[ 4*n+idx] = fni[ 4];
    ddf[ 5*n+idx] = fni[ 5];
    ddf[ 6*n+idx] = fni[ 6];
    ddf[ 7*n+idx] = fni[ 7];
    ddf[ 8*n+idx] = fni[ 8];
    ddf[ 9*n+idx] = fni[ 9];
    ddf[10*n+idx] = fni[10];
    ddf[11*n+idx] = fni[11];
    ddf[12*n+idx] = fni[12];
    ddf[13*n+idx] = fni[13];
    ddf[14*n+idx] = fni[14];
    ddf[15*n+idx] = fni[15];
    ddf[16*n+idx] = fni[16];
    ddf[17*n+idx] = fni[17];
    ddf[18*n+idx] = fni[18];
    ddf[19*n+idx] = fni[19];
    ddf[20*n+idx] = fni[20];
    ddf[21*n+idx] = fni[21];
    ddf[22*n+idx] = fni[22];
    ddf[23*n+idx] = fni[23];
    ddf[24*n+idx] = fni[24];
    ddf[25*n+idx] = fni[25];
    ddf[26*n+idx] = fni[26];
}


__launch_bounds__(D3Q27BlockDDFSimulator::blockX*D3Q27BlockDDFSimulator::blockY*D3Q27BlockDDFSimulator::blockZ) __global__ void D3Q27BlockDDFKernel(const D3Q27BlockDDFKernelParam __grid_constant__ param)
{
    const Int blkIdx = defBlkX+defBlkNx*(defBlkY+defBlkNy*defBlkZ);
    const Int glbX   = param.blkXOff+defBlkX;
    const Int glbY   = param.blkYOff+defBlkY;
    const Int glbZ   = param.blkZOff+defBlkZ;
    const Int glbIdx = glbX+defDomNx*(glbY+defDomNy*glbZ);

    Real rhoi, vxi, vyi, vzi;
    Real fni[27], feqi[27];

    Int pdx =  (glbX==(defDomNx-1)) ? 1-defDomNx :   1;
    Int ndx =  (glbX==0           ) ? defDomNx-1 :  -1;
    Int pdy = ((glbY==(defDomNy-1)) ? 1-defDomNy :   1) * defDomNx;
    Int ndy = ((glbY==0           ) ? defDomNy-1 :  -1) * defDomNx;
    Int pdz = ((glbZ==(defDomNz-1)) ? 1-defDomNz :   1) * defDomNx * defDomNy;
    Int ndz = ((glbZ==0           ) ? defDomNz-1 :  -1) * defDomNx * defDomNy;

    Flag flagi = param.blkFlag[blkIdx];

    if((flagi & D3Q27BlockDDFSimulator::LOAD_DDF_BIT) != 0)
    {
        D3Q27PullLoadDDF(glbIdx, D3Q27BlockDDFSimulator::domSize, pdx, ndx, pdy, ndy, pdz, ndz, param.srcDDF, fni);
    }

    if((flagi & D3Q27BlockDDFSimulator::REV_LOAD_DDF_BIT) != 0)
    {
        D3Q27RevPullLoadDDF(glbIdx, D3Q27BlockDDFSimulator::domSize, pdx, ndx, pdy, ndy, pdz, ndz, param.srcDDF, fni);
    }

    if((flagi & D3Q27BlockDDFSimulator::EQU_DDF_BIT) != 0)
    {
        rhoi = param.rho[glbIdx];
        vxi  = param.vx[glbIdx];
        vyi  = param.vy[glbIdx];
        vzi  = param.vz[glbIdx];
        srt::calcEqu3D<27, Real>(fni, rhoi, vxi, vyi, vzi);
    }

    if((flagi & D3Q27BlockDDFSimulator::COLLIDE_BIT) != 0)
    {
        srt::calcRhoU3D<27, Real>(rhoi, vxi, vyi, vzi, fni);
        srt::calcEqu3D<27, Real>(feqi, rhoi, vxi, vyi, vzi);
        srt::relaxation<27, Real, D3Q27BlockDDFSimulator::invTau>(fni, feqi);
    }

    if constexpr (D3Q27BlockDDFSimulator::blkIter > 1)
    {
        auto thisGrid = cooperative_groups::this_grid();
        pdx =  (defBlkX==(defBlkNx-1)) ? 1-defBlkNx :   1;
        ndx =  (defBlkX==0           ) ? defBlkNx-1 :  -1;
        pdy = ((defBlkY==(defBlkNy-1)) ? 1-defBlkNy :   1) * defBlkNx;
        ndy = ((defBlkY==0           ) ? defBlkNy-1 :  -1) * defBlkNx;
        pdz = ((defBlkZ==(defBlkNz-1)) ? 1-defBlkNz :   1) * defBlkNx * defBlkNy;
        ndz = ((defBlkZ==0           ) ? defBlkNz-1 :  -1) * defBlkNx * defBlkNy;

        #pragma unroll
        for(UInt iter=1 ; iter<D3Q27BlockDDFSimulator::blkIter ; ++iter)
        {
            if((flagi & D3Q27BlockDDFSimulator::STORE_DDF_BIT) != 0)
            {
                D3Q27StoreDDF(blkIdx, D3Q27BlockDDFSimulator::blkSize, (iter % 2 != 0) ?  param.blkOddBuf : param.blkEvenBuf, fni);
            }

            thisGrid.sync();

            if((flagi & D3Q27BlockDDFSimulator::LOAD_DDF_BIT) != 0)
            {
                D3Q27PullLoadDDF(blkIdx, D3Q27BlockDDFSimulator::blkSize, pdx, ndx, pdy, ndy, pdz, ndz, (iter % 2 != 0) ? param.blkOddBuf : param.blkEvenBuf, fni);
            }

            if((flagi & D3Q27BlockDDFSimulator::REV_LOAD_DDF_BIT) != 0)
            {
                D3Q27RevPullLoadDDF(blkIdx, D3Q27BlockDDFSimulator::blkSize, pdx, ndx, pdy, ndy, pdz, ndz, (iter % 2 != 0) ? param.blkOddBuf : param.blkEvenBuf, fni);
            }

            if((flagi & D3Q27BlockDDFSimulator::COLLIDE_BIT) != 0)
            {
                srt::calcRhoU3D<27, Real>(rhoi, vxi, vyi, vzi, fni);
                srt::calcEqu3D<27, Real>(feqi, rhoi, vxi, vyi, vzi);
                srt::relaxation<27, Real, D3Q27BlockDDFSimulator::invTau>(fni, feqi);
            }
        }

        pdx =  (glbX==(defDomNx-1)) ? 1-defDomNx :   1;
        ndx =  (glbX==0           ) ? defDomNx-1 :  -1;
        pdy = ((glbY==(defDomNy-1)) ? 1-defDomNy :   1) * defDomNx;
        ndy = ((glbY==0           ) ? defDomNy-1 :  -1) * defDomNx;
        pdz = ((glbZ==(defDomNz-1)) ? 1-defDomNz :   1) * defDomNx * defDomNy;
        ndz = ((glbZ==0           ) ? defDomNz-1 :  -1) * defDomNx * defDomNy;
    }

    if((flagi & D3Q27BlockDDFSimulator::FINAL_STORE_DDF_FLAG) == D3Q27BlockDDFSimulator::FINAL_STORE_DDF_FLAG)
    {
        D3Q27StoreDDF(glbIdx, D3Q27BlockDDFSimulator::domSize, param.dstDDF, fni);
    }

    if((flagi & D3Q27BlockDDFSimulator::FINAL_STORE_RHO_FLAG) == D3Q27BlockDDFSimulator::FINAL_STORE_RHO_FLAG)
    {
        param.rho[glbIdx] = rhoi;
    }

    if((flagi & D3Q27BlockDDFSimulator::FINAL_STORE_U_FLAG) == D3Q27BlockDDFSimulator::FINAL_STORE_U_FLAG)
    {
        param.vx[glbIdx] = vxi;
        param.vy[glbIdx] = vyi;
        param.vz[glbIdx] = vzi;
    }
}