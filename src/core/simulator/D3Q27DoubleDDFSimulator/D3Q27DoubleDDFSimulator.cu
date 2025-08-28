#include "stdafx.cuh"
#include "cu_check.cuh"
#include "srt_collision.hpp"
#include "D3Q27DoubleDDFSimulator.cuh"

struct D3Q27DoubleDDFKernelParam
{
    Real *srcDDF    = nullptr;
    Real *dstDDF    = nullptr;
    Real *rho       = nullptr;
    Real *vx        = nullptr;
    Real *vy        = nullptr;
    Real *vz        = nullptr;
    Flag *flag      = nullptr;  
};

__global__ void D3Q27DoubleDDFKernel(const D3Q27DoubleDDFKernelParam __grid_constant__ param);



D3Q27DoubleDDFSimulator::D3Q27DoubleDDFSimulator(
    Real dftRho, 
    std::array<Real, 3> dftV
) : 
    FlagBufData(size), 
    DensityBufData(size), 
    VelocityBufData(size), 
    DoubleDDFBufData(size), 
    timeStep(0),
    dftRho(dftRho), 
    dftV(dftV)
{
    cudaStreamCreate(&stream);
    cudaEventCreate(&start);
    cudaEventCreate(&end);

    {
        thrust::device_ptr<Flag> dFlagPtr {getFlagPtr()};
        thrust::fill_n(dFlagPtr, size, FLUID_FLAG);
    }

    {
        thrust::device_ptr<Real> dRhoPtr {getRhoPtr()};
        thrust::fill_n(dRhoPtr, size, dftRho);
    }

    {
        thrust::device_ptr<Real> dVxPtr {getVxPtr()};
        thrust::device_ptr<Real> dVyPtr {getVyPtr()};
        thrust::device_ptr<Real> dVzPtr {getVzPtr()};
        thrust::fill_n(dVxPtr, size, dftV[0]);
        thrust::fill_n(dVyPtr, size, dftV[1]);
        thrust::fill_n(dVzPtr, size, dftV[2]);
    }

    {
        Real dftFeq[27];
        thrust::device_ptr<Real> dSrcDDFPtr {getSrcDDFPtr()};
        thrust::device_ptr<Real> dDstDDFPtr {getDstDDFPtr()};
        srt::calcEqu3D<27, Real>(dftFeq, dftRho, dftV[0], dftV[1], dftV[2]);
        #pragma unroll
        for(Int i=0 ; i<27 ; ++i)
        {
            thrust::fill_n(dSrcDDFPtr+i*size, size, dftFeq[i]);
            thrust::fill_n(dDstDDFPtr+i*size, size, dftFeq[i]);
        }
    }
}

D3Q27DoubleDDFSimulator::~D3Q27DoubleDDFSimulator() noexcept
{
    cudaStreamDestroy(stream);
    cudaEventDestroy(start);
    cudaEventDestroy(end);
}

void D3Q27DoubleDDFSimulator::setup()
{
    thrust::for_each_n(
        thrust::make_counting_iterator<UInt>(0), 
        size, 
        [
            dFlagPtr = getFlagPtr(), 
            dVxPtr = getVxPtr(), 
            dVyPtr = getVyPtr(), 
            dVzPtr = getVzPtr(),
            nx = nx, 
            ny = ny, 
            nz = nz
        ]
        __device__ (UInt idx) -> void
        {
            const UInt x = idx % nx;
            const UInt y = (idx / nx) % ny;
            const UInt z = idx / (nx * ny);
            if(y==0 or y==ny-1 or z==0 or z==nz-1)
            {
                dFlagPtr[idx] = BOUNCE_BACK_BC_FLAG;
            }
            else if (x==0)
            {
                dFlagPtr[idx] = EQU_BC_FLAG;
                dVxPtr[idx]   = 0.01;
            }
            else if (x==nx-1)
            {
                dFlagPtr[idx] = EQU_BC_FLAG;
                dVxPtr[idx] = static_cast<Real>(0);
                dVyPtr[idx] = static_cast<Real>(0);
                dVzPtr[idx] = static_cast<Real>(0);
            }
        }
    );
}

void D3Q27DoubleDDFSimulator::run(UInt step)
{
    float ms;
    D3Q27DoubleDDFKernelParam param
    {
        .srcDDF = nullptr, 
        .dstDDF = nullptr, 
        .rho    = getRhoPtr(), 
        .vx     = getVxPtr(), 
        .vy     = getVyPtr(), 
        .vz     = getVzPtr(), 
        .flag   = getFlagPtr()
    };
    cudaEventRecord(start, stream);
    for(UInt i=0 ; i<step ; ++i)
    {
        param.srcDDF = getSrcDDFPtr();
        param.dstDDF = getDstDDFPtr();
        D3Q27DoubleDDFKernel<<<dim3{ny, nz, 1}, nx, 0, stream>>>(param);
        swapDDFPtr();
        ++timeStep;
    }
    cudaEventRecord(end, stream);
    cudaEventSynchronize(end);
    cudaEventElapsedTime(&ms, start, end);
    cu::check();
    const float mlups = (static_cast<float>(size)*step/(1024*1024)) / (ms / 1000);
    printf("[Info]: D3Q27DoubleDDFSimulator run, speed = %.2f (MLUPS)\n", mlups);
}

#define defX()  static_cast<Int>(threadIdx.x)
#define defY()  static_cast<Int>(blockIdx.x)
#define defZ()  static_cast<Int>(blockIdx.y)
#define defNx() static_cast<Int>(blockDim.x)
#define defNy() static_cast<Int>(gridDim.x)
#define defNz() static_cast<Int>(gridDim.y)

__launch_bounds__(D3Q27DoubleDDFSimulator::nx) __global__ void D3Q27DoubleDDFKernel(const D3Q27DoubleDDFKernelParam __grid_constant__ param)
{
    const Int n         = defNx()*defNy()*defNz();
    const Int idx       = defX()+defNx()*(defY()+defNy()*defZ());
    const Flag flagi    = param.flag[idx];

    //positive delta x
    const Int pdx       = (defX()==(defNx()-1)) ? 1-defNx() : 1;
    //negative delta x
    const Int ndx       = (defX()==0) ? defNx()-1 : -1;
    //positive delta y
    const Int pdy       = (defY()==(defNy()-1)) ? defNx()*(1-defNy()) : defNx();
    //negative delta y
    const Int ndy       = (defY()==0) ? (defNy()-1)*defNx() : -defNx();
    //positive delta z
    const Int pdz       = (defZ()==(defNz()-1)) ? (1-defNz())*defNy()*defNx() : defNy()*defNx();
    //negative delta z
    const Int ndz       = (defZ()==0) ? (defNz()-1)*defNy()*defNx() : -defNy()*defNx();

    Real rhoi, vxi, vyi, vzi;
    Real fni[27];

    if((flagi & D3Q27DoubleDDFSimulator::LOAD_DDF_BIT) != 0)
    {
        /*load f[x:-,y:-,z:-] from nbr[x:+,y:+,z:+]*/
        fni[ 0] = param.srcDDF[ 0*n+idx+pdx+pdy+pdz]; 
        /*load f[x:0,y:-,z:-] from nbr[x:0,y:+,z:+]*/
        fni[ 1] = param.srcDDF[ 1*n+idx+    pdy+pdz]; 
        /*load f[x:+,y:-,z:-] from nbr[x:-,y:+,z:+]*/
        fni[ 2] = param.srcDDF[ 2*n+idx+ndx+pdy+pdz];
        /*load f[x:-,y:0,z:-] from nbr[x:+,y:0,z:+]*/
        fni[ 3] = param.srcDDF[ 3*n+idx+pdx+    pdz];
        /*load f[x:0,y:0,z:-] from nbr[x:0,y:0,z:+]*/
        fni[ 4] = param.srcDDF[ 4*n+idx+        pdz];
        /*load f[x:+,y:0,z:-] from nbr[x:-,y:0,z:+]*/
        fni[ 5] = param.srcDDF[ 5*n+idx+ndx+    pdz];
        /*load f[x:-,y:+,z:-] from nbr[x:+,y:-,z:+]*/
        fni[ 6] = param.srcDDF[ 6*n+idx+pdx+ndy+pdz];
        /*load f[x:0,y:+,z:-] from nbr[x:0,y:-,z:+]*/
        fni[ 7] = param.srcDDF[ 7*n+idx+    ndy+pdz];
        /*load f[x:+,y:+,z:-] from nbr[x:-,y:-,z:+]*/
        fni[ 8] = param.srcDDF[ 8*n+idx+ndx+ndy+pdz];
                                                    
        /*load f[x:-,y:-,z:0] from nbr[x:+,y:+,z:0]*/
        fni[ 9] = param.srcDDF[ 9*n+idx+pdx+pdy    ];
        /*load f[x:0,y:-,z:0] from nbr[x:0,y:+,z:0]*/
        fni[10] = param.srcDDF[10*n+idx+    pdy    ];
        /*load f[x:+,y:-,z:0] from nbr[x:-,y:+,z:0]*/
        fni[11] = param.srcDDF[11*n+idx+ndx+pdy    ];
        /*load f[x:-,y:0,z:0] from nbr[x:+,y:0,z:0]*/
        fni[12] = param.srcDDF[12*n+idx+pdx        ];
        /*load f[x:0,y:0,z:0] from nbr[x:0,y:0,z:0]*/
        fni[13] = param.srcDDF[13*n+idx            ];
        /*load f[x:+,y:0,z:0] from nbr[x:-,y:0,z:0]*/
        fni[14] = param.srcDDF[14*n+idx+ndx        ];
        /*load f[x:-,y:+,z:0] from nbr[x:+,y:-,z:0]*/
        fni[15] = param.srcDDF[15*n+idx+pdx+ndy    ];
        /*load f[x:0,y:+,z:0] from nbr[x:0,y:-,z:0]*/
        fni[16] = param.srcDDF[16*n+idx+    ndy    ];
        /*load f[x:+,y:+,z:0] from nbr[x:-,y:-,z:0]*/
        fni[17] = param.srcDDF[17*n+idx+ndx+ndy    ];
                                                    
        /*load f[x:-,y:-,z:+] from nbr[x:+,y:+,z:-]*/
        fni[18] = param.srcDDF[18*n+idx+pdx+pdy+ndz];
        /*load f[x:0,y:-,z:+] from nbr[x:0,y:+,z:-]*/
        fni[19] = param.srcDDF[19*n+idx+    pdy+ndz];
        /*load f[x:+,y:-,z:+] from nbr[x:-,y:+,z:-]*/
        fni[20] = param.srcDDF[20*n+idx+ndx+pdy+ndz];
        /*load f[x:-,y:0,z:+] from nbr[x:+,y:0,z:-]*/
        fni[21] = param.srcDDF[21*n+idx+pdx+    ndz];
        /*load f[x:0,y:0,z:+] from nbr[x:0,y:0,z:-]*/
        fni[22] = param.srcDDF[22*n+idx+        ndz];
        /*load f[x:+,y:0,z:+] from nbr[x:-,y:0,z:-]*/
        fni[23] = param.srcDDF[23*n+idx+ndx+    ndz];
        /*load f[x:-,y:+,z:+] from nbr[x:+,y:-,z:-]*/
        fni[24] = param.srcDDF[24*n+idx+pdx+ndy+ndz];
        /*load f[x:0,y:+,z:+] from nbr[x:0,y:-,z:-]*/
        fni[25] = param.srcDDF[25*n+idx+    ndy+ndz];
        /*load f[x:+,y:+,z:+] from nbr[x:-,y:-,z:-]*/
        fni[26] = param.srcDDF[26*n+idx+ndx+ndy+ndz];
    }

    if((flagi & D3Q27DoubleDDFSimulator::REV_LOAD_DDF_BIT) != 0)
    {
        /*load f[x:+,y:+,z:+] from nbr[x:+,y:+,z:+]*/
        fni[26] = param.srcDDF[ 0*n+idx+pdx+pdy+pdz];
        /*load f[x:0,y:+,z:+] from nbr[x:0,y:+,z:+]*/
        fni[25] = param.srcDDF[ 1*n+idx+    pdy+pdz];
        /*load f[x:-,y:+,z:+] from nbr[x:-,y:+,z:+]*/
        fni[24] = param.srcDDF[ 2*n+idx+ndx+pdy+pdz];
        /*load f[x:+,y:0,z:+] from nbr[x:+,y:0,z:+]*/
        fni[23] = param.srcDDF[ 3*n+idx+pdx+    pdz];
        /*load f[x:0,y:0,z:+] from nbr[x:0,y:0,z:+]*/
        fni[22] = param.srcDDF[ 4*n+idx+        pdz];
        /*load f[x:-,y:0,z:+] from nbr[x:-,y:0,z:+]*/
        fni[21] = param.srcDDF[ 5*n+idx+ndx+    pdz];
        /*load f[x:+,y:-,z:+] from nbr[x:+,y:-,z:+]*/
        fni[20] = param.srcDDF[ 6*n+idx+pdx+ndy+pdz];
        /*load f[x:0,y:-,z:+] from nbr[x:0,y:-,z:+]*/
        fni[19] = param.srcDDF[ 7*n+idx+    ndy+pdz];
        /*load f[x:-,y:-,z:+] from nbr[x:-,y:-,z:+]*/
        fni[18] = param.srcDDF[ 8*n+idx+ndx+ndy+pdz];

        /*load f[x:+,y:+,z:0] from nbr[x:+,y:+,z:0]*/
        fni[17] = param.srcDDF[ 9*n+idx+pdx+pdy    ];
        /*load f[x:0,y:+,z:0] from nbr[x:0,y:+,z:0]*/
        fni[16] = param.srcDDF[10*n+idx+    pdy    ];
        /*load f[x:-,y:+,z:0] from nbr[x:-,y:+,z:0]*/
        fni[15] = param.srcDDF[11*n+idx+ndx+pdy    ];
        /*load f[x:+,y:0,z:0] from nbr[x:+,y:0,z:0]*/
        fni[14] = param.srcDDF[12*n+idx+pdx        ];
        /*load f[x:0,y:0,z:0] from nbr[x:0,y:0,z:0]*/
        fni[13] = param.srcDDF[13*n+idx            ];
        /*load f[x:-,y:0,z:0] from nbr[x:-,y:0,z:0]*/
        fni[12] = param.srcDDF[14*n+idx+ndx        ];
        /*load f[x:+,y:-,z:0] from nbr[x:+,y:-,z:0]*/
        fni[11] = param.srcDDF[15*n+idx+pdx+ndy    ];
        /*load f[x:0,y:-,z:0] from nbr[x:0,y:-,z:0]*/
        fni[10] = param.srcDDF[16*n+idx+    ndy    ];
        /*load f[x:-,y:-,z:0] from nbr[x:-,y:-,z:0]*/
        fni[ 9] = param.srcDDF[17*n+idx+ndx+ndy    ];

        /*load f[x:+,y:+,z:-] from nbr[x:+,y:+,z:-]*/
        fni[ 8] = param.srcDDF[18*n+idx+pdx+pdy+ndz];
        /*load f[x:0,y:+,z:-] from nbr[x:0,y:+,z:-]*/
        fni[ 7] = param.srcDDF[19*n+idx+    pdy+ndz];
        /*load f[x:-,y:+,z:-] from nbr[x:-,y:+,z:-]*/
        fni[ 6] = param.srcDDF[20*n+idx+ndx+pdy+ndz];
        /*load f[x:+,y:0,z:-] from nbr[x:+,y:0,z:-]*/
        fni[ 5] = param.srcDDF[21*n+idx+pdx+    ndz];
        /*load f[x:0,y:0,z:-] from nbr[x:0,y:0,z:-]*/
        fni[ 4] = param.srcDDF[22*n+idx+        ndz];
        /*load f[x:-,y:0,z:-] from nbr[x:-,y:0,z:-]*/
        fni[ 3] = param.srcDDF[23*n+idx+ndx+    ndz];
        /*load f[x:+,y:-,z:-] from nbr[x:+,y:-,z:-]*/
        fni[ 2] = param.srcDDF[24*n+idx+pdx+ndy+ndz];
        /*load f[x:0,y:-,z:-] from nbr[x:0,y:-,z:-]*/
        fni[ 1] = param.srcDDF[25*n+idx+    ndy+ndz];
        /*load f[x:-,y:-,z:-] from nbr[x:-,y:-,z:-]*/
        fni[ 0] = param.srcDDF[26*n+idx+ndx+ndy+ndz];
    }

    if((flagi & D3Q27DoubleDDFSimulator::EQU_DDF_BIT) != 0)
    {
        rhoi = param.rho[idx];
        vxi  = param.vx[idx];
        vyi  = param.vy[idx];
        vzi  = param.vz[idx];
        srt::calcEqu3D<27, Real>(fni, rhoi, vxi, vyi, vzi);
    }

    if((flagi & D3Q27DoubleDDFSimulator::COLLIDE_BIT) != 0)
    {
        Real feqi[27];
        srt::calcRhoU3D<27, Real>(rhoi, vxi, vyi, vzi, fni);
        srt::calcEqu3D<27, Real>(feqi, rhoi, vxi, vyi, vzi);
        srt::relaxation<27, Real, D3Q27DoubleDDFSimulator::invTau>(fni, feqi);
    }

    if((flagi & D3Q27DoubleDDFSimulator::STORE_DDF_BIT) != 0)
    {
        param.dstDDF[ 0*n+idx] = fni[ 0];
        param.dstDDF[ 1*n+idx] = fni[ 1];
        param.dstDDF[ 2*n+idx] = fni[ 2];
        param.dstDDF[ 3*n+idx] = fni[ 3];
        param.dstDDF[ 4*n+idx] = fni[ 4];
        param.dstDDF[ 5*n+idx] = fni[ 5];
        param.dstDDF[ 6*n+idx] = fni[ 6];
        param.dstDDF[ 7*n+idx] = fni[ 7];
        param.dstDDF[ 8*n+idx] = fni[ 8];
        param.dstDDF[ 9*n+idx] = fni[ 9];
        param.dstDDF[10*n+idx] = fni[10];
        param.dstDDF[11*n+idx] = fni[11];
        param.dstDDF[12*n+idx] = fni[12];
        param.dstDDF[13*n+idx] = fni[13];
        param.dstDDF[14*n+idx] = fni[14];
        param.dstDDF[15*n+idx] = fni[15];
        param.dstDDF[16*n+idx] = fni[16];
        param.dstDDF[17*n+idx] = fni[17];
        param.dstDDF[18*n+idx] = fni[18];
        param.dstDDF[19*n+idx] = fni[19];
        param.dstDDF[20*n+idx] = fni[20];
        param.dstDDF[21*n+idx] = fni[21];
        param.dstDDF[22*n+idx] = fni[22];
        param.dstDDF[23*n+idx] = fni[23];
        param.dstDDF[24*n+idx] = fni[24];
        param.dstDDF[25*n+idx] = fni[25];
        param.dstDDF[26*n+idx] = fni[26];
    }

    if((flagi & D3Q27DoubleDDFSimulator::STORE_RHO_BIT) != 0)
    {
        param.rho[idx] = rhoi;
    }

    if((flagi & D3Q27DoubleDDFSimulator::STORE_U_BIT) != 0)
    {
        param.vx[idx] = vxi;
        param.vy[idx] = vyi;
        param.vz[idx] = vzi;
    }
}