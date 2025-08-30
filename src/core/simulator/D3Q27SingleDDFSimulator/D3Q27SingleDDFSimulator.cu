#include "stdafx.cuh"
#include "cu_check.cuh"
#include "srt_collision.hpp"
#include "D3Q27SingleDDFSimulator.cuh"

struct D3Q27SingleDDFKernelParam
{
    Real *DDF = nullptr;
    Real *rho = nullptr;
    Real *vx  = nullptr;
    Real *vy  = nullptr;
    Real *vz  = nullptr;
    Flag *flag= nullptr;
};

template<bool isEven>
__global__ void D3Q27SingleDDFKernel(const D3Q27SingleDDFKernelParam __grid_constant__ param);



D3Q27SingleDDFSimulator::D3Q27SingleDDFSimulator(
    Real dftRho, 
    std::array<Real, 3> dftV
) : 
    FlagBufData(size), 
    DensityBufData(size), 
    VelocityBufData(size), 
    SingleDDFBufData(size),
    timeStep(0), 
    dftRho(dftRho), 
    dftV(dftV)
{
    cudaStreamCreate(&stream);
    cudaEventCreate(&start);
    cudaEventCreate(&end);

    setDftFlag(FLUID_FLAG, size);

    setDftRho(dftRho, size);

    setDftVx(dftV[0], size);
    setDftVy(dftV[1], size);
    setDftVz(dftV[2], size);

    Real dftFeq[27];
    srt::calcEqu3D<27, Real>(dftFeq, dftRho, dftV[0], dftV[1], dftV[2]);
    setDftDDF<27>(dftFeq, size);
}

D3Q27SingleDDFSimulator::~D3Q27SingleDDFSimulator() noexcept
{
    cudaStreamDestroy(stream);
    cudaEventDestroy(start);
    cudaEventDestroy(end);
}

void D3Q27SingleDDFSimulator::setup()
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
            const UInt x = idx%nx;
            const UInt y = (idx/nx)%ny;
            const UInt z = idx/(nx*ny);
            if(y==0 or y==ny-1 or z==0 or z==nz-1)
            {
                dFlagPtr[idx] = BOUNCE_BACK_BC_FLAG;
            }
            else if (x==0)
            {
                dFlagPtr[idx] = EQU_BC_FLAG;
                dVxPtr[idx] = 0.01;
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

void D3Q27SingleDDFSimulator::run(UInt step)
{
    float ms;
    D3Q27SingleDDFKernelParam param
    {
        .DDF = getDDFPtr(), 
        .rho = getRhoPtr(), 
        .vx  = getVxPtr(), 
        .vy  = getVyPtr(), 
        .vz  = getVzPtr(), 
        .flag= getFlagPtr()
    };
    cudaEventRecord(start, stream);
    for(UInt i=0 ; i<step ; ++i)
    {
        if(timeStep%2==0)
        {
            D3Q27SingleDDFKernel<true><<<dim3{ny, nz, 1}, nx, 0, stream>>>(param);
        }
        else
        {
            D3Q27SingleDDFKernel<false><<<dim3{ny, nz, 1}, nx, 0, stream>>>(param);
        }
        ++timeStep;
    }
    cudaEventRecord(end, stream);
    cudaEventSynchronize(end);
    cudaEventElapsedTime(&ms, start, end);
    cu::check();
    const float mlups = (static_cast<float>(size)*step/(1024*1024) / (ms/1000));
    printf("[Info: D3Q27SingleDDFSimulator run, speed = %.2f (MLUPS)]\n", mlups);
}

#define defX()  static_cast<Int>(threadIdx.x)
#define defY()  static_cast<Int>(blockIdx.x)
#define defZ()  static_cast<Int>(blockIdx.y)
#define defNx() static_cast<Int>(blockDim.x)
#define defNy() static_cast<Int>(gridDim.x)
#define defNz() static_cast<Int>(gridDim.y)

template<bool isEven>
__launch_bounds__(D3Q27SingleDDFSimulator::nx) __global__ void D3Q27SingleDDFKernel(const D3Q27SingleDDFKernelParam __grid_constant__ param)
{
    const Int n         = defNx()*defNy()*defNz();
    const Int idx       = defX()+defNx()*(defY()+defNy()*defZ());
    const Flag flagi    = param.flag[idx];

    //positive delta x
    const Int pdx       = (defX()==(defNx()-1)) ? 1-defNx() : 1;
    //positive delta y
    const Int pdy       = (defY()==(defNy()-1)) ? defNx()*(1-defNy()) : defNx();
    //positive delta z
    const Int pdz       = (defZ()==(defNz()-1)) ? (1-defNz())*defNy()*defNx() : defNy()*defNx();

    const Int pdxy = pdx+pdy;
    const Int pdxz = pdx+pdz;
    const Int pdyz = pdy+pdz;
    const Int pdxyz= pdxy+pdz;

    Real rhoi, vxi, vyi, vzi;
    Real fni[27];

    if((flagi & D3Q27SingleDDFSimulator::LOAD_DDF_BIT) != 0)
    {
        if constexpr (isEven)
        {
            /*load f[x:-,y:-,z:-] from nbr[x:+,y:+,z:+]*/
            fni[ 0] = param.DDF[26*n+idx+pdxyz];
            /*load f[x:0,y:-,z:-] from nbr[x:0,y:+,z:+]*/
            fni[ 1] = param.DDF[25*n+idx+pdyz ];
            /*load f[x:+,y:-,z:-] from nbr[x:0,y:+,z:+]*/
            fni[ 2] = param.DDF[24*n+idx+pdyz ];
            /*load f[x:-,y:0,z:-] from nbr[x:+,y:0,z:+]*/
            fni[ 3] = param.DDF[23*n+idx+pdxz ];
            /*load f[x:0,y:0,z:-] form nbr[x:0,y:0,z:+]*/
            fni[ 4] = param.DDF[22*n+idx+pdz  ];
            /*load f[x:+,y:0,z:-] from nbr[x:0,y:0,z:+]*/
            fni[ 5] = param.DDF[21*n+idx+pdz  ];
            /*load f[x:-,y:+,z:-] from nbr[x:+,y:0,z:+]*/
            fni[ 6] = param.DDF[20*n+idx+pdxz ];
            /*load f[x:0,y:+,z:-] form nbr[x:0,y:0,z:+]*/
            fni[ 7] = param.DDF[19*n+idx+pdz  ];
            /*load f[x:+,y:+,z:-] from nbr[x:0,y:0,z:+]*/
            fni[ 8] = param.DDF[18*n+idx+pdz  ];

            /*load f[x:-,y:-,z:0] from nbr[x:+,y:+,z:0]*/
            fni[ 9] = param.DDF[17*n+idx+pdxy ];
            /*load f[x:0,y:-,z:0] from nbr[x:0,y:+,z:0]*/
            fni[10] = param.DDF[16*n+idx+pdy  ];
            /*load f[x:+,y:-,z:0] from nbr[x:0,y:+,z:0]*/
            fni[11] = param.DDF[15*n+idx+pdy  ];
            /*load f[x:-,y:0,z:0] from nbr[x:+,y:0,z:0]*/
            fni[12] = param.DDF[14*n+idx+pdx  ];
            /*load f[x:0,y:0,z:0] form nbr[x:0,y:0,z:0]*/
            fni[13] = param.DDF[13*n+idx      ];
            /*load f[x:+,y:0,z:0] from nbr[x:0,y:0,z:0]*/
            fni[14] = param.DDF[12*n+idx      ];
            /*load f[x:-,y:+,z:0] from nbr[x:+,y:0,z:0]*/
            fni[15] = param.DDF[11*n+idx+pdx  ];
            /*load f[x:0,y:+,z:0] form nbr[x:0,y:0,z:0]*/
            fni[16] = param.DDF[10*n+idx      ];
            /*load f[x:+,y:+,z:0] from nbr[x:0,y:0,z:0]*/
            fni[17] = param.DDF[ 9*n+idx      ];

            /*load f[x:-,y:-,z:+] from nbr[x:+,y:+,z:0]*/
            fni[18] = param.DDF[ 8*n+idx+pdxy ];
            /*load f[x:0,y:-,z:+] from nbr[x:0,y:+,z:0]*/
            fni[19] = param.DDF[ 7*n+idx+pdy  ];
            /*load f[x:+,y:-,z:+] from nbr[x:0,y:+,z:0]*/
            fni[20] = param.DDF[ 6*n+idx+pdy  ];
            /*load f[x:-,y:0,z:+] from nbr[x:+,y:0,z:0]*/
            fni[21] = param.DDF[ 5*n+idx+pdx  ];
            /*load f[x:0,y:0,z:+] form nbr[x:0,y:0,z:0]*/
            fni[22] = param.DDF[ 4*n+idx      ];
            /*load f[x:+,y:0,z:+] from nbr[x:0,y:0,z:0]*/
            fni[23] = param.DDF[ 3*n+idx      ];
            /*load f[x:-,y:+,z:+] from nbr[x:+,y:0,z:0]*/
            fni[24] = param.DDF[ 2*n+idx+pdx  ];
            /*load f[x:0,y:+,z:+] form nbr[x:0,y:0,z:0]*/
            fni[25] = param.DDF[ 1*n+idx      ];
            /*load f[x:+,y:+,z:+] from nbr[x:0,y:0,z:0]*/
            fni[26] = param.DDF[ 0*n+idx      ];
        }
        else
        {
            /*load f[x:-,y:-,z:-] from nbr[x:+,y:+,z:+]*/
            fni[ 0] = param.DDF[ 0*n+idx+pdxyz];
            /*load f[x:0,y:-,z:-] from nbr[x:0,y:+,z:+]*/
            fni[ 1] = param.DDF[ 1*n+idx+pdyz ];
            /*load f[x:+,y:-,z:-] from nbr[x:0,y:+,z:+]*/
            fni[ 2] = param.DDF[ 2*n+idx+pdyz ];
            /*load f[x:-,y:0,z:-] from nbr[x:+,y:0,z:+]*/
            fni[ 3] = param.DDF[ 3*n+idx+pdxz ];
            /*load f[x:0,y:0,z:-] form nbr[x:0,y:0,z:+]*/
            fni[ 4] = param.DDF[ 4*n+idx+pdz  ];
            /*load f[x:+,y:0,z:-] from nbr[x:0,y:0,z:+]*/
            fni[ 5] = param.DDF[ 5*n+idx+pdz  ];
            /*load f[x:-,y:+,z:-] from nbr[x:+,y:0,z:+]*/
            fni[ 6] = param.DDF[ 6*n+idx+pdxz ];
            /*load f[x:0,y:+,z:-] form nbr[x:0,y:0,z:+]*/
            fni[ 7] = param.DDF[ 7*n+idx+pdz  ];
            /*load f[x:+,y:+,z:-] from nbr[x:0,y:0,z:+]*/
            fni[ 8] = param.DDF[ 8*n+idx+pdz  ];

            /*load f[x:-,y:-,z:0] from nbr[x:+,y:+,z:0]*/
            fni[ 9] = param.DDF[ 9*n+idx+pdxy ];
            /*load f[x:0,y:-,z:0] from nbr[x:0,y:+,z:0]*/
            fni[10] = param.DDF[10*n+idx+pdy  ];
            /*load f[x:+,y:-,z:0] from nbr[x:0,y:+,z:0]*/
            fni[11] = param.DDF[11*n+idx+pdy  ];
            /*load f[x:-,y:0,z:0] from nbr[x:+,y:0,z:0]*/
            fni[12] = param.DDF[12*n+idx+pdx  ];
            /*load f[x:0,y:0,z:0] form nbr[x:0,y:0,z:0]*/
            fni[13] = param.DDF[13*n+idx      ];
            /*load f[x:+,y:0,z:0] from nbr[x:0,y:0,z:0]*/
            fni[14] = param.DDF[14*n+idx      ];
            /*load f[x:-,y:+,z:0] from nbr[x:+,y:0,z:0]*/
            fni[15] = param.DDF[15*n+idx+pdx  ];
            /*load f[x:0,y:+,z:0] form nbr[x:0,y:0,z:0]*/
            fni[16] = param.DDF[16*n+idx      ];
            /*load f[x:+,y:+,z:0] from nbr[x:0,y:0,z:0]*/
            fni[17] = param.DDF[17*n+idx      ];

            /*load f[x:-,y:-,z:+] from nbr[x:+,y:+,z:0]*/
            fni[18] = param.DDF[18*n+idx+pdxy ];
            /*load f[x:0,y:-,z:+] from nbr[x:0,y:+,z:0]*/
            fni[19] = param.DDF[19*n+idx+pdy  ];
            /*load f[x:+,y:-,z:+] from nbr[x:0,y:+,z:0]*/
            fni[20] = param.DDF[20*n+idx+pdy  ];
            /*load f[x:-,y:0,z:+] from nbr[x:+,y:0,z:0]*/
            fni[21] = param.DDF[21*n+idx+pdx  ];
            /*load f[x:0,y:0,z:+] form nbr[x:0,y:0,z:0]*/
            fni[22] = param.DDF[22*n+idx      ];
            /*load f[x:+,y:0,z:+] from nbr[x:0,y:0,z:0]*/
            fni[23] = param.DDF[23*n+idx      ];
            /*load f[x:-,y:+,z:+] from nbr[x:+,y:0,z:0]*/
            fni[24] = param.DDF[24*n+idx+pdx  ];
            /*load f[x:0,y:+,z:+] form nbr[x:0,y:0,z:0]*/
            fni[25] = param.DDF[25*n+idx      ];
            /*load f[x:+,y:+,z:+] from nbr[x:0,y:0,z:0]*/
            fni[26] = param.DDF[26*n+idx      ];
        }
    }

    if((flagi & D3Q27SingleDDFSimulator::EQU_DDF_BIT) != 0)
    {
        rhoi = param.rho[idx];
        vxi  = param.vx[idx];
        vyi  = param.vy[idx];
        vzi  = param.vz[idx];
        srt::calcEqu3D<27, Real>(fni, rhoi, vxi, vyi, vzi);
    }

    if((flagi & D3Q27SingleDDFSimulator::COLLIDE_BIT) != 0)
    {
        Real feqi[27];
        srt::calcRhoU3D<27, Real>(rhoi, vxi, vyi, vzi, feqi);
        srt::calcEqu3D<27, Real>(feqi, rhoi, vxi, vyi, vzi);
        srt::relaxation<27, Real, D3Q27SingleDDFSimulator::invTau>(fni, feqi);
    }

    if((flagi & D3Q27SingleDDFSimulator::STORE_DDF_BIT) != 0)
    {
        if constexpr (isEven)
        {
            /*store f[x:-,y:-,z:-] from nbr[x:0,y:0,z:0]*/
            param.DDF[ 0*n+idx      ] = fni[ 0];
            /*store f[x:0,y:-,z:-] from nbr[x:0,y:0,z:0]*/
            param.DDF[ 1*n+idx      ] = fni[ 1];
            /*store f[x:+,y:-,z:-] from nbr[x:+,y:0,z:0]*/
            param.DDF[ 2*n+idx+pdx  ] = fni[ 2];
            /*store f[x:-,y:0,z:-] from nbr[x:0,y:0,z:0]*/
            param.DDF[ 3*n+idx      ] = fni[ 3];
            /*store f[x:0,y:0,z:-] from nbr[x:0,y:0,z:0]*/
            param.DDF[ 4*n+idx      ] = fni[ 4];
            /*store f[x:+,y:0,z:-] from nbr[x:+,y:0,z:0]*/
            param.DDF[ 5*n+idx+pdx  ] = fni[ 5];
            /*store f[x:-,y:+,z:-] from nbr[x:0,y:+,z:0]*/
            param.DDF[ 6*n+idx+pdy  ] = fni[ 6];
            /*store f[x:0,y:+,z:-] from nbr[x:0,y:+,z:0]*/
            param.DDF[ 7*n+idx+pdy  ] = fni[ 7];
            /*store f[x:+,y:+,z:-] from nbr[x:+,y:+,z:0]*/
            param.DDF[ 8*n+idx+pdxy ] = fni[ 8];

            /*store f[x:-,y:-,z:0] from nbr[x:0,y:0,z:0]*/
            param.DDF[ 9*n+idx      ] = fni[ 9];
            /*store f[x:0,y:-,z:0] from nbr[x:0,y:0,z:0]*/
            param.DDF[10*n+idx      ] = fni[10];
            /*store f[x:+,y:-,z:0] from nbr[x:+,y:0,z:0]*/
            param.DDF[11*n+idx+pdx  ] = fni[11];
            /*store f[x:-,y:0,z:0] from nbr[x:0,y:0,z:0]*/
            param.DDF[12*n+idx      ] = fni[12];
            /*store f[x:0,y:0,z:0] from nbr[x:0,y:0,z:0]*/
            param.DDF[13*n+idx      ] = fni[13];
            /*store f[x:+,y:0,z:0] from nbr[x:+,y:0,z:0]*/
            param.DDF[14*n+idx+pdx  ] = fni[14];
            /*store f[x:-,y:+,z:0] from nbr[x:0,y:+,z:0]*/
            param.DDF[15*n+idx+pdy  ] = fni[15];
            /*store f[x:0,y:+,z:0] from nbr[x:0,y:+,z:0]*/
            param.DDF[16*n+idx+pdy  ] = fni[16];
            /*store f[x:+,y:+,z:0] from nbr[x:+,y:+,z:0]*/
            param.DDF[17*n+idx+pdxy ] = fni[17];

            /*store f[x:-,y:-,z:+] from nbr[x:0,y:0,z:+]*/
            param.DDF[18*n+idx+pdz  ] = fni[18];
            /*store f[x:0,y:-,z:+] from nbr[x:0,y:0,z:+]*/
            param.DDF[19*n+idx+pdz  ] = fni[19];
            /*store f[x:+,y:-,z:+] from nbr[x:+,y:0,z:+]*/
            param.DDF[20*n+idx+pdxz ] = fni[20];
            /*store f[x:-,y:0,z:+] from nbr[x:0,y:0,z:+]*/
            param.DDF[21*n+idx+pdz  ] = fni[21];
            /*store f[x:0,y:0,z:+] from nbr[x:0,y:0,z:+]*/
            param.DDF[22*n+idx+pdz  ] = fni[22];
            /*store f[x:+,y:0,z:+] from nbr[x:+,y:0,z:+]*/
            param.DDF[23*n+idx+pdxz ] = fni[23];
            /*store f[x:-,y:+,z:+] from nbr[x:0,y:+,z:+]*/
            param.DDF[24*n+idx+pdyz ] = fni[24];
            /*store f[x:0,y:+,z:+] from nbr[x:0,y:+,z:+]*/
            param.DDF[25*n+idx+pdyz ] = fni[25];
            /*store f[x:+,y:+,z:+] from nbr[x:+,y:+,z:+]*/
            param.DDF[26*n+idx+pdxyz] = fni[26];
        }
        else
        {
            /*store f[x:-,y:-,z:-] from nbr[x:0,y:0,z:0]*/
            param.DDF[26*n+idx      ] = fni[ 0];
            /*store f[x:0,y:-,z:-] from nbr[x:0,y:0,z:0]*/
            param.DDF[25*n+idx      ] = fni[ 1];
            /*store f[x:+,y:-,z:-] from nbr[x:+,y:0,z:0]*/
            param.DDF[24*n+idx+pdx  ] = fni[ 2];
            /*store f[x:-,y:0,z:-] from nbr[x:0,y:0,z:0]*/
            param.DDF[23*n+idx      ] = fni[ 3];
            /*store f[x:0,y:0,z:-] from nbr[x:0,y:0,z:0]*/
            param.DDF[22*n+idx      ] = fni[ 4];
            /*store f[x:+,y:0,z:-] from nbr[x:+,y:0,z:0]*/
            param.DDF[21*n+idx+pdx  ] = fni[ 5];
            /*store f[x:-,y:+,z:-] from nbr[x:0,y:+,z:0]*/
            param.DDF[20*n+idx+pdy  ] = fni[ 6];
            /*store f[x:0,y:+,z:-] from nbr[x:0,y:+,z:0]*/
            param.DDF[19*n+idx+pdy  ] = fni[ 7];
            /*store f[x:+,y:+,z:-] from nbr[x:+,y:+,z:0]*/
            param.DDF[18*n+idx+pdxy ] = fni[ 8];

            /*store f[x:-,y:-,z:0] from nbr[x:0,y:0,z:0]*/
            param.DDF[17*n+idx      ] = fni[ 9];
            /*store f[x:0,y:-,z:0] from nbr[x:0,y:0,z:0]*/
            param.DDF[16*n+idx      ] = fni[10];
            /*store f[x:+,y:-,z:0] from nbr[x:+,y:0,z:0]*/
            param.DDF[15*n+idx+pdx  ] = fni[11];
            /*store f[x:-,y:0,z:0] from nbr[x:0,y:0,z:0]*/
            param.DDF[14*n+idx      ] = fni[12];
            /*store f[x:0,y:0,z:0] from nbr[x:0,y:0,z:0]*/
            param.DDF[13*n+idx      ] = fni[13];
            /*store f[x:+,y:0,z:0] from nbr[x:+,y:0,z:0]*/
            param.DDF[12*n+idx+pdx  ] = fni[14];
            /*store f[x:-,y:+,z:0] from nbr[x:0,y:+,z:0]*/
            param.DDF[11*n+idx+pdy  ] = fni[15];
            /*store f[x:0,y:+,z:0] from nbr[x:0,y:+,z:0]*/
            param.DDF[10*n+idx+pdy  ] = fni[16];
            /*store f[x:+,y:+,z:0] from nbr[x:+,y:+,z:0]*/
            param.DDF[ 9*n+idx+pdxy ] = fni[17];
            
            /*store f[x:-,y:-,z:+] from nbr[x:0,y:0,z:+]*/
            param.DDF[ 8*n+idx+pdz  ] = fni[18];
            /*store f[x:0,y:-,z:+] from nbr[x:0,y:0,z:+]*/
            param.DDF[ 7*n+idx+pdz  ] = fni[19];
            /*store f[x:+,y:-,z:+] from nbr[x:+,y:0,z:+]*/
            param.DDF[ 6*n+idx+pdxz ] = fni[20];
            /*store f[x:-,y:0,z:+] from nbr[x:0,y:0,z:+]*/
            param.DDF[ 5*n+idx+pdz  ] = fni[21];
            /*store f[x:0,y:0,z:+] from nbr[x:0,y:0,z:+]*/
            param.DDF[ 4*n+idx+pdz  ] = fni[22];
            /*store f[x:+,y:0,z:+] from nbr[x:+,y:0,z:+]*/
            param.DDF[ 3*n+idx+pdxz ] = fni[23];
            /*store f[x:-,y:+,z:+] from nbr[x:0,y:+,z:+]*/
            param.DDF[ 2*n+idx+pdyz ] = fni[24];
            /*store f[x:0,y:+,z:+] from nbr[x:0,y:+,z:+]*/
            param.DDF[ 1*n+idx+pdyz ] = fni[25];
            /*store f[x:+,y:+,z:+] from nbr[x:+,y:+,z:+]*/
            param.DDF[ 0*n+idx+pdxyz] = fni[26];       
        }
    }

    if((flagi & D3Q27SingleDDFSimulator::STORE_RHO_BIT) != 0)
    {
        param.rho[idx] = rhoi;
    }

    if((flagi & D3Q27SingleDDFSimulator::STORE_U_BIT) != 0)
    {
        param.vx[idx] = vxi;
        param.vy[idx] = vyi;
        param.vz[idx] = vzi;
    }
}