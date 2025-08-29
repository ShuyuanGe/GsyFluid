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
    Real *flag= nullptr;
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
            fni[ 9] = param.DDF[26*n+idx+pdxy ];
            /*load f[x:0,y:-,z:0] from nbr[x:0,y:+,z:0]*/
            fni[10] = param.DDF[25*n+idx+pdy  ];
            /*load f[x:+,y:-,z:0] from nbr[x:0,y:+,z:0]*/
            fni[11] = param.DDF[24*n+idx+pdy  ];
            /*load f[x:-,y:0,z:0] from nbr[x:+,y:0,z:0]*/
            fni[12] = param.DDF[23*n+idx+pdx  ];
            /*load f[x:0,y:0,z:0] form nbr[x:0,y:0,z:0]*/
            fni[13] = param.DDF[22*n+idx      ];
            /*load f[x:+,y:0,z:0] from nbr[x:0,y:0,z:0]*/
            fni[14] = param.DDF[21*n+idx      ];
            /*load f[x:-,y:+,z:0] from nbr[x:+,y:0,z:0]*/
            fni[15] = param.DDF[20*n+idx+pdx  ];
            /*load f[x:0,y:+,z:0] form nbr[x:0,y:0,z:0]*/
            fni[16] = param.DDF[19*n+idx      ];
            /*load f[x:+,y:+,z:0] from nbr[x:0,y:0,z:0]*/
            fni[17] = param.DDF[18*n+idx      ];

            /*load f[x:-,y:-,z:+] from nbr[x:+,y:+,z:0]*/
            fni[18] = param.DDF[26*n+idx+pdxy ];
            /*load f[x:0,y:-,z:+] from nbr[x:0,y:+,z:0]*/
            fni[19] = param.DDF[25*n+idx+pdy  ];
            /*load f[x:+,y:-,z:+] from nbr[x:0,y:+,z:0]*/
            fni[20] = param.DDF[24*n+idx+pdy  ];
            /*load f[x:-,y:0,z:+] from nbr[x:+,y:0,z:0]*/
            fni[21] = param.DDF[23*n+idx+pdx  ];
            /*load f[x:0,y:0,z:+] form nbr[x:0,y:0,z:0]*/
            fni[22] = param.DDF[22*n+idx      ];
            /*load f[x:+,y:0,z:+] from nbr[x:0,y:0,z:0]*/
            fni[23] = param.DDF[21*n+idx      ];
            /*load f[x:-,y:+,z:+] from nbr[x:+,y:0,z:0]*/
            fni[24] = param.DDF[20*n+idx+pdx  ];
            /*load f[x:0,y:+,z:+] form nbr[x:0,y:0,z:0]*/
            fni[25] = param.DDF[19*n+idx      ];
            /*load f[x:+,y:+,z:+] from nbr[x:0,y:0,z:0]*/
            fni[26] = param.DDF[18*n+idx      ];
        }
        else
        {

        }
    }
}