#include "srt_collision.hpp"

constexpr int Q = 27;

__global__ void testKernel(const float rhoi, const float vxi, const float vyi, const float vzi)
{
    float feqi[Q];
    srt::calcEqu3D<Q>(feqi, rhoi, vxi, vyi, vzi);
    float rhoi_, vxi_, vyi_, vzi_;
    srt::calcRhoU3D<Q>(rhoi_, vxi_, vyi_, vzi_, feqi);
     printf("rhoi_ = %.6f, vxi_ = %.6f, vyi_ = %.6f, vzi_ = %.6f\n", rhoi_, vxi_, vyi_, vzi_);
}

int main()
{
    const float rhoi = 1.f;
    const float vxi = 0.05;
    const float vyi = 0.03;
    const float vzi = -0.01;
    testKernel<<<1,1>>>(rhoi, vxi, vyi, vzi);
    cudaStreamSynchronize(0);
    return 0;
}