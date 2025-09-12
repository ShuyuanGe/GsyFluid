#include "stdafx.hpp"
#include "cu_check.cuh"
#include "D3Q27BlockDDFSimulator.cuh"

int main()
{
    try
    {
        constexpr UInt N = 20;
        constexpr UInt dStep = D3Q27BlockDDFSimulator::blkIter;
        D3Q27BlockDDFSimulator sim;
        std::vector<Real> hVxBuf (D3Q27BlockDDFSimulator::domSize, 0);
        sim.setup();
        for(UInt i=0 ; i<N ; ++i)
        {
            sim.run();
            std::ofstream out (std::string("frame_")+std::to_string(i*dStep)+".dat", std::ios::binary);
            cu::memcpy(hVxBuf.data(), sim.getVxPtr(), D3Q27BlockDDFSimulator::domSize);
            out.write((const char*)hVxBuf.data(), sizeof(Real)*hVxBuf.size());
        }
    }
    catch(const std::exception & e)
    {
        std::cerr << e.what() << std::endl;
        exit(1);
    }
    
    return 0;
}