#include "stdafx.hpp"
#include "cu_check.cuh"
#include "D3Q27BlockDDFSimulator.cuh"

int main()
{
    try
    {
        constexpr UInt plotDelta = 50;
        constexpr UInt plotNum = 20;
        //constexpr UInt dStep = D3Q27BlockDDFSimulator::blkIter;
        D3Q27BlockDDFSimulator sim;
        std::vector<Real> hRhoBuf (D3Q27BlockDDFSimulator::domSize, 0);
        std::vector<Real> hVxBuf  (D3Q27BlockDDFSimulator::domSize, 0);
        std::vector<Real> hVyBuf  (D3Q27BlockDDFSimulator::domSize, 0);
        std::vector<Real> hVzBuf  (D3Q27BlockDDFSimulator::domSize, 0);
        sim.setup();
        UInt pltIdx = 0;
        for(UInt i=0 ; i<plotDelta*plotNum ; ++i)
        {
            sim.run();
            if(i % plotDelta == 0)
            {
                std::ofstream rhoOut (std::string("frame_rho_")+std::to_string(pltIdx)+".dat", std::ios::binary);
                std::ofstream vxOut  (std::string("frame_vx_")+std::to_string(pltIdx)+".dat", std::ios::binary);
                std::ofstream vyOut  (std::string("frame_vy_")+std::to_string(pltIdx)+".dat", std::ios::binary);
                std::ofstream vzOut  (std::string("frame_vz_")+std::to_string(pltIdx)+".dat", std::ios::binary);
                cu::memcpy(hRhoBuf.data(), sim.getRhoPtr(), D3Q27BlockDDFSimulator::domSize);
                cu::memcpy(hVxBuf.data(), sim.getVxPtr(), D3Q27BlockDDFSimulator::domSize);
                cu::memcpy(hVyBuf.data(), sim.getVyPtr(), D3Q27BlockDDFSimulator::domSize);
                cu::memcpy(hVzBuf.data(), sim.getVzPtr(), D3Q27BlockDDFSimulator::domSize);
                rhoOut.write((const char*)hRhoBuf.data(), sizeof(Real)*hRhoBuf.size());
                vxOut.write((const char*)hVxBuf.data(), sizeof(Real)*hVxBuf.size());
                vyOut.write((const char*)hVyBuf.data(), sizeof(Real)*hVyBuf.size());
                vzOut.write((const char*)hVzBuf.data(), sizeof(Real)*hVzBuf.size());
                ++pltIdx;
            }
        }
    }
    catch(const std::exception & e)
    {
        std::cerr << e.what() << std::endl;
        exit(1);
    }
    
    return 0;
}