#include "cu_memory.cuh"
#include "D3Q27DoubleDDFSimulator.cuh"

int main()
{
    try
    {
        constexpr UInt dStep = 10;
        constexpr UInt N = 20;
        D3Q27DoubleDDFSimulator sim;
        std::vector<Real> hVxBuf (D3Q27DoubleDDFSimulator::size, 0);
        sim.setup();
        for(int i=0 ; i<N ; ++i)
        {
            sim.run(dStep);
            std::ofstream out (std::string("frame_")+std::to_string(i)+".dat", std::ios::binary);
            cu::memcpy(hVxBuf.data(), sim.getVxPtr(), D3Q27DoubleDDFSimulator::size);
            out.write((const char*)hVxBuf.data(), sizeof(Real)*hVxBuf.size());
        }
    }
    catch (const std::exception &e)
    {
        std::cerr << e.what() << std::endl;
        return 1;
    }
    return 0;
}