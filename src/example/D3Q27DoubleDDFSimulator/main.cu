#include "D3Q27DoubleDDFSimulator.cuh"


int main()
{
    try
    {
        constexpr UInt dStep = 10;
        constexpr UInt N = 10;
        D3Q27DoubleDDFSimulator sim;
        sim.setup();
        for(int i=0 ; i<N ; ++i)
        {
            sim.run(dStep);
        }
    }
    catch (const std::exception &e)
    {
        std::cerr << e.what() << std::endl;
        return 1;
    }
    return 0;
}