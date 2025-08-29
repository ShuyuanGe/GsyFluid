#include "D3Q27SingleDDFSimulator.cuh"

int main()
{
    try
    {
        D3Q27SingleDDFSimulator sim;
    }
    catch(const std::exception& e)
    {
        std::cerr << e.what() << std::endl;
        exit(1);
    }
    
    return 0;
}