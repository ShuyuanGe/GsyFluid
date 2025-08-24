#include "stdafx.hpp"
#include "invalid_free.hpp"

int main() {
    try 
    {
        raise_invalid_free();
    }
    catch(const std::exception &e)
    {
        std::cerr << e.what() << std::endl;
    }
}