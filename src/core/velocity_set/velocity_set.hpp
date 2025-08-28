#pragma once

#include "stdafx.hpp"

template<int Q>
struct VelocitySet3D
{
    static constexpr int q = Q;
    static constexpr int centIdx = Q / 2;
    static constexpr std::array<int, Q> getDx();
    static constexpr std::array<int, Q> getDy();
    static constexpr std::array<int, Q> getDz();
    static constexpr std::array<float, Q> getSrtOmega();
    static constexpr std::array<int, Q> dx = getDx();
    static constexpr std::array<int, Q> dy = getDy();
    static constexpr std::array<int, Q> dz = getDz();
    static constexpr std::array<float, Q> srtOmega = getSrtOmega();
};

template<>
constexpr std::array<int, 15> VelocitySet3D<15>::getDx() 
{
    return
    {
        -1,      1,
             0,
        -1,      1,
             0,
        -1,  0,  1, 
             0, 
        -1,      1, 
             0,
        -1,      1
    };
}

template<>
constexpr std::array<int, 15> VelocitySet3D<15>::getDy() 
{
    return
    {
        -1,     -1,
             0,
         1,      1,
            -1,
         0,  0,  0, 
             1, 
        -1,     -1, 
             0,
         1,      1
    };
}

template<>
constexpr std::array<int, 15> VelocitySet3D<15>::getDz()
{
    return
    {
        -1,     -1,
            -1,
        -1,     -1,
             0,
         0,  0,  0, 
             0, 
         1,      1, 
             1,
         1,      1 
    };
}

template<>
constexpr std::array<int, 19> VelocitySet3D<19>::getDx() 
{
    return
    {
             0,
        -1,  0,  1, 
             0, 
        -1,  0,  1, 
        -1,  0,  1, 
        -1,  0,  1, 
             0, 
        -1,  0,  1, 
             0
    };
}

template<>
constexpr std::array<int, 19> VelocitySet3D<19>::getDy() 
{
    return
    {
            -1,
         0,  0,  0,  
             1,
        -1, -1, -1,  
         0,  0,  0,  
         1,  1,  1,
            -1, 
         0,  0,  0,  
             1
    };
}

template<>
constexpr std::array<int, 19> VelocitySet3D<19>::getDz()
{
    return
    {
            -1,
        -1, -1, -1, 
            -1,
         0,  0,  0,  
         0,  0,  0,  
         0,  0,  0, 
             1,
         1,  1,  1,  
             1
    };
}

template<>
constexpr std::array<int, 27> VelocitySet3D<27>::getDx()
{
    return
    {
        -1,  0,  1, 
        -1,  0,  1, 
        -1,  0,  1, 
        -1,  0,  1, 
        -1,  0,  1, 
        -1,  0,  1, 
        -1,  0,  1, 
        -1,  0,  1, 
        -1,  0,  1
    };
}

template<>
constexpr std::array<int, 27> VelocitySet3D<27>::getDy()
{
    return
    {
        -1, -1, -1,  
         0,  0,  0,  
         1,  1,  1,
        -1, -1, -1,  
         0,  0,  0,  
         1,  1,  1,
        -1, -1, -1,  
         0,  0,  0,  
         1,  1,  1
    };
}

template<>
constexpr std::array<int, 27> VelocitySet3D<27>::getDz()
{
    return
    {
        -1, -1, -1, 
        -1, -1, -1, 
        -1, -1, -1, 
         0,  0,  0,  
         0,  0,  0,  
         0,  0,  0, 
         1,  1,  1,  
         1,  1,  1,  
         1,  1,  1
    };
}

template<>
constexpr std::array<float, 15> VelocitySet3D<15>::getSrtOmega()
{
    return
    {
        1.f/72,        1.f/72, 
                1.f/9,
        1.f/72,        1.f/72, 
                1.f/9,
        1.f/9,  2.f/9, 1.f/9, 
                1.f/9, 
        1.f/72,        1.f/72, 
                1.f/9,
        1.f/72,        1.f/72
    };
}

template<>
constexpr std::array<float, 19> VelocitySet3D<19>::getSrtOmega()
{
    return
    {
                1.f/36, 
        1.f/36, 1.f/18, 1.f/36, 
                1.f/36,
        1.f/36, 1.f/18, 1.f/36,
        1.f/18, 1.f/3 , 1.f/18, 
        1.f/36, 1.f/18, 1.f/36, 
                1.f/36, 
        1.f/36, 1.f/18, 1.f/36, 
                1.f/36
    };
}

template<>
constexpr std::array<float, 27> VelocitySet3D<27>::getSrtOmega()
{
    return
    {
        1.f/216, 1.f/54 , 1.f/216, 
        1.f/54 , 2.f/27 , 1.f/54 ,
        1.f/216, 1.f/54 , 1.f/216,
        1.f/54 , 2.f/27 , 1.f/54 ,
        2.f/27 , 8.f/27 , 2.f/27 , 
        1.f/54 , 2.f/27 , 1.f/54 ,
        1.f/216, 1.f/54 , 1.f/216, 
        1.f/54 , 2.f/27 , 1.f/54 ,
        1.f/216, 1.f/54 , 1.f/216
    };
}