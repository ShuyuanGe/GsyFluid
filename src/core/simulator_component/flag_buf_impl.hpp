#pragma once

#include "stdafx.hpp"

template<typename DeriveData, typename flag_t>
class FlagBufImpl
{
    protected:
        FlagBufImpl() = default;
    public:
        flag_t *getFlagPtr()
        {
            return static_cast<DeriveData*>(this)->flagPtr;
        }
};