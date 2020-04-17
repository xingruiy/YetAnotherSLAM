#ifndef _GLOBAL_PARAMS_H
#define _GLOBAL_PARAMS_H

#include <string>
#include "Intrinsics.h"

struct GlobalParams
{
    GlobalParams(const std::string filename);

    Intrinsics K;
    int coarsestLvlForTracking = 5;
    std::vector<int> maxIterForTracking = {10, 10, 10, 10, 10};
};

#endif