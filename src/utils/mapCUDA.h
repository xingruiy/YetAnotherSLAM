#pragma once
#include "utils/frame.h"
#include "utils/mapPoint.h"
#include "utils/numType.h"

Mat createAdjacencyMat(
    size_t numPairs,
    Mat descriptorDist,
    Mat srcPointPos,
    Mat dstPointPos);