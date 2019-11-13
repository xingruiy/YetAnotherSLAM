#pragma once
#include "dataStruct/frame.h"
#include "dataStruct/mapPoint.h"
#include "utils/numType.h"

void createAdjacencyMat(
    const size_t numPairs,
    const Mat descriptorDist,
    const Mat srcPointPos,
    const Mat dstPointPos,
    const Mat validPairPt,
    Mat &adjacentMat);

void createAdjacencyMatWithNormal(
    const size_t numPairs,
    const Mat descriptorDist,
    const Mat srcPointPos,
    const Mat dstPointPos,
    const Mat srcPtNormal,
    const Mat dstPtNormal,
    const Mat validPairPt,
    Mat &adjacentMat);