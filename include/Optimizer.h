#pragma once
#include "Map.h"
#include "MapPoint.h"
#include "KeyFrame.h"

class Optimizer
{
public:
    void static BundleAdjustment(const std::vector<KeyFrame *> &vpKF, const std::vector<MapPoint *> &vpMP,
                                 int nIterations = 5, bool *pbStopFlag = NULL, const unsigned long nLoopKF = 0,
                                 const bool bRobust = true);

    void static GlobalBundleAdjustemnt(Map *pMap, int nIterations = 5, bool *pbStopFlag = NULL,
                                       const unsigned long nLoopKF = 0, const bool bRobust = true);

    void static LocalBundleAdjustment(KeyFrame *pKF, bool *pbStopFlag, Map *pMap);

    int static PoseOptimization(Frame *pFrame);
};