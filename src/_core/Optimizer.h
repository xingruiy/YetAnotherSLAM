#ifndef OPTIMIZER_H
#define OPTIMIZER_H

#include "Map.h"
#include "MapPoint.h"
#include "KeyFrame.h"
#include "LoopClosing.h"
#include <g2o/types/sim3/types_seven_dof_expmap.h>

namespace slam
{

class LoopClosing;

class Optimizer
{
public:
    void static BundleAdjustment(const std::vector<KeyFrame *> &vpKF, const std::vector<MapPoint *> &vpMP,
                                 int nIterations = 5, bool *pbStopFlag = nullptr, const unsigned long nLoopKF = 0,
                                 const bool bRobust = true);

    void static GlobalBundleAdjustemnt(Map *mpMap, int nIterations = 5, bool *pbStopFlag = nullptr,
                                       const unsigned long nLoopKF = 0, const bool bRobust = true);

    void static LocalBundleAdjustment(KeyFrame *pKF, bool *pbStopFlag, Map *mpMap);

    int static PoseOptimization(Frame &pKF);

    void static OptimizeEssentialGraph(Map *mpMap, KeyFrame *pLoopKF, KeyFrame *pCurKF,
                                       const LoopClosing::KeyFrameAndPose &NonCorrectedSim3,
                                       const LoopClosing::KeyFrameAndPose &CorrectedSim3,
                                       const std::map<KeyFrame *, std::set<KeyFrame *>> &LoopConnections,
                                       const bool &bFixScale);

    static int OptimizeSim3(KeyFrame *pKF1, KeyFrame *pKF2, std::vector<MapPoint *> &vpMatches1,
                            g2o::Sim3 &g2oS12, const float th2, const bool bFixScale);

protected:
    static g2o::SE3Quat ToSE3Quat(const Sophus::SE3d &Tcw);
};

} // namespace slam

#endif