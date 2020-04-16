/**
* This file is part of ORB-SLAM2.
*
* Copyright (C) 2014-2016 Raúl Mur-Artal <raulmur at unizar dot es> (University of Zaragoza)
* For more information see <https://github.com/raulmur/ORB_SLAM2>
*
* ORB-SLAM2 is free software: you can redistribute it and/or modify
* it under the terms of the GNU General Public License as published by
* the Free Software Foundation, either version 3 of the License, or
* (at your option) any later version.
*
* ORB-SLAM2 is distributed in the hope that it will be useful,
* but WITHOUT ANY WARRANTY; without even the implied warranty of
* MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the
* GNU General Public License for more details.
*
* You should have received a copy of the GNU General Public License
* along with ORB-SLAM2. If not, see <http://www.gnu.org/licenses/>.
*/

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