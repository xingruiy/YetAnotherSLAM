#ifndef LOOPCLOSING_H
#define LOOPCLOSING_H

#include <sophus/se3.hpp>
#include <sophus/sim3.hpp>
#include <g2o/types/sim3/types_seven_dof_expmap.h>

#include "Map.h"
#include "Tracking.h"
#include "LocalMapping.h"
#include "KeyFrameDatabase.h"

namespace SLAM
{

class Map;
class Tracking;
class LocalMapping;
class KeyFrameDatabase;

class LoopClosing
{
public:
    typedef std::pair<std::set<KeyFrame *>, int> ConsistentGroup;
    typedef std::map<KeyFrame *, g2o::Sim3, std::less<KeyFrame *>,
                     Eigen::aligned_allocator<std::pair<const KeyFrame *, g2o::Sim3>>>
        KeyFrameAndPose;

public:
    LoopClosing(Map *pMap, KeyFrameDatabase *pDB, ORBVocabulary *pVoc);

    void SetTracker(Tracking *pTracker);

    void SetLocalMapper(LocalMapping *pLocalMapper);

    // Main function
    void Run();

    void InsertKeyFrame(KeyFrame *pKF);

    void RequestReset();

    // This function will run in a separate thread
    void RunGlobalBundleAdjustment(unsigned long nLoopKF);

    bool isRunningGBA()
    {
        std::unique_lock<std::mutex> lock(mMutexGBA);
        return mbRunningGBA;
    }

    bool isFinishedGBA()
    {
        std::unique_lock<std::mutex> lock(mMutexGBA);
        return mbFinishedGBA;
    }

    EIGEN_MAKE_ALIGNED_OPERATOR_NEW

private:
    bool CheckNewKeyFrames();

    bool DetectLoop();

    bool ComputeSim3();

    void SearchAndFuse(const KeyFrameAndPose &CorrectedPosesMap);

    void CorrectLoop();

    Map *mpMap;
    Tracking *mpTracker;

    KeyFrameDatabase *mpKeyFrameDB;
    ORBVocabulary *mpORBVocabulary;

    LocalMapping *mpLocalMapper;

    std::list<KeyFrame *> mlpLoopKeyFrameQueue;

    std::mutex mMutexLoopQueue;

    // Loop detector parameters
    float mnCovisibilityConsistencyTh;

    // Loop detector variables
    KeyFrame *mpCurrentKF;
    KeyFrame *mpMatchedKF;
    std::vector<ConsistentGroup> mvConsistentGroups;
    std::vector<KeyFrame *> mvpEnoughConsistentCandidates;
    std::vector<KeyFrame *> mvpCurrentConnectedKFs;
    std::vector<MapPoint *> mvpCurrentMatchedPoints;
    std::vector<MapPoint *> mvpLoopMapPoints;
    Sophus::Sim3d mScw;
    g2o::Sim3 mg2oScw;

    long unsigned int mLastLoopKFid;

    // Variables related to Global Bundle Adjustment
    bool mbRunningGBA;
    bool mbFinishedGBA;
    bool mbStopGBA;
    std::mutex mMutexGBA;
    std::thread *mpThreadGBA;

    // Fix scale in the stereo/RGB-D case
    bool mbFixScale;

    bool mnFullBAIdx;
};

} // namespace SLAM

#endif