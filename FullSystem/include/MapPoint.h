#pragma once
#include <Eigen/Core>
#include <opencv2/opencv.hpp>

#include "Map.h"
#include "Frame.h"
#include "KeyFrame.h"

namespace SLAM
{

class Map;
class Frame;
class KeyFrame;

class MapPoint
{
public:
    MapPoint(const Eigen::Vector3d &Pos, KeyFrame *pRefKF, Map *pMap);
    MapPoint(const Eigen::Vector3d &pos, Map *pMap, KeyFrame *pRefKF, const int &idxF);

    void SetWorldPos(const Eigen::Vector3d &pos);
    Eigen::Vector3d GetWorldPos();

    // Get Normal
    Eigen::Vector3d GetNormal();
    KeyFrame *GetReferenceKeyFrame();

    std::map<KeyFrame *, size_t> GetObservations();
    int Observations();

    void AddObservation(KeyFrame *pKF, size_t idx);
    void EraseObservation(KeyFrame *pKF);

    int GetIndexInKeyFrame(KeyFrame *pKF);
    bool IsInKeyFrame(KeyFrame *pKF);

    void SetBadFlag();
    bool isBad();

    void Replace(MapPoint *pMP);
    MapPoint *GetReplaced();

    void IncreaseFound(int n = 1);
    void IncreaseVisible(int n = 1);
    float GetFoundRatio();

    void ComputeDistinctiveDescriptors();
    cv::Mat GetDescriptor();

    void UpdateNormalAndDepth();

    float GetMinDistanceInvariance();
    float GetMaxDistanceInvariance();
    int PredictScale(const float &currentDist, Frame *pFrame);
    int PredictScale(const float &currentDist, KeyFrame *pKF);

public:
    long unsigned int mnId;
    static long unsigned int nNextId;
    long int mnFirstKFid;
    long int mnFirstFrame;
    int nObs;

    // Variables used by local mapper
    float mTrackProjX;
    float mTrackProjY;
    float mTrackProjXR;
    bool mbTrackInView;
    int mnTrackScaleLevel;
    float mTrackViewCos;
    long unsigned int mnTrackReferenceForFrame;

    // Variables used by local mapper
    long unsigned int mnBALocalForKF;
    long unsigned int mnFuseCandidateForKF;

    // Variables used by loop closing
    long unsigned int mnLoopPointForKF;
    long unsigned int mnCorrectedByKF;
    long unsigned int mnCorrectedReference;
    Eigen::Vector3d mPosGBA;
    long unsigned int mnBAGlobalForKF;

    static std::mutex mGlobalMutex;

public:
    // Position in absolute coordinates
    Eigen::Vector3d mWorldPos;

    // Keyframes observing the point and associated index in keyframe
    std::map<KeyFrame *, size_t> mObservations;

    // Mean viewing direction
    Eigen::Vector3d mAvgViewingDir;

    // Best descriptor to fast matching
    cv::Mat mDescriptor;

    // Reference KeyFrame
    KeyFrame *mpRefKF;

    // Tracking counters
    int mnVisible;
    int mnFound;

    // Bad flag (we do not currently erase MapPoint from memory)
    bool mbBad;
    MapPoint *mpReplaced;

    // Scale invariance distances
    float mfMinDistance;
    float mfMaxDistance;

    Map *mpMap;

    std::mutex mMutexPos;
    std::mutex mMutexFeatures;
};

} // namespace SLAM