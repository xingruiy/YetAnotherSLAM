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
    std::map<KeyFrame *, size_t> GetObservations();
    int Observations();

    void AddObservation(KeyFrame *pKF, size_t idx);
    void EraseObservation(KeyFrame *pKF);
    void SetBadFlag();
    bool isBad();

    Eigen::Vector3d GetWorldPos();
    Eigen::Vector3d GetNormal();
    void Replace(MapPoint *pMP);
    MapPoint *GetReplaced();

    cv::Mat GetDescriptor();
    void IncreaseVisible(int n = 1);

    bool IsInKeyFrame(KeyFrame *pKF);
    void UpdateNormalAndDepth();
    void ComputeDistinctiveDescriptors();

    float GetMinDistanceInvariance();
    float GetMaxDistanceInvariance();

    int PredictScale(const float &currentDist, Frame *pFrame);
    int PredictScale(const float &currentDist, KeyFrame *pKF);

public:
    unsigned long mnId;
    static unsigned long nNextId;

    long int mnFirstKFid;
    long int mnFirstFrame;

    // Reference KeyFrame
    KeyFrame *mpRefKF;

    // Keyframes observing the point and associated index in keyframe
    std::map<KeyFrame *, size_t> mObservations;

    // Position in absolute coordinates
    Eigen::Vector3d mWorldPos;

    // Mean viewing direction
    Eigen::Vector3d mNormalVector;

    // The point's normal direction
    Eigen::Vector3d mPointNormal;

    // Best descriptor to fast matching
    cv::Mat mDescriptor;

    // Bad flag (we do not currently erase MapPoint from memory)
    bool mbBad;
    MapPoint *mpReplaced;

    Map *mpMap;

    std::mutex mMutexPos;
    std::mutex mMutexFeatures;
    static std::mutex mGlobalMutex;

    // Tracking counters
    int mnVisible;
    int mnFound;
    int nObs;

    // Scale invariance distances
    float mfMinDistance;
    float mfMaxDistance;

    // Temporary varialbes
    unsigned long mnBALocalForKF;
    unsigned long mnFuseCandidateForKF;
    unsigned long mnTrackReferenceForFrame;

    // Variables used by the tracking
    float mTrackProjX;
    float mTrackProjY;
    float mTrackProjXR;
    bool mbTrackInView;
    int mnTrackScaleLevel;
    float mTrackViewCos;
};

} // namespace SLAM