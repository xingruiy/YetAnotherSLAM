#pragma once

#include "KeyFrame.h"
#include "Map.h"
#include <Eigen/Core>
#include <opencv2/opencv.hpp>

class Map;
class KeyFrame;

using namespace std;

class MapPoint
{
public:
    MapPoint(const Eigen::Vector3d &pos, KeyFrame *pRefKF, Map *pMap);

    std::map<KeyFrame *, size_t> GetObservations();

    int Observations();

    void AddObservation(KeyFrame *pKF, size_t idx);

    void EraseObservation(KeyFrame *pKF);

    void SetBadFlag();

    bool isBad();

    void Replace(MapPoint *pMP);

    MapPoint *GetReplaced();

    bool IsInKeyFrame(KeyFrame *pKF);
    void UpdateNormalAndDepth();
    void ComputeDistinctiveDescriptors();

    float GetMinDistanceInvariance();
    float GetMaxDistanceInvariance();

public:
    unsigned long mnId;
    static unsigned long nNextId;

    long int mnFirstKFid;
    long int mnFirstFrame;

    // Reference KeyFrame
    KeyFrame *mpRefKF;

    // Keyframes observing the point and associated index in keyframe
    std::map<KeyFrame *, size_t> mObservations;

    // Mean viewing direction
    cv::Mat mNormalVector;

    // Best descriptor to fast matching
    cv::Mat mDescriptor;

    // Bad flag (we do not currently erase MapPoint from memory)
    bool mbBad;
    MapPoint *mpReplaced;

    Map *mpMap;

    std::mutex mMutexPos;
    std::mutex mMutexFeatures;
    static std::mutex mGlobalMutex;

    // Position in absolute coordinates
    Eigen::Vector3d mWorldPos;

    int nObs;
    int nFrameObs;

    // Scale invariance distances
    float mfMinDistance;
    float mfMaxDistance;

    bool mbTrackInView;
    bool mnTrackScaleLevel;
};