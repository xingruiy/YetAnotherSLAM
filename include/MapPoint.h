#pragma once

#include "KeyFrame.h"
#include "Map.h"
#include <Eigen/Core>
#include <opencv2/opencv.hpp>

class Map;
class KeyFrame;

class MapPoint
{
    // public:
    //     MapPoint();

    // public:
    //     size_t mpId;
    //     static size_t nextMpId;

    //     Eigen::Vector3d pos;
    //     Eigen::Vector3d normal;
    //     cv::Mat descriptor;
    //     std::shared_ptr<KeyFrame> hostKF;
    //     std::map<std::shared_ptr<KeyFrame>, size_t> observations;

    //     bool setToRemove;
    //     size_t localReferenceId;
    //     size_t referenceCounter;

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

    // Position in absolute coordinates
    Eigen::Vector3d mWorldPos;
};