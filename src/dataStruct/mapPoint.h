#pragma once
#include "utils/numType.h"
#include "dataStruct/keyFrame.h"
#include <memory>
#include <mutex>
#include <unordered_map>

class KeyFrame;

class MapPoint
{
public:
    MapPoint();

public:
    size_t mpId;
    static size_t nextMpId;

    Eigen::Vector3d pos;
    Eigen::Vector3d normal;
    cv::Mat descriptor;
    std::shared_ptr<KeyFrame> hostKF;
    std::map<std::shared_ptr<KeyFrame>, Vec2d> observations;

    bool setToRemove;
    size_t localReferenceId;
};