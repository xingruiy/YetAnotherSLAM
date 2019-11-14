#pragma once
#include "dataStruct/frame.h"
#include "dataStruct/mapPoint.h"
#include "utils/numType.h"
#include <memory>

class Frame;
class MapPoint;

class KeyFrame
{
public:
    KeyFrame(const Frame &F);

    double getDepth(double x, double y);

    std::vector<size_t> getKeyPointsInArea(const double x, const double y, const double th);

    inline void setPose(const SE3 &T)
    {
        RT = T;
        RTinv = T.inverse();
    }

public:
    size_t KFId;
    static size_t nextKFId;

    cv::Mat imRGB, imDepth;
    cv::Mat nmap;

    cv::Mat descriptors;
    std::shared_ptr<KeyFrame> parent;
    std::vector<bool> isInliers;
    std::vector<cv::KeyPoint> keyPoints;
    std::vector<std::shared_ptr<MapPoint>> mapPoints;

    SE3 RT, RTinv;
    size_t localReferenceId;
};