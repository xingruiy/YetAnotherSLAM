#pragma once
#include "DataStruct/frame.h"
#include "DataStruct/mapPoint.h"
#include "utils/numType.h"
#include <memory>

class Frame;
class MapPoint;

class KeyFrame
{
public:
    KeyFrame(const Frame &F);
    KeyFrame(Mat imRGB, Mat imDepth, Mat imGray, Mat nmap, Mat33d &K);

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
    std::vector<bool> outliers;
    std::vector<std::shared_ptr<MapPoint>> mapPoints;
    std::vector<cv::KeyPoint> keyPoints;
    std::vector<Vec3d> x3DPoints;
    std::vector<Vec3d> x3DNormal;

    size_t localReferenceId;
    SE3 RT, RTinv;
};