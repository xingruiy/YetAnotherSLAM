#pragma once
#include "utils/frame.h"
#include "utils/mapPoint.h"

class Localizer
{
public:
    Localizer();
    void AbsoluteOrientation(
        std::vector<Vec3d> &ref,
        std::vector<Vec3d> &src,
        SE3 &finalEstimate,
        size_t &numInliers,
        const int maxIterations);

    SE3 getRelativeTransform(
        std::shared_ptr<Frame> reference,
        std::shared_ptr<Frame> current);

    SE3 getWorldTransform(
        std::shared_ptr<Frame> frame,
        std::vector<cv::DMatch> &matches,
        std::vector<std::shared_ptr<MapPoint>> &pts);
};