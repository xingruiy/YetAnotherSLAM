#pragma once
#include "utils/map.h"
#include "utils/frame.h"
#include "utils/mapPoint.h"

class Localizer
{
    std::vector<cv::DMatch> getMatches2NN(
        Mat src, Mat dst,
        bool allowAmbiguity);

    void getWorldTransform(
        const std::vector<std::vector<Vec3d>> &src,
        const std::vector<std::vector<Vec3d>> &dst,
        std::vector<std::vector<bool>> &filter,
        std::vector<SE3> &result);

    void runRansacAO(
        const std::vector<Vec3d> &src,
        const std::vector<Vec3d> &dst,
        std::vector<bool> &outliers,
        SE3 &bestEsimate,
        size_t &numInliers,
        const int maxIterations);

    void absoluteOrientation(
        const std::vector<Vec3d> &srcPts,
        const std::vector<Vec3d> &dstPts,
        SE3 &estimate);

    void absoluteOrientation(
        const std::vector<Vec3d> &srcPts,
        const std::vector<Vec3d> &dstPts,
        const std::vector<bool> &outliers,
        SE3 &estimate);

    int evaluateOutlier(
        const std::vector<Vec3d> &srcPts,
        const std::vector<Vec3d> &dstPts,
        SE3 &estimate,
        std::vector<bool> &outliers);

    void createAdjacencyMat(
        const std::vector<std::shared_ptr<MapPoint>> &mapPoints,
        const std::vector<Vec3d> &framePoints,
        const std::vector<bool> &framePtValid,
        const std::vector<cv::DMatch> &matches,
        Mat &adjacentMat);

    void createAdjacencyMat(
        const std::vector<std::shared_ptr<MapPoint>> &mapPoints,
        const std::vector<Vec3d> &framePoints,
        const std::vector<Vec3f> &frameNormal,
        const std::vector<bool> &framePtValid,
        const std::vector<cv::DMatch> &matches,
        Mat &adjacentMat);

    void selectMatches(
        const Mat &adjacencyMat,
        const std::vector<cv::DMatch> &matches,
        std::vector<std::vector<cv::DMatch>> &subMatches);

public:
    SE3 getRelativeTransform(
        std::shared_ptr<Frame> reference,
        std::shared_ptr<Frame> current);

    SE3 getWorldTransform(
        std::shared_ptr<Frame> frame,
        std::vector<cv::DMatch> &matches,
        std::vector<std::shared_ptr<MapPoint>> &pts);

    bool getRelocHypotheses(
        const std::shared_ptr<Map> map,
        const std::vector<Vec3d> &framePts,
        const std::vector<Vec3f> &frameNormal,
        const Mat framePtDesc,
        const std::vector<bool> &framePtValid,
        std::vector<SE3> &estimateList,
        std::vector<std::vector<cv::DMatch>> &subMatches,
        std::vector<std::vector<bool>> &filter,
        const bool &useGraphMatching);

    // TODO
    bool evalHypotheses(
        const std::shared_ptr<Map> map,
        const std::vector<SE3> &estimateList,
        const std::vector<cv::KeyPoint> &cvKeyPoint,
        const Mat33d &camIntrinsics,
        SE3 &bestEstimate);
};