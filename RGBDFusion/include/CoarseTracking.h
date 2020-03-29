#ifndef COARSE_TRACKING_H
#define COARSE_TRACKING_H

#include <opencv2/opencv.hpp>
#include <sophus/se3.hpp>
#include <Eigen/Core>

class CoarseTracking
{
public:
    CoarseTracking(int w, int h, int minLvl, int maxLvl);

    void setTrackingReference(cv::Mat img, cv::Mat depth);
    void setTrackingTarget(cv::Mat img, cv::Mat depth);
    void setCameraCalibration(float fx, float fy, float cx, float cy);
    Sophus::SE3d getCoarseAlignment(const Sophus::SE3d &Tini);

private:
    int nLvl;
    int minLvl;
    int maxLvl;
    std::vector<float> fxi, fyi;
    std::vector<float> fx, fy, cx, cy;
    std::vector<int> w, h;

    std::vector<float *> mvReferenceImage;
    std::vector<float *> mvTrackingImage;
    std::vector<float *> mvReferenceDepth;
    std::vector<float *> mvTrackingDepth;
};

#endif