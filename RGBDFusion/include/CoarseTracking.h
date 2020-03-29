#ifndef COARSE_TRACKING_H
#define COARSE_TRACKING_H

#include <opencv2/opencv.hpp>
#include <sophus/se3.hpp>
#include <Eigen/Core>

#define NUM_PYR 6

class CoarseTracking
{
public:
    CoarseTracking(int w, int h, float fx, float fy, float cx, float cy);
    Sophus::SE3d getCoarseAlignment(const Sophus::SE3d &Tini);

    void makeTrackingImage(float *img);
    void makeReferenceImage(float *img);

private:
    float fxi[NUM_PYR], fyi[NUM_PYR];
    float fx[NUM_PYR], fy[NUM_PYR];
    float cx[NUM_PYR], cy[NUM_PYR];
    int w[NUM_PYR], h[NUM_PYR];

    float *referenceImage[NUM_PYR];
    float *trackingImage[NUM_PYR];
};

#endif