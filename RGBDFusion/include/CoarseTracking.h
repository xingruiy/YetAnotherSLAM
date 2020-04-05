#ifndef COARSE_TRACKING_H
#define COARSE_TRACKING_H

#include <opencv2/opencv.hpp>
#include <sophus/se3.hpp>
#include <Eigen/Core>
#include <vector>

#define NUM_PYR 5

class FrameShell
{
public:
    float *gradSqrPyr[NUM_PYR];
    Eigen::Vector3f *img;
    Eigen::Vector3f *imgPyr[NUM_PYR];

    Eigen::Vector<float, 3> *depth;
    Eigen::Vector<float, 3> *depthPyr[NUM_PYR];

    Sophus::SE3d Tcw;
    double timestamp;

    bool bKF = false;
};

class CoarseTracking
{
public:
    CoarseTracking(int w, int h, float fx, float fy, float cx, float cy);

    void AddFrame(float *img, float *depth, double timestamp);
    void MakeImages(FrameShell *F, float *img, float *depth);
    void trackCoarseLevel(FrameShell *F);
    void discardLastFrame();

    Sophus::SE3d estimate;

private:
    float fxi[NUM_PYR];
    float fyi[NUM_PYR];
    float fx[NUM_PYR];
    float fy[NUM_PYR];
    float cx[NUM_PYR];
    float cy[NUM_PYR];
    int w[NUM_PYR];
    int h[NUM_PYR];

    FrameShell *lastF;
    FrameShell *lastlastF;
    std::vector<FrameShell *> FsHist;
};

#endif