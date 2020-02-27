#pragma once

#include <pangolin/pangolin.h>
#include <pangolin/gl/glcuda.h>
#include <mutex>

#include "System.h"
#include "MapDrawer.h"
#include "GlobalDef.h"

namespace SLAM
{

class System;

class Viewer
{
public:
    Viewer(System *pSystem, MapDrawer *pMapDrawer);

    void Run();
    void reset();

    void setLiveImage(const cv::Mat &ImgRGB);
    void setLiveDepth(const cv::Mat &ImgDepth);
    void setLivePose(const Eigen::Matrix4d &Tcw);
    void setKeyFrameImage(const cv::Mat &im, std::vector<cv::KeyPoint> vKeys);

private:
    MapDrawer *mpMapDrawer;
    System *mpSystem;

    void RenderImagesToScreen();
    void RenderLiveCameraFrustum();

    bool mbNewImage;
    bool mbNewDepth;
    bool mbNewKF;

    int width, height;
    Eigen::Matrix3f mCalibInv;
    pangolin::GlTexture mTextureKF;
    pangolin::GlTexture mTextureColour;
    pangolin::GlTexture mTextureDepth;

    pangolin::View *mpMapView;
    pangolin::View *mpRightImageBar;
    pangolin::View *mpCurrentKFView;
    pangolin::View *mpCurrentImageView;
    pangolin::View *mpCurrentDepthView;

    cv::Mat mCvImageRGB;   // Colour image
    cv::Mat mCvImageKF;    // KF image
    cv::Mat mCvImageDepth; // Depth image

    std::mutex mPoseMutex;
    std::mutex mImageMutex;
    std::vector<bool> mvMatches;
    Eigen::Matrix4f mCurrentCameraPose;
    std::vector<cv::KeyPoint> mvCurrentKeys;
};

} // namespace SLAM
