#pragma once
#include <OpenNI2/OpenNI.h>
#include <opencv2/opencv.hpp>

namespace OpenNI2
{

using namespace cv;
using namespace openni;

class Camera
{

public:
    ~Camera();
    Camera();
    Camera(const int &w, const int &h, const int &fps);
    bool TryFetchingImages(Mat &imDepth, Mat &imRGB);

private:
    int mWidth;
    int mHeight;
    int mFrameRate;

    Device mDevice;
    VideoStream mDepthStream;
    VideoStream mColourStream;
    VideoFrameRef mDepthFrameRef;
    VideoFrameRef mColourFrameRef;
};

} // namespace OpenNI