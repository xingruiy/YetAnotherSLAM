#pragma once
#include <OpenNI2/OpenNI.h>
#include <opencv2/opencv.hpp>

namespace ONI
{

using namespace cv;
using namespace openni;

class Camera
{

public:
    Camera();
    Camera(const int &w, const int &h, const int &fps);

    ~Camera();

    bool TryFetchingImages(Mat &imDepth, Mat &imRGB);

private:
    int mnWidth;
    int mnHeight;
    int mnFrameRate;

    Device mDevice;
    VideoStream mDepthStream;
    VideoStream mColourStream;
    VideoFrameRef mDepthFrameRef;
    VideoFrameRef mColourFrameRef;
};

} // namespace ONI