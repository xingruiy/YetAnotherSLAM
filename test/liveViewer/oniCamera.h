#pragma once
#include <OpenNI2/OpenNI.h>
#include <opencv2/opencv.hpp>

class ONICamera
{
    int width;
    int height;
    int frameRate;

    openni::Device device;
    openni::VideoStream depthStream;
    openni::VideoStream colourStream;
    openni::VideoFrameRef depthFrameRef;
    openni::VideoFrameRef colourFrameRef;

public:
    ONICamera();
    ~ONICamera();
    ONICamera(int w, int h, int fps);
    bool getNextImages(cv::Mat &depth, cv::Mat &image);
};
