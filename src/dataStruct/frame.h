#pragma once
#include <mutex>
#include <memory>
#include "utils/numType.h"

class Frame
{
public:
    Frame() = default;
    Frame(Mat imRGB, Mat imDepth, Mat imGray, Mat nmap, Mat33d &K);

public:
    Mat33d K;
    cv::Mat imDepth;
    cv::Mat imRGB;
    cv::Mat imGray;
    cv::Mat nmap;
};
