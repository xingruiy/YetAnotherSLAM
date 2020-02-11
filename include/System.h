#pragma once
#include <memory>
#include <thread>
#include <iostream>

#include "Map.h"
#include "Frame.h"
#include "Viewer.h"
#include "KeyFrame.h"
#include "Tracking.h"
#include "Mapping.h"

#include "DENSE/include/DenseMapping.h"
#include "DENSE/include/DenseTracking.h"

namespace SLAM
{

class Viewer;
class Mapping;
class Tracking;
class MapViewer;

class System
{
public:
    ~System();
    System(const std::string &strSettingFile);
    void TrackImage(const cv::Mat &ImgRGB, const cv::Mat &ImgDepth, const double TimeStamp);

    bool IsAlive() const;
    void Kill();
    void Reset();
    void Pause();
    void UnPause();

private:
    Tracking *mpTracker;
    Map *mpMap;
    Viewer *mpViewer;
    Mapping *mpMapping;

    std::thread *mpThreadMapping;
    std::thread *mpThreadViewer;

    bool mbIsAlive;
    bool mbReverseRGB;
    bool mbIsRunning;

    double mDepthScale;
    cv::Mat mImGray, mImDepth;
};

} // namespace SLAM