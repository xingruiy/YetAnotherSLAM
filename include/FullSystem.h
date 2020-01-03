#pragma once
#include <memory>
#include <thread>
#include <iostream>

#include "Map.h"
#include "Frame.h"
#include "KeyFrame.h"
#include "Viewer.h"
#include "Tracking.h"
#include "LocalMapping.h"

#include "DENSE/include/DenseMapping.h"
#include "DENSE/include/DenseTracking.h"
#include "ORBVocabulary.h"

class Viewer;
class MapViewer;
class Tracking;
class LocalMapping;

class FullSystem
{
public:
    FullSystem(const std::string &strSettingFile, const std::string &strVocFile);

    void TrackImageRGBD(const cv::Mat &imRGB, const cv::Mat &imDepth, const double TimeStamp);

    void SetToPause();
    void SetToUnPause();
    void SetToFinish();
    bool IsFinished();

    void Reset();

private:
    Tracking *mpTracker;

    Viewer *mpViewer;

    Map *mpMap;

    // System threads: Local Mapping, Loop Closing, Viewer.
    // The Tracking thread "lives" in the main execution thread that creates the System object.
    std::thread *mptLocalMapping;
    std::thread *mptLoopClosing;
    std::thread *mptViewer;

    bool mbRGB;
    bool mbUseVoc;
    bool mbUseViewer;
    bool mbPaused;

    bool mbFinished;

    cv::Mat mImGray, mImDepth;
    double mDepthScale;

    ORB_SLAM2::ORBVocabulary *mpORBVocabulary;
};