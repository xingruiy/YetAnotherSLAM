#include "System.h"
#include "DENSE/include/ImageProc.h"

namespace SLAM
{

System::System(const std::string &strSettingFile)
    : mbIsAlive(true), mbIsRunning(false)
{
    cv::FileStorage settingsFile(strSettingFile, cv::FileStorage::READ);

    if (settingsFile.isOpened())
    {
        mbReverseRGB = (int)settingsFile["System.ReverseRGB"] == 1;
        mDepthScale = 1.0 / (float)settingsFile["System.DepthScale"];

        mpMap = new Map();

        mpMapping = new Mapping(mpMap);
        mpThreadMapping = new thread(&Mapping::Run, mpMapping);

        mpTracker = new Tracking(strSettingFile, this, mpMap);
        mpTracker->SetLocalMapper(mpMapping);

        if ((int)settingsFile["System.EnableViewer"] == 1)
        {
            mpViewer = new Viewer(strSettingFile, this, mpMap);
            mpThreadViewer = new thread(&Viewer::Run, mpViewer);
            mpTracker->SetViewer(mpViewer);
            mpMapping->SetViewer(mpViewer);
        }
    }
    else
    {
        printf("Reading settings failed at line %d in file %s\n", __LINE__, __FILE__);
        exit(-1);
    }
}

System::~System()
{
    mpMapping->Kill();
    mpThreadMapping->join();
    mpThreadViewer->join();

    delete mpMap;
    delete mpTracker;
    delete mpMapping;
    delete mpThreadMapping;
    delete mpThreadViewer;
}

void System::TrackImage(const cv::Mat &ImgColour, const cv::Mat &ImgDepth, const double TimeStamp)
{
    if (!mbReverseRGB)
        cv::cvtColor(ImgColour, mImGray, cv::COLOR_RGB2GRAY);
    else
        cv::cvtColor(ImgColour, mImGray, cv::COLOR_BGR2GRAY);

    // Convert depth to floating point
    ImgDepth.convertTo(mImDepth, CV_32FC1, mDepthScale);

    if (mpViewer)
    {
        mpViewer->SetCurrentRGBImage(ImgColour);
        mpViewer->SetCurrentDepthImage(mImDepth);
    }

    if (!mbIsRunning)
        return;

    // Invoke the main tracking thread
    mpTracker->TrackImage(mImGray, mImDepth, TimeStamp);
}

void System::Reset()
{
    mpTracker->Reset();
    mpMap->Reset();
}

bool System::IsAlive() const
{
    return mbIsAlive;
}

void System::Kill()
{
    mbIsAlive = false;
    printf("System Killed.\n");
}

void System::Pause()
{
    mbIsRunning = false;
}

void System::UnPause()
{
    mbIsRunning = true;
}

} // namespace SLAM