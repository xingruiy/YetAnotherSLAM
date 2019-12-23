#include "FullSystem.h"
#include "DENSE/include/ImageProc.h"

FullSystem::FullSystem(const std::string &strSettingFile, const std::string &strVocFile)
    : mbFinished(false), mbPaused(true)
{
    {
        cv::FileStorage fsSetting(strSettingFile, cv::FileStorage::READ);

        if (!fsSetting.isOpened())
        {
            std::cout << "Reading file failed..." << std::endl;
            exit(-1);
        }

        int nRGB = fsSetting["Camera.RGB"];
        int nUseVoc = fsSetting["Global.use_voc"];
        int nUseViewer = fsSetting["Global.use_viewer"];

        mDepthScale = fsSetting["Camera.depth_scale"];
        mbRGB = (nRGB != 0);
        mbUseViewer = (nUseViewer != 0);
        mbUseVoc = (nUseVoc != 0);
    }

    if (mbUseVoc)
    {
        std::cout << "Loading ORB vocabulary file..." << std::endl;
        mpORBVocabulary = new ORB_SLAM2::ORBVocabulary();
        bool bVocLoad = mpORBVocabulary->loadFromTextFile(strVocFile);

        if (!bVocLoad)
        {
            cerr << "Wrong path to vocabulary. " << endl;
            cerr << "Falied to open at: " << strVocFile << endl;
            exit(-1);
        }
    }

    // Create map
    mpMap = new Map();

    // Craete main tracking thread
    mpTracker = new Tracking(strSettingFile, this, mpMap, mpORBVocabulary);

    // Create viewing thread if applicable
    if (mbUseViewer)
    {
        mpViewer = new Viewer(strSettingFile, this, mpMap);
        mptViewer = new thread(&Viewer::Spin, mpViewer);
        mpTracker->SetViewer(mpViewer);
    }
}

void FullSystem::TrackImageRGBD(const cv::Mat &imRGB, const cv::Mat &imDepth, const double TimeStamp)
{
    if (mbPaused)
        return;

    if (mbRGB)
        cv::cvtColor(imRGB, mImGray, cv::COLOR_RGB2GRAY);
    else
        cv::cvtColor(imRGB, mImGray, cv::COLOR_BGR2GRAY);

    imDepth.convertTo(mImDepth, CV_32FC1, 1.0 / mDepthScale);

    mpTracker->TrackImageRGBD(mImGray, mImDepth);
}

void FullSystem::SetToFinish()
{
    mbFinished = true;
}

bool FullSystem::IsFinished()
{
    return mbFinished;
}

void FullSystem::SetToPause()
{
    mbPaused = true;
}

void FullSystem::SetToUnPause()
{
    mbPaused = false;
}

void FullSystem::Reset()
{
    mpTracker->Reset();
    mpMap->Reset();
}