#include "CoreSystem.h"
#include "MapManager.h"
#include "Map.h"
#include "Frame.h"
#include "Viewer.h"
#include "KeyFrame.h"
#include "Tracking.h"
#include "LocalMapping.h"
#include "GlobalDef.h"
#include "LoopClosing.h"
#include "MapDrawer.h"
#include "utils/GlobalSettings.h"
#include "KeyFrameDatabase.h"
#include "IOSystem/BaseOutput.h"

#define LOG(os, strr) \
    if (os)           \
        *os << strr << std::endl;

namespace slam
{

CoreSystem::CoreSystem(const std::string &strSettingFile, const std::string &strVocFile)
    : mpViewer(nullptr)
{
    //Load Settings
    readSettings(strSettingFile);

    //Load ORB Vocabulary
    loadORBVocabulary(strVocFile);

    //Create the Map
    mpMapManager = new MapManager();
    mpMapDrawer = new MapDrawer(mpMapManager);

    //Create KeyFrame Database
    mpKeyFrameDB = new KeyFrameDatabase(*mpORBVocabulary);

    mpLoopClosing = new LoopClosing(mpMapManager, mpKeyFrameDB, mpORBVocabulary);
    mpLoopThread = new std::thread(&LoopClosing::Run, mpLoopClosing);

    mpLocalMapper = new LocalMapping(mpORBVocabulary, mpMapManager);
    mpLocalMapper->SetLoopCloser(mpLoopClosing);
    mpLoopClosing->SetLocalMapper(mpLocalMapper);
    mpLocalMappingThread = new std::thread(&LocalMapping::Run, mpLocalMapper);

    //Initialize the Tracking thread
    mpTracker = new Tracking(this, mpORBVocabulary, mpMapManager, mpKeyFrameDB);
    mpTracker->SetLocalMapper(mpLocalMapper);

    if (g_bEnableViewer)
    {
        mpViewer = new Viewer(this, mpMapDrawer);
        mpViewerThread = new std::thread(&Viewer::Run, mpViewer);
        mpTracker->SetViewer(mpViewer);
        mpLocalMapper->SetViewer(mpViewer);
    }
}

void CoreSystem::TrackRGBD(cv::Mat img, cv::Mat depth, const double timeStamp)
{
    // Covert colour images to grayscale
    if (!g_bReverseRGB)
        cv::cvtColor(img, grayScale, cv::COLOR_RGB2GRAY);
    else
        cv::cvtColor(img, grayScale, cv::COLOR_BGR2GRAY);

    // Convert depth to floating point
    depth.convertTo(depthFloat, CV_32FC1, g_DepthScaleInv);

    if (mpViewer)
    {
        mpViewer->setLiveImage(img);
        mpViewer->setLiveDepth(depthFloat);
    }

    if (!g_bSystemRunning)
        return;

    // Invoke the main tracking thread
    mpTracker->GrabImageRGBD(grayScale, depthFloat, timeStamp);
}

void CoreSystem::reset()
{
    mpTracker->reset();
    mpMapManager->Reset();
    // mpMap->reset();
}

void CoreSystem::FuseAllMapStruct()
{
    Map *pMap = mpMapManager->GetActiveMap();
    auto mapStructs = pMap->GetAllVoxelMaps();

    MapStruct *pMSini = pMap->mpMapStructOrigin;
    if (pMSini->mbInHibernation)
        pMSini->ReActivate();

    pMSini->Reserve(800000, 600000, 800000);
    pMSini->SetActiveFlag(false);
    if (!pMSini || mapStructs.size() == 0)
        return;

    for (auto vit : mapStructs)
    {
        if (pMSini == vit || !vit || vit->isActive())
            continue;

        if (vit->mbInHibernation)
            vit->ReActivate();

        pMSini->Fuse(vit);
        vit->Release();
        vit->mbSubsumed = true;
        vit->mpParent = pMSini;
        pMap->EraseMapStruct(vit);
        std::cout << "fusing map: " << vit->mnId << std::endl;
    }

    pMSini->GenerateMesh();
    pMSini->SetActiveFlag(false);
}

void CoreSystem::DisplayNextMap()
{
}

void CoreSystem::WriteToFile(const std::string &strFile)
{
    // mpMap->WriteToFile(strFile);
}

void CoreSystem::ReadFromFile(const std::string &strFile)
{
    // mpMap->ReadFromFile(strFile);
}

void CoreSystem::Shutdown()
{
    g_bSystemKilled = true;
}

CoreSystem::~CoreSystem()
{
    mpLoopThread->join();
    mpViewerThread->join();
    mpLocalMappingThread->join();

    // delete mpMap;
    delete mpViewer;
    delete mpTracker;
    delete mpLoopThread;
    delete mpLocalMapper;
    delete mpLoopClosing;
    delete mpViewerThread;
    delete mpLocalMappingThread;
}

void CoreSystem::readSettings(const std::string &strSettingFile)
{
    cv::FileStorage settingsFile(strSettingFile, cv::FileStorage::READ);
    if (!settingsFile.isOpened())
    {
        printf("Reading settings failed at line %d in file %s\n", __LINE__, __FILE__);
        exit(-1);
    }

    // read system configurations
    g_bEnableViewer = (int)settingsFile["CoreSystem.EnableViewer"] == 1;
    g_bReverseRGB = (int)settingsFile["CoreSystem.ReverseRGB"] == 1;
    g_DepthScaleInv = 1.0 / (double)settingsFile["CoreSystem.DepthScale"];

    // read orb parameters
    g_ORBScaleFactor = settingsFile["ORB_SLAM2.scaleFactor"];
    g_ORBNFeatures = settingsFile["ORB_SLAM2.nFeatures"];
    g_ORBNLevels = settingsFile["ORB_SLAM2.nLevels"];
    g_ORBIniThFAST = settingsFile["ORB_SLAM2.iniThFAST"];
    g_ORBMinThFAST = settingsFile["ORB_SLAM2.minThFAST"];

    // read calibration parameters
    int width = settingsFile["Calibration.width"];
    int height = settingsFile["Calibration.height"];
    float fx = settingsFile["Calibration.fx"];
    float fy = settingsFile["Calibration.fy"];
    float cx = settingsFile["Calibration.cx"];
    float cy = settingsFile["Calibration.cy"];
    Eigen::Matrix3d calib;
    calib << fx, 0, cx, 0, fy, cy, 0, 0, 1.0;
    setGlobalCalibration(width, height, calib);

    // Update tracking parameters
    g_bf = settingsFile["Calibration.bf"];
    g_thDepth = g_bf * (float)settingsFile["Tracking.ThDepth"] / fx;
    g_bUseColour = (int)settingsFile["Tracking.UseColour"] == 1;
    g_bUseDepth = (int)settingsFile["Tracking.UseDepth"] == 1;

    // read distortion coefficients
    g_distCoeff = cv::Mat(4, 1, CV_32F);
    g_distCoeff.at<float>(0) = settingsFile["UnDistortion.k1"];
    g_distCoeff.at<float>(1) = settingsFile["UnDistortion.k2"];
    g_distCoeff.at<float>(2) = settingsFile["UnDistortion.p1"];
    g_distCoeff.at<float>(3) = settingsFile["UnDistortion.p2"];
    const float k3 = settingsFile["UnDistortion.k3"];
    if (k3 != 0)
    {
        g_distCoeff.resize(5);
        g_distCoeff.at<float>(4) = k3;
    }

    g_pointSize = settingsFile["Viewer.PointSize"];
    g_bSystemRunning = (int)settingsFile["Viewer.StartWhenReady"] == 1;

    std::cout << "===================================================\n"
              << "The system is created with the following parameters:\n"
              << "pyramid level - " << NUM_PYR << "\n";
    for (int i = 0; i < NUM_PYR; ++i)
    {
        std::cout << "pyramid " << i << " -"
                  << " width: " << g_width[i]
                  << " height: " << g_height[i]
                  << " fx: " << g_fx[i]
                  << " fy: " << g_fy[i]
                  << " cx: " << g_cx[i]
                  << " cy: " << g_cy[i] << "\n";
    }
    std::cout << "camera baseline - " << g_bf / fx << "\n"
              << "close point th - " << g_thDepth << "\n"
              << "enable mpViewer? - " << (g_bEnableViewer ? "yes" : "no") << "\n"
              << "===================================================" << std::endl;
}

void CoreSystem::loadORBVocabulary(const std::string &strVocFile)
{
    std::cout << "loading ORB vocabulary..." << std::endl;

    mpORBVocabulary = new ORBVocabulary();
    mpORBVocabulary->loadFromBinaryFile(strVocFile);

    std::cout << "ORB vocabulary loaded..." << std::endl;
}

CoreSystem::CoreSystem(GlobalSettings *settings, const std::string &strVocFile)
    : settings(settings), rawLogs(0)
{
    if (settings->logRawOutputs)
    {
        if (system("mkdir -p logs") != 0)
            return;

        rawLogs = new std::ofstream("logs/rawOutput.txt");
    }

    ORBVocab = new ORBVocabulary;
    ORBVocab->loadFromBinaryFile(strVocFile);
    LOG(rawLogs, "Orb vocabulary loaded.");

    coarseTracker = new CoarseTracking(settings);
    maps = new MapManager();
    KFDatabase = new KeyFrameDatabase(*ORBVocab);

    {
        std::thread *thd = new std::thread(&CoreSystem::loopClosingThread, this);
        childThreads.push_back(thd);
        thd = new std::thread(&CoreSystem::localMappingThread, this);
        childThreads.push_back(thd);
    }
}

void CoreSystem::takeNewImages(float *img, float *depth, const double ts)
{
    Frame *newFrame = new Frame();
    allFrames.push_back(newFrame);

    newFrame->img = new Eigen::Vector3f[width * height];
    newFrame->depth = new Eigen::Vector3f[width * height];
    newFrame->timeStamp = ts;

    for (int idx = 0; idx < width * height; ++idx)
    {
        newFrame->img[idx][0] = img[idx];
        newFrame->depth[idx][1] = img[idx];
    }

    if (hasLost)
    {
        // TODO: strat relocalisation
    }
    else
    {
        if (hasInitialized)
        {
            Sophus::SE3d lastF2Ref;
            bool trackingOK = coarseTracker->trackNewFrame(newFrame, lastF2Ref);

            if (trackingOK)
            {
                Eigen::Vector3f lastFlow = coarseTracker->getLastFlowVec();
                needKF = true;

                for (auto output : outputs)
                    output->publishFrame(newFrame);
            }

            if (needKF)
                makeKeyFrame();
        }
        else
        {
            coarseTracker->setRefFrame(newFrame);
            hasInitialized = true;
        }
    }
}

void CoreSystem::blockUntilReset()
{
}

void CoreSystem::makeKeyFrame()
{
}

void CoreSystem::localMappingThread()
{
    while (!emergencyBreak)
    {
        if (tellChildThreadsToWrapup)
        {
            break;
        }
        else
        {
            usleep(3000);
        }
    }
}

void CoreSystem::loopClosingThread()
{
    LoopClosing loopCloser(maps, KFDatabase, ORBVocab);

    while (!emergencyBreak)
    {
        loopCloser.Run();

        if (tellChildThreadsToWrapup)
        {
            break;
        }
        else
        {
            usleep(3000);
        }
    }
}

void CoreSystem::blockUntilFinished()
{
    // wait for threads to finish
    tellChildThreadsToWrapup = true;
    for (auto *td : childThreads)
        td->join();

    // wait for outputs to finish
    for (auto output : outputs)
    {
        while (output && !output->isFinished)
            usleep(5000);
    }
}

} // namespace slam