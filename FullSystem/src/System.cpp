#include "System.h"

namespace SLAM
{

System::System(const std::string &strSettingFile, const std::string &strVocFile)
    : mpViewer(nullptr)
{
    readSettings(strSettingFile);
    loadORBVocabulary(strVocFile);

    mpMap = new Map();
    mpMapDrawer = new MapDrawer(mpMap);
    mpKeyFrameDB = new KeyFrameDatabase(*mpORBVocabulary);

    mpLoopClosing = new LoopClosing(mpMap, mpKeyFrameDB, mpORBVocabulary);
    mpLoopThread = new std::thread(&LoopClosing::Run, mpLoopClosing);

    mpLocalMapper = new LocalMapping(mpORBVocabulary, mpMap);
    mpLocalMapper->setLoopCloser(mpLoopClosing);
    mpLocalMappingThread = new std::thread(&LocalMapping::Run, mpLocalMapper);

    mpTracker = new Tracking(this, mpMap);
    mpTracker->SetLocalMapper(mpLocalMapper);

    if (g_bEnableViewer)
    {
        mpViewer = new Viewer(this, mpMapDrawer);
        mpViewerThread = new std::thread(&Viewer::Run, mpViewer);
        mpTracker->SetViewer(mpViewer);
        mpLocalMapper->setViewer(mpViewer);
    }
}

void System::TrackRGBD(cv::Mat img, cv::Mat depth, const double timeStamp)
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

void System::reset()
{
    mpTracker->reset();
    mpMap->reset();
}

void System::Kill()
{
    g_bSystemKilled = true;
}

System::~System()
{
    mpLoopThread->join();
    mpViewerThread->join();
    mpLocalMappingThread->join();

    delete mpMap;
    delete mpViewer;
    delete mpTracker;
    delete mpLoopThread;
    delete mpLocalMapper;
    delete mpLoopClosing;
    delete mpViewerThread;
    delete mpLocalMappingThread;
}

void System::readSettings(const std::string &strSettingFile)
{
    cv::FileStorage settingsFile(strSettingFile, cv::FileStorage::READ);
    if (!settingsFile.isOpened())
    {
        printf("Reading settings failed at line %d in file %s\n", __LINE__, __FILE__);
        exit(-1);
    }

    // read system configurations
    g_bEnableViewer = (int)settingsFile["System.EnableViewer"] == 1;
    g_bReverseRGB = (int)settingsFile["System.ReverseRGB"] == 1;
    g_DepthScaleInv = 1.0 / (double)settingsFile["System.DepthScale"];

    // read orb parameters
    g_ORBScaleFactor = settingsFile["ORB_SLAM2.scaleFactor"];
    g_ORBNFeatures = settingsFile["ORB_SLAM2.nFeatures"];
    g_ORBNLevels = settingsFile["ORB_SLAM2.nLevels"];
    g_ORBIniThFAST = settingsFile["ORB_SLAM2.iniThFAST"];
    g_ORBMinThFAST = settingsFile["ORB_SLAM2.minThFAST"];
    g_pORBExtractor = new ORBextractor(g_ORBNFeatures, g_ORBScaleFactor, g_ORBNLevels, g_ORBIniThFAST, g_ORBMinThFAST);

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

    computeImageBounds();

    g_pointSize = settingsFile["Viewer.PointSize"];
    g_bSystemRunning = (int)settingsFile["Viewer.StartWhenReady"] == 1;

    std::cout << "===================================================\n"
              << "The system is created with the following parameters:\n"
              << "image bounds: " << g_minX << " " << g_maxX << " " << g_minY << " " << g_maxY << "\n"
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

void System::loadORBVocabulary(const std::string &strVocFile)
{
    std::cout << "loading ORB vocabulary..." << std::endl;
    mpORBVocabulary = new ORBVocabulary();
    mpORBVocabulary->loadFromTextFile(strVocFile);
    std::cout << "ORB vocabulary loaded..." << std::endl;
}

} // namespace SLAM