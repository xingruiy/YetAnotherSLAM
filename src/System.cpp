#include "System.h"
#include "DENSE/include/ImageProc.h"

namespace SLAM
{

System::System(const std::string &strSettingFile) : viewer(nullptr)
{
    readSettings(strSettingFile);

    mpMap = new Map();

    if (g_bEnableViewer)
    {
        viewer = new Viewer(this, mpMap);
        viewerThread = new thread(&Viewer::Run, viewer);
    }

    mapping = new Mapping(mpMap);
    mappingThread = new thread(&Mapping::Run, mapping);

    tracker = new Tracking(this, mpMap, viewer, mapping);
    std::cout << "Main Thread Started." << std::endl;
}

void System::trackImage(cv::Mat img, cv::Mat depth, const double timeStamp)
{
    if (!g_bReverseRGB)
        cv::cvtColor(img, grayScale, cv::COLOR_RGB2GRAY);
    else
        cv::cvtColor(img, grayScale, cv::COLOR_BGR2GRAY);

    // Convert depth to floating point
    depth.convertTo(depthFloat, CV_32FC1, g_DepthScaleInv);

    if (g_bEnableViewer)
    {
        viewer->setLiveImage(img);
        viewer->setLiveDepth(depthFloat);
    }

    if (!g_bSystemRunning)
        return;

    // Invoke the main tracking thread
    tracker->trackImage(grayScale, depthFloat, timeStamp);
}

void System::reset()
{
    tracker->reset();
    mpMap->reset();
}

void System::kill()
{
    g_bSystemKilled = true;
}

System::~System()
{
    std::cout << "System Waits for Other Threads." << std::endl;
    mappingThread->join();
    viewerThread->join();

    delete mpMap;
    delete viewer;
    delete tracker;
    delete mapping;
    delete mappingThread;
    delete viewerThread;

    std::cout << "System Killed." << std::endl;
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
    g_DepthScaleInv = 1.0 / (float)settingsFile["System.DepthScale"];

    // read orb parameters
    g_ORBScaleFactor = settingsFile["ORB_SLAM2.scaleFactor"];
    g_ORBNFeatures = settingsFile["ORB_SLAM2.nFeatures"];
    g_ORBNLevels = settingsFile["ORB_SLAM2.nLevels"];
    g_ORBIniThFAST = settingsFile["ORB_SLAM2.iniThFAST"];
    g_ORBMinThFAST = settingsFile["ORB_SLAM2.minThFAST"];

    // read calibration parameters
    g_bf = settingsFile["Calibration.bf"];
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
    g_thDepth = g_bf * (float)settingsFile["Tracking.ThDepth"] / g_fx[0];

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

    std::cout << "===================================================\n"
              << "The system is created with the following parameters:\n"
              << "fx - " << g_fx[0] << "\n"
              << "fy - " << g_fy[0] << "\n"
              << "cx - " << g_cx[0] << "\n"
              << "cy - " << g_cy[0] << "\n"
              << "frame width - " << g_width[0] << "\n"
              << "frame height - " << g_height[0] << "\n"
              << "pyramid level - " << NUM_PYR << "\n"
              << "enable viewer? - " << (g_bEnableViewer ? "yes" : "no") << "\n"
              << "===================================================" << std::endl;
}

} // namespace SLAM