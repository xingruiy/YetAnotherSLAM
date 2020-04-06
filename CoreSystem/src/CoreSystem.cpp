#include "CoreSystem.h"
#include "MapManager.h"

namespace slam
{

CoreSystem::CoreSystem(const std::string &strSettingFile, const std::string &strVocFile)
    : mpViewer(0)
{
    //Load Settings
    readSettings(strSettingFile);

    //Load ORB Vocabulary
    ORBVoc = new ORBVocabulary();
    ORBVoc->loadFromBinaryFile(strVocFile);

    //Create the Map
    mpMapManager = new MapManager();
    mpMapDrawer = new MapDrawer(mpMapManager);

    //Create KeyFrame Database
    mpKeyFrameDB = new KeyFrameDatabase(*ORBVoc);

    mpLoopClosing = new LoopClosing(mpMapManager, mpKeyFrameDB, ORBVoc);
    mpLoopThread = new std::thread(&LoopClosing::Run, mpLoopClosing);

    mpLocalMapper = new LocalMapping(ORBVoc, mpMapManager);
    mpLocalMapper->SetLoopCloser(mpLoopClosing);
    mpLoopClosing->SetLocalMapper(mpLocalMapper);
    mpLocalMappingThread = new std::thread(&LocalMapping::Run, mpLocalMapper);

    //Initialize the Tracking thread
    mpTracker = new Tracking(this, ORBVoc, mpMapManager, mpKeyFrameDB);
    mpTracker->SetLocalMapper(mpLocalMapper);

    if (g_bEnableViewer)
    {
        mpViewer = new Viewer(this, mpMapDrawer);
        mpViewerThread = new std::thread(&Viewer::Run, mpViewer);
        mpTracker->SetViewer(mpViewer);
        mpLocalMapper->SetViewer(mpViewer);
    }
}

void CoreSystem::takeNewFrame(cv::Mat img, cv::Mat depth, const double timeStamp)
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

void CoreSystem::writeTrajectoryToFile(const std::string &filename)
{
    std::vector<KeyFrame *> vpKFs = mpMapManager->GetActiveMap()->GetAllKeyFrames();
    sort(vpKFs.begin(), vpKFs.end(), [&](KeyFrame *l, KeyFrame *r) { return l->mnId < r->mnId; });

    // Transform all keyframes so that the first keyframe is at the origin.
    // After a loop closure the first keyframe might not be at the origin.
    Sophus::SE3d Tow = vpKFs[0]->GetPose();

    std::ofstream file(filename);
    file << std::fixed;

    // Frame pose is stored relative to its reference keyframe (which is optimized by BA and pose graph).
    // We need to get first the keyframe pose and then concatenate the relative transformation.
    // Frames not localized (tracking failure) are not saved.

    // For each frame we have a reference keyframe (lRit), the timestamp (lT) and a flag
    // which is true when tracking failed (lbL).
    std::list<KeyFrame *>::iterator lRit = mpTracker->mlpReferences.begin();
    std::list<double>::iterator lT = mpTracker->mlFrameTimes.begin();
    std::list<bool>::iterator lbL = mpTracker->mlbLost.begin();
    for (auto lit = mpTracker->mlRelativeFramePoses.begin(),
              lend = mpTracker->mlRelativeFramePoses.end();
         lit != lend; lit++, lRit++, lT++, lbL++)
    {
        if (*lbL)
            continue;

        KeyFrame *pKF = *lRit;
        Sophus::SE3d Trw;

        // If the reference keyframe was culled, traverse the spanning tree to get a suitable keyframe.
        while (pKF->isBad())
        {
            Trw = pKF->mTcp;
            pKF = pKF->GetParent();
        }

        Trw = Tow * pKF->GetPose() * Trw;

        Sophus::SE3d Tcw = Trw * (*lit);
        Eigen::Matrix3d Rwc = Tcw.rotationMatrix();
        Eigen::Vector3d twc = Tcw.translation();

        Eigen::Quaterniond q(Rwc);

        file << std::setprecision(6) << *lT << " "
             << std::setprecision(9)
             << twc[0] << " "
             << twc[1] << " "
             << twc[2] << " "
             << q.x() << " "
             << q.y() << " "
             << q.z() << " "
             << q.w() << std::endl;
    }

    file.close();
}

} // namespace slam