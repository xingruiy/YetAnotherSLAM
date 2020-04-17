#include "FullSystem.h"
#include "Map.h"
#include "Frame.h"
#include "KeyFrame.h"
#include "Tracking.h"
#include "LocalMapping.h"
#include "GlobalSettings.h"
#include "LoopClosing.h"
#include "BaseIOWrapper.h"
#include "KeyFrameDatabase.h"
#include "ORBextractor.h"

namespace slam
{

FullSystem::FullSystem(const std::string &strSettingFile, const std::string &strVocFile)
{
    //Load Settings
    readSettings(strSettingFile);

    //Load ORB Vocabulary
    OrbVoc = new ORBVocabulary();
    OrbVoc->loadFromBinaryFile(strVocFile);

    OrbExt = new ORBextractor();

    //Create the Map
    mpMap = new Map();
    // mpMapDrawer = new MapDrawer(mpMap);

    //Create KeyFrame Database
    mpKeyFrameDB = new KeyFrameDatabase(*OrbVoc);

    loopCloser = new LoopClosing(mpMap, mpKeyFrameDB, OrbVoc);
    std::thread *thd = new std::thread(&LoopClosing::Run, loopCloser);
    allChildThreads.push_back(thd);

    localMapper = new LocalMapping(OrbVoc, mpMap);
    localMapper->SetLoopCloser(loopCloser);
    loopCloser->SetLocalMapper(localMapper);
    thd = new std::thread(&LocalMapping::Run, localMapper);
    allChildThreads.push_back(thd);

    //Initialize the Tracking thread
    mpTracker = new Tracking(this, OrbVoc, mpMap, mpKeyFrameDB);
    mpTracker->SetLocalMapper(localMapper);

    // if (g_bEnableViewer)
    // {
    //     mpViewer = new Viewer(this, mpMapDrawer);
    //     thd = new std::thread(&Viewer::Run, mpViewer);
    //     allChildThreads.push_back(thd);
    //     mpTracker->setIOWrapper(mpViewer);
    // }
}

FullSystem::~FullSystem()
{
    for (auto th : allChildThreads)
    {
        th->join();
        delete th;
    }

    delete mpMap;
    delete mpTracker;
    delete localMapper;
    delete loopCloser;
}

void FullSystem::addImages(cv::Mat img, cv::Mat depth, double ts)
{
    FrameMetaData *meta = new FrameMetaData();
    meta->id = allFrameHistory.size();
    meta->timestamp = ts;
    allFrameHistory.push_back(meta);

    // Frame *newF = new Frame(img, depth, ts, ORBext, OrbVoc);
    // newF->metaData = meta;
    // if (mpViewer)
    // {
    //     // mpViewer->setLiveImage(img);
    //     // mpViewer->setLiveDepth(depth);
    // }
    Frame currFrame = Frame(img, depth, OrbExt, OrbVoc);
    currFrame.meta = meta;

    mpTracker->trackNewFrame(currFrame);

    traceKeyFramePoints();
}

void FullSystem::traceKeyFramePoints()
{
}

void FullSystem::reset()
{
    mpTracker->reset();
    mpMap->reset();
    mpKeyFrameDB->clear();

    KeyFrame::nNextId = 0;
    MapPoint::nNextId = 0;

    for (auto meta : allFrameHistory)
        delete meta;

    allFrameHistory.clear();
    allKeyFramesHistory.clear();
}

void FullSystem::shutdown()
{
    std::cout << "shutdown called." << std::endl;
}

void FullSystem::addOutput(BaseIOWrapper *io)
{
    if (io)
    {
        io->setSystemIO(this);
        io->setGlobalMap(mpMap);
        outputs.push_back(io);
    }
}

void FullSystem::readSettings(const std::string &filename)
{
    cv::FileStorage file(filename, cv::FileStorage::READ);
    RUNTIME_ASSERT(file.isOpened());

    // read system configurations
    g_bEnableViewer = (int)file["FullSystem.EnableViewer"] == 1;
    g_bReverseRGB = (int)file["FullSystem.ReverseRGB"] == 1;
    g_DepthScaleInv = 1.0 / (double)file["FullSystem.DepthScale"];

    // read orb parameters
    g_ORBScaleFactor = file["ORB_SLAM2.scaleFactor"];
    g_ORBNFeatures = file["ORB_SLAM2.nFeatures"];
    g_ORBNLevels = file["ORB_SLAM2.nLevels"];
    g_ORBIniThFAST = file["ORB_SLAM2.iniThFAST"];
    g_ORBMinThFAST = file["ORB_SLAM2.minThFAST"];

    // read calibration parameters
    int width = file["Calibration.width"];
    int height = file["Calibration.height"];
    float fx = file["Calibration.fx"];
    float fy = file["Calibration.fy"];
    float cx = file["Calibration.cx"];
    float cy = file["Calibration.cy"];
    Eigen::Matrix3d calib;
    calib << fx, 0, cx, 0, fy, cy, 0, 0, 1.0;
    setGlobalCalibration(width, height, calib);

    // Update tracking parameters
    g_bf = file["Calibration.bf"];
    g_thDepth = g_bf * (float)file["Tracking.ThDepth"] / fx;
    g_bUseColour = (int)file["Tracking.UseColour"] == 1;
    g_bUseDepth = (int)file["Tracking.UseDepth"] == 1;

    // read distortion coefficients
    g_distCoeff = cv::Mat(4, 1, CV_32F);
    g_distCoeff.at<float>(0) = file["UnDistortion.k1"];
    g_distCoeff.at<float>(1) = file["UnDistortion.k2"];
    g_distCoeff.at<float>(2) = file["UnDistortion.p1"];
    g_distCoeff.at<float>(3) = file["UnDistortion.p2"];
    const float k3 = file["UnDistortion.k3"];
    if (k3 != 0)
    {
        g_distCoeff.resize(5);
        g_distCoeff.at<float>(4) = k3;
    }

    g_pointSize = file["Viewer.PointSize"];
    g_bSystemRunning = (int)file["Viewer.StartWhenReady"] == 1;

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

void FullSystem::FuseAllMapStruct()
{
    auto mapStructs = mpMap->GetAllVoxelMaps();
    MapStruct *pMSini = mpMap->mpMapStructOrigin;
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
        mpMap->EraseMapStruct(vit);
        std::cout << "fusing map: " << vit->mnId << std::endl;
    }

    pMSini->GenerateMesh();
    pMSini->SetActiveFlag(false);
}

void FullSystem::SaveTrajectoryTUM(const std::string &filename)
{
    std::vector<KeyFrame *> vpKFs = mpMap->GetAllKeyFrames();
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
            std::cout << "KF: " << pKF->mnId << std::endl;
            std::cout << Trw.matrix3x4() << std::endl;
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

void FullSystem::SaveKeyFrameTrajectoryTUM(const std::string &filename)
{
    std::vector<KeyFrame *> vpKFs = mpMap->GetAllKeyFrames();
    std::sort(vpKFs.begin(), vpKFs.end(), [&](KeyFrame *l, KeyFrame *r) { return l->mnId < r->mnId; });

    // Transform all keyframes so that the first keyframe is at the origin.
    // After a loop closure the first keyframe might not be at the origin.
    //cv::Mat Two = vpKFs[0]->GetPoseInverse();

    std::ofstream f;
    f.open(filename.c_str());
    f << std::fixed;

    for (size_t i = 0; i < vpKFs.size(); i++)
    {
        KeyFrame *pKF = vpKFs[i];

        // pKF->SetPose(pKF->GetPose()*Two);

        if (pKF->isBad())
            continue;

        // cv::Mat R = pKF->GetRotation().t();
        // std::vector<float> q = Converter::toQuaternion(R);
        // cv::Mat t = pKF->GetCameraCenter();
        Sophus::SE3d Tcw = pKF->GetPose();
        Eigen::Matrix3d R = Tcw.rotationMatrix();
        Eigen::Vector3d t = Tcw.translation();
        Eigen::Quaterniond q(R);

        f << std::setprecision(6) << pKF->timestamp << std::setprecision(7)
          << " " << t[0]
          << " " << t[1]
          << " " << t[2]
          << " " << q.x()
          << " " << q.y()
          << " " << q.z()
          << " " << q.w() << std::endl;
    }

    f.close();
    std::cout << std::endl
              << "trajectory saved!" << std::endl;
}

} // namespace slam