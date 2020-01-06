#include "Tracking.h"

Tracking::Tracking(const std::string &strSettingPath, FullSystem *pSys, Map *pMap, ORB_SLAM2::ORBVocabulary *pVoc)
    : mpFullSystem(pSys), mpMap(pMap), mpORBVocabulary(pVoc), mbOnlyTracking(false)
{
    meState = TrackingState::NOTInit;

    cv::FileStorage fSettings(strSettingPath, cv::FileStorage::READ);

    if (!fSettings.isOpened())
    {
        std::cout << "Reading file failed..." << std::endl;
        exit(-1);
    }

    double fx = fSettings["Camera.fx"];
    double fy = fSettings["Camera.fy"];
    double cx = fSettings["Camera.cx"];
    double cy = fSettings["Camera.cy"];

    mImgWidth = fSettings["Camera.width"];
    mImgHeight = fSettings["Camera.height"];

    mK = Eigen::Matrix3d::Identity();
    mK(0, 0) = fx;
    mK(1, 1) = fy;
    mK(0, 2) = cx;
    mK(1, 2) = cy;

    mbf = fSettings["Camera.bf"];
    mThDepth = mbf * (float)fSettings["ThDepth"] / fx;

    int nFeatures = fSettings["ORBextractor.nFeatures"];
    float fScaleFactor = fSettings["ORBextractor.scaleFactor"];
    int nLevels = fSettings["ORBextractor.nLevels"];
    int fIniThFAST = fSettings["ORBextractor.iniThFAST"];
    int fMinThFAST = fSettings["ORBextractor.minThFAST"];

    mpORBextractor = new ORB_SLAM2::ORBextractor(nFeatures, fScaleFactor, nLevels, fIniThFAST, fMinThFAST);
    mpMapper = new DenseMapping(mImgWidth, mImgHeight, mK);

    int nUseRGB = fSettings["Tracking.use_rgb"];
    int nNumPyr = fSettings["Tracking.num_pyr"];
    int nUseDepth = fSettings["Tracking.use_depth"];
    bool bUseRGB = (nUseRGB != 0);
    bool bUseDepth = (nUseDepth != 0);
    std::vector<int> vIterations(nNumPyr);
    for (int i = 0; i < nNumPyr; ++i)
    {
        std::string str = "Tracking.num_iter_lvl" + std::to_string(i + 1);
        int nIter = fSettings[str.c_str()];
        vIterations[i] = nIter;
    }

    mpTracker = new DenseTracking(mImgWidth, mImgHeight, mK, nNumPyr, vIterations, bUseRGB, bUseDepth);
}

void Tracking::TrackImageRGBD(const cv::Mat &imGray, const cv::Mat &imDepth)
{
    mCurrentFrame = Frame(imGray, imDepth, 0, mK, mbf, mThDepth, mpORBextractor, mpORBVocabulary);

    meLastState = meState;
    bool bOK = false;

    switch (meState)
    {
    case TrackingState::NOTInit:
    {
        // Try to initialize the system
        InitializeTracking();

        if (meState != TrackingState::OK)
            return;
    }

    case TrackingState::OK:
    {
        // System is initialized. Track Frame.

        if (!mbOnlyTracking)
        {
            bOK = TrackLastFrame();
        }
        else
        {
        }

        mCurrentFrame.mpReferenceKF = mpReferenceKF;

        if (bOK)
        {
            if (NeedNewKeyFrame())
                CreateNewKeyFrame();
        }
        else
        {
            meState = TrackingState::LOST;
        }

        break;
    }

    case TrackingState::LOST:
    {
        bool bOK = Relocalization();

        break;
    }

    default:
        std::cout << "Control flow shouldn't be reaching here..." << std::endl;
        return;
    }

    if (bOK)
        mLastFrame = Frame(mCurrentFrame);
}

void Tracking::InitializeTracking()
{
    mCurrentFrame.ExtractORB();

    if (mCurrentFrame.N > 500)
    {
        // Create KeyFrame
        KeyFrame *pKFini = new KeyFrame(mCurrentFrame, mpMap);

        for (int i = 0; i < mCurrentFrame.N; i++)
        {
            float z = mCurrentFrame.mvDepth[i];
            if (z > 0)
            {
                Eigen::Vector3d x3D;
                cv::KeyPoint kp = mCurrentFrame.mvKeys[i];
                const float x = kp.pt.x;
                const float y = kp.pt.y;
                x3D(0) = (x - pKFini->cx) * pKFini->invfx * z;
                x3D(1) = (y - pKFini->cy) * pKFini->invfy * z;
                x3D(2) = z;
                MapPoint *pNewMP = new MapPoint(x3D, mpMap, pKFini, i);
                pKFini->mvpMapPoints[i] = pNewMP;
                mpMap->AddMapPoint(pNewMP);
            }
        }

        // Insert KeyFrame into the map
        mpMap->AddKeyFrame(pKFini);

        mpReferenceKF = pKFini;
        mCurrentFrame.mpReferenceKF = pKFini;
        meState = TrackingState::OK;
        printf("Map created with %lu points\n", mpMap->GetMapPointVec().size());

        mpTracker->SetReferenceImage(mCurrentFrame.mImGray);
        mpTracker->SetReferenceDepth(mCurrentFrame.mImDepth);
    }
}

bool Tracking::TrackLastFrame()
{
    mpTracker->SetTrackingImage(mCurrentFrame.mImGray);
    mpTracker->SetTrackingDepth(mCurrentFrame.mImDepth);

    Sophus::SE3d Tpc = mpTracker->GetTransform();

    mCurrentFrame.mTcw = mLastFrame.mTcw * Tpc.inverse();

    // Update viewer if applicable
    if (mpViewer)
        mpViewer->SetCurrentCameraPose(mCurrentFrame.mTcw.matrix());

    return true;
}

bool Tracking::Relocalization()
{
}

bool Tracking::NeedNewKeyFrame()
{
    if (mbOnlyTracking)
        return false;

    if (!mpReferenceKF)
        return false;

    Sophus::SE3d DT = mpReferenceKF->mTcw.inverse() * mCurrentFrame.mTcw;

    std::cout << DT.log().topRows<3>().norm() << std::endl;
    if (DT.log().topRows<3>().norm() > 0.3)
        return true;

    if (DT.log().bottomRows<3>().norm() > 0.3)
        return true;

    // criteria 1: when observed points falls bellow a threshold
    // if (mObs < 200 || mObsRatio <= 0.4)
    //     return true;

    return false;
}

void Tracking::CreateNewKeyFrame()
{
    mCurrentFrame.ExtractORB();

    if (mCurrentFrame.N > 500)
    {
        KeyFrame *pKF = new KeyFrame(mCurrentFrame, mpMap);
        size_t nObsMPs = 0;

        for (int i = 0; i < mpReferenceKF->mvpMapPoints.size(); ++i)
        {
            MapPoint *pMP = mpReferenceKF->mvpMapPoints[i];
            if (pMP && pKF->IsInFrustum(pMP, 0.5))
            {
                pKF->mvpParentMPs.push_back(pMP);
                nObsMPs++;
            }
        }

        mpLocalMapper->InsertKeyFrame(pKF);
        mpReferenceKF = pKF;
        mCurrentFrame.mpReferenceKF = pKF;

        // // We sort points by the measured depth by the RGBD sensor.
        // // We create all those MapPoints whose depth < mThDepth.
        // // If there are less than 100 close points we create the 100 closest.
        // vector<pair<float, int>> vDepthIdx;
        // vDepthIdx.reserve(mCurrentFrame.N);
        // for (int i = 0; i < mCurrentFrame.N; i++)
        // {
        //     float z = mCurrentFrame.mvDepth[i];
        //     if (z > 0)
        //     {
        //         vDepthIdx.push_back(make_pair(z, i));
        //     }
        // }

        // if (!vDepthIdx.empty())
        // {
        //     sort(vDepthIdx.begin(), vDepthIdx.end());

        //     int nPoints = 0;
        //     for (size_t j = 0; j < vDepthIdx.size(); j++)
        //     {
        //         int i = vDepthIdx[j].second;

        //         bool bCreateNew = false;

        //         MapPoint *pMP = mCurrentFrame.mvpMapPoints[i];
        //         if (!pMP)
        //             bCreateNew = true;
        //         else if (pMP->Observations() < 1)
        //         {
        //             bCreateNew = true;
        //             mCurrentFrame.mvpMapPoints[i] = static_cast<MapPoint *>(NULL);
        //         }

        //         if (bCreateNew)
        //         {
        //             Eigen::Vector3d x3D;
        //             cv::KeyPoint kp = mCurrentFrame.mvKeys[i];
        //             const float x = kp.pt.x;
        //             const float y = kp.pt.y;
        //             const float z = mCurrentFrame.mvDepth[i];
        //             x3D(0) = (x - Frame::cx) * Frame::invfx * z;
        //             x3D(1) = (y - Frame::cy) * Frame::invfy * z;
        //             x3D(2) = z;
        //             x3D = pKF->mTcw * x3D;
        //             MapPoint *pNewMP = new MapPoint(x3D, mpMap, pKF, i);
        //             pKF->mvpMapPoints[i] = pNewMP;
        //             mpMap->AddMapPoint(pNewMP);
        //         }
        //         else
        //         {
        //             nPoints++;
        //         }

        //         if (vDepthIdx[j].first > mThDepth && nPoints > 100)
        //             break;
        //     }
        // }
    }

    // if (mCurrentFrame.N > 300)
    // {
    //     // Check map points
    //     const size_t nObs = mCurrentFrame.mvObsMapPoints.size();
    //     const size_t nKPs = mCurrentFrame.mvKeys.size();

    //     // Create a new keyframe
    //     KeyFrame *pKF = new KeyFrame(mCurrentFrame, mpMap);
    //     mpReferenceKF = pKF;
    //     mCurrentFrame.mpReferenceKF = pKF;

    // }
}

void Tracking::SetViewer(Viewer *pViewer)
{
    mpViewer = pViewer;
}

void Tracking::SetLocalMapper(LocalMapping *pLocalMapper)
{
    mpLocalMapper = pLocalMapper;
}

void Tracking::Reset()
{
    meState = TrackingState::NOTInit;
    mLastFrame.mTcw = Sophus::SE3d();
}