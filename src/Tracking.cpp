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

    cv::Mat DistCoef(4, 1, CV_32F);
    DistCoef.at<float>(0) = fSettings["Camera.k1"];
    DistCoef.at<float>(1) = fSettings["Camera.k2"];
    DistCoef.at<float>(2) = fSettings["Camera.p1"];
    DistCoef.at<float>(3) = fSettings["Camera.p2"];
    const float k3 = fSettings["Camera.k3"];
    if (k3 != 0)
    {
        DistCoef.resize(5);
        DistCoef.at<float>(4) = k3;
    }

    DistCoef.copyTo(mDistCoef);

    float fps = fSettings["Camera.fps"];
    if (fps == 0)
        fps = 30;
    mMaxFrameRate = fps;

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
    mCurrentFrame = Frame(imGray, imDepth, 0, mK, mbf, mThDepth, mDistCoef, mpORBextractor, mpORBVocabulary);
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
        // Create the initial keyframe
        KeyFrame *pKFini = new KeyFrame(mCurrentFrame, mpMap);

        for (int i = 0; i < mCurrentFrame.N; i++)
        {
            float z = mCurrentFrame.mvDepth[i];

            if (z > 0)
            {
                Eigen::Vector3d x3D = pKFini->UnprojectKeyPoint(i);
                MapPoint *pNewMP = new MapPoint(x3D, mpMap, pKFini, i);
                pNewMP->AddObservation(pKFini, i);
                pKFini->mvpMapPoints[i] = pNewMP;
                mpMap->AddMapPoint(pNewMP);
            }
        }

        // Insert KeyFrame into the map
        mpMap->AddKeyFrame(pKFini);

        mpReferenceKF = pKFini;
        mCurrentFrame.mpReferenceKF = pKFini;
        meState = TrackingState::OK;
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
        pKF->mvpParentMPs.clear();

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
    }
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