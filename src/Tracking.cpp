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

    mnImageWidth = fSettings["Camera.width"];
    mnImageHeight = fSettings["Camera.height"];

    mK = Eigen::Matrix3d::Identity();
    mK(0, 0) = fx;
    mK(1, 1) = fy;
    mK(0, 2) = cx;
    mK(1, 2) = cy;

    int nFeatures = fSettings["ORBextractor.nFeatures"];
    float fScaleFactor = fSettings["ORBextractor.scaleFactor"];
    int nLevels = fSettings["ORBextractor.nLevels"];
    int fIniThFAST = fSettings["ORBextractor.iniThFAST"];
    int fMinThFAST = fSettings["ORBextractor.minThFAST"];

    mpORBextractor = new ORB_SLAM2::ORBextractor(nFeatures, fScaleFactor, nLevels, fIniThFAST, fMinThFAST);
    mpMapper = new DenseMapping(mnImageWidth, mnImageHeight, mK);

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

    mpTracker = new DenseTracking(mnImageWidth, mnImageHeight, mK, nNumPyr, vIterations, bUseRGB, bUseDepth);
}

void Tracking::TrackImageRGBD(const cv::Mat &imGray, const cv::Mat &imDepth)
{
    mCurrentFrame = Frame(imGray, imDepth, 0, mK, mpORBextractor, mpORBVocabulary);

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
            int nPt = CheckObservations();

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
    if (mCurrentFrame.N > 500)
    {
        mCurrentFrame.mTcw = Sophus::SE3d(Eigen::Matrix4d::Identity());

        // Create KeyFrame
        KeyFrame *pKFini = new KeyFrame(mCurrentFrame, mpMap);

        // Insert KeyFrame in the map
        mpMap->AddKeyFrame(pKFini);

        for (int i = 0; i < mCurrentFrame.N; i++)
        {
            float z = mCurrentFrame.mvDepth[i];
            if (z > 0)
            {
                Eigen::Vector3d x3D;
                cv::KeyPoint kp = mCurrentFrame.mvKeys[i];
                const float x = kp.pt.x;
                const float y = kp.pt.y;
                x3D(0) = (x - mCurrentFrame.cx) * mCurrentFrame.invfx * z;
                x3D(1) = (y - mCurrentFrame.cy) * mCurrentFrame.invfy * z;
                x3D(2) = z;
                MapPoint *pNewMP = new MapPoint(x3D, pKFini, mpMap);
                mCurrentFrame.mvpMapPoints[i] = pNewMP;
                mpMap->AddMapPoint(pNewMP);
            }
        }

        mpReferenceKF = pKFini;
        mvpLocalKeyFrames.push_back(pKFini);
        mvpLocalMapPoints = mpMap->GetMapPointVec();
        mCurrentFrame.mpReferenceKF = pKFini;

        mpTracker->SetReferenceImage(mCurrentFrame.mImGray);
        mpTracker->SetReferenceDepth(mCurrentFrame.mImDepth);
        meState = TrackingState::OK;

        printf("Map created with %lu points\n", mvpLocalMapPoints.size());
    }
}

int Tracking::CheckObservations()
{
    Sophus::SE3d TRefCurr;
    TRefCurr = mCurrentFrame.mTcw.inverse() * mpReferenceKF->mTcw;
    vector<cv::KeyPoint> vKeyPointWarped;

    for (int i = 0; i < mpReferenceKF->mvKeys.size(); ++i)
    {
        MapPoint *pMP = mpReferenceKF->mvpMapPoints[i];
        if (pMP == NULL)
            continue;

        Eigen::Vector3d ptTransformed = TRefCurr * pMP->mWorldPos;
        float warpedX = Frame::fx * ptTransformed(0) / ptTransformed(2) + Frame::cx;
        float warpedY = Frame::fy * ptTransformed(1) / ptTransformed(2) + Frame::cy;
        float warpedZ = ptTransformed(2);

        cv::KeyPoint Key = mpReferenceKF->mvKeys[i];
        Key.pt = cv::Point2f(warpedX, warpedY);
        vKeyPointWarped.push_back(Key);
    }

    std::cout << vKeyPointWarped.size() << std::endl;
}

void Tracking::UpdateLocalMap()
{
}

void Tracking::UpdateLocalPoints()
{
}

void Tracking::UpdateLocalKeyFrames()
{
}

bool Tracking::TrackLastFrame()
{
    mpTracker->SetTrackingImage(mCurrentFrame.mImGray);
    mpTracker->SetTrackingDepth(mCurrentFrame.mImDepth);

    Sophus::SE3d Tpc = mpTracker->GetTransform();

    mCurrentFrame.mTcw = mLastFrame.mTcw * Tpc.inverse();

    // Update viewer if applicable
    if (mpViewer)
    {
        mpViewer->SetCurrentCameraPose(mCurrentFrame.mTcw.matrix());
    }

    return true;
}

bool Tracking::Relocalization()
{
}

bool Tracking::TrackLocalMap()
{
}

bool Tracking::NeedNewKeyFrame()
{
}

void Tracking::CreateNewKeyFrame()
{
}

void Tracking::SetViewer(Viewer *pViewer)
{
    mpViewer = pViewer;
}