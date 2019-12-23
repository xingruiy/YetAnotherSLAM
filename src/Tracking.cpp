#include "Tracking.h"

Tracking::Tracking(const std::string &strSettingPath, FullSystem *pSys, Map *pMap, ORB_SLAM2::ORBVocabulary *pVoc)
    : mpFullSystem(pSys), mpMap(pMap), mpORBVocabulary(pVoc), mbOnlyTracking(false),
      mObs(1000), mObsRatio(1.0)
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
            CheckObservations();

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

        // Insert KeyFrame into the map
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
                x3D(0) = (x - Frame::cx) * Frame::invfx * z;
                x3D(1) = (y - Frame::cy) * Frame::invfy * z;
                x3D(2) = z;
                MapPoint *pNewMP = new MapPoint(x3D, pKFini, mpMap);
                pKFini->mvpMapPoints[i] = pNewMP;
                mpMap->AddMapPoint(pNewMP);
            }
        }

        mpReferenceKF = pKFini;
        mCurrentFrame.mpReferenceKF = pKFini;

        mpTracker->SetReferenceImage(mCurrentFrame.mImGray);
        mpTracker->SetReferenceDepth(mCurrentFrame.mImDepth);
        meState = TrackingState::OK;
        printf("Map created with %lu points\n", mvpLocalMapPoints.size());
    }
}

int Tracking::CheckObservations()
{
    if (mpReferenceKF == NULL)
    {
        std::cout << "error: no reference keyframe" << std::endl;
        exit(-1);
    }

    float minX = Frame::width;
    float minY = Frame::height;
    float maxX = -1;
    float maxY = -1;

    vector<cv::KeyPoint> vKeyPointsWarped;
    vector<MapPoint *> vpObsMapPoints;

    for (int i = 0; i < mpReferenceKF->N; ++i)
    {
        MapPoint *pMP = mpReferenceKF->mvpMapPoints[i];
        if (pMP == NULL)
            continue;

        Eigen::Vector3d ptTransformed = mCurrentFrame.mTcw.inverse() * pMP->mWorldPos;
        float warpedX = Frame::fx * ptTransformed(0) / ptTransformed(2) + Frame::cx;
        float warpedY = Frame::fy * ptTransformed(1) / ptTransformed(2) + Frame::cy;
        float warpedZ = ptTransformed(2);

        if (warpedX >= 0 && warpedY >= 0 && warpedX < Frame::width && warpedY < Frame::height)
        {
            cv::KeyPoint Key = mpReferenceKF->mvKeys[i];
            Key.pt = cv::Point2f(warpedX, warpedY);
            vKeyPointsWarped.push_back(Key);
            vpObsMapPoints.push_back(mpReferenceKF->mvpMapPoints[i]);

            if (warpedX < minX)
                minX = warpedX;
            if (warpedX > maxX)
                maxX = warpedX;
            if (warpedY < minY)
                minY = warpedY;
            if (warpedY > maxY)
                maxY = warpedY;
        }
    }

    mObs = vKeyPointsWarped.size();
    mCurrentFrame.mvObsKeys = vKeyPointsWarped;
    mCurrentFrame.mvObsMapPoints = vpObsMapPoints;
    // std::cout << "min-max: " << minX << "," << maxX << ";" << minY << "," << maxY << std::endl;
    float obsWidth = maxX - minX;
    float obsHeight = maxY - minY;
    mObsRatio = obsWidth * obsHeight / (Frame::width * Frame::height);
    // std::cout << mObs << ": " << mObsRatio << std::endl;
    // std::cout << mObsRatio << std::endl;

    // cv::Mat outImg;
    // cv::drawKeypoints(mCurrentFrame.mImGray, vKeyPointsWarped, outImg, cv::Scalar(0, 255, 0));
    // cv::imshow("Keypoints", outImg);
    // cv::waitKey(1);

    // return vKeyPointWarped.size();
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
    if (mbOnlyTracking)
        return false;

    // criteria 1: when observed points falls bellow a threshold
    if (mObs < 300 || mObsRatio <= 0.45)
        return true;

    return false;
}

void Tracking::CreateNewKeyFrame()
{
    mCurrentFrame.ExtractORB();
    if (mCurrentFrame.N > 500)
    {
        // Create KeyFrame
        KeyFrame *pKF = new KeyFrame(mCurrentFrame, mpMap);

        // Insert KeyFrame into the map
        mpMap->AddKeyFrame(pKF);

        for (int i = 0; i < mCurrentFrame.N; i++)
        {
            if (mCurrentFrame.mvpMapPoints[i])
                continue;

            float z = mCurrentFrame.mvDepth[i];
            if (z > 0)
            {
                Eigen::Vector3d x3D;
                cv::KeyPoint kp = mCurrentFrame.mvKeys[i];
                const float x = kp.pt.x;
                const float y = kp.pt.y;
                x3D(0) = (x - Frame::cx) * Frame::invfx * z;
                x3D(1) = (y - Frame::cy) * Frame::invfy * z;
                x3D(2) = z;
                x3D = pKF->mTcw * x3D;
                MapPoint *pNewMP = new MapPoint(x3D, pKF, mpMap);
                pKF->mvpMapPoints[i] = pNewMP;
                mpMap->AddMapPoint(pNewMP);
            }
        }

        mpReferenceKF = pKF;
        mCurrentFrame.mpReferenceKF = pKF;
    }
}

void Tracking::SetViewer(Viewer *pViewer)
{
    mpViewer = pViewer;
}