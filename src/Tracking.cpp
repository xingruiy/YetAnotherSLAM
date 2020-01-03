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
        meState = TrackingState::OK;
        printf("Map created with %lu points\n", mpMap->GetMapPointVec().size());

        mpTracker->SetReferenceImage(mCurrentFrame.mImGray);
        mpTracker->SetReferenceDepth(mCurrentFrame.mImDepth);
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

    int nMapPointObs = 0;
    int nMapPointNotObs = 0;

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
            float z = mCurrentFrame.mImDepth.at<float>(cv::Point2f(warpedX, warpedY));

            if (std::abs(warpedZ - z) < 0.1)
            {
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

                nMapPointObs++;
            }
            else
                nMapPointNotObs++;
        }
    }

    mObs = vKeyPointsWarped.size();
    mCurrentFrame.mvObsKeys = vKeyPointsWarped;
    mCurrentFrame.mvObsMapPoints = vpObsMapPoints;
    float obsWidth = maxX - minX;
    float obsHeight = maxY - minY;
    mObsRatio = obsWidth * obsHeight / (Frame::width * Frame::height);
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
        mpViewer->SetCurrentCameraPose(mCurrentFrame.mTcw.matrix());

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
    if (mObs < 200 || mObsRatio <= 0.4)
        return true;

    return false;
}

void Tracking::CreateNewKeyFrame()
{
    mCurrentFrame.ExtractORB();

    if (mCurrentFrame.N > 300)
    {
        // Check map points
        const size_t nObs = mCurrentFrame.mvObsMapPoints.size();
        const size_t nKPs = mCurrentFrame.mvKeys.size();

        // for (int i = 0; i < nObs; ++i)
        // {
        //     Sophus::SE3d Twc = mCurrentFrame.mTcw.inverse();
        //     MapPoint *pMP = mCurrentFrame.mvObsMapPoints[i];
        //     if (pMP)
        //     {
        //         Eigen::Vector3d ptTransformed = Twc * pMP->mWorldPos;
        //         float x = Frame::fx * ptTransformed(0) / ptTransformed(2) + Frame::cx;
        //         float y = Frame::fy * ptTransformed(1) / ptTransformed(2) + Frame::cy;
        //         float z = ptTransformed(2);

        //         for (int j = 0; j < nKPs; ++j)
        //         {
        //             cv::KeyPoint kp = mCurrentFrame.mvKeys[j];
        //             double dist = (Eigen::Vector2d(x, y) - Eigen::Vector2d(kp.pt.x, kp.pt.y)).norm();
        //             if (dist < 2)
        //             {
        //                 cv::Mat KPDesc = mCurrentFrame.mDescriptors.row(j);
        //                 Descriptor
        //             }
        //         }
        //     }
        // }

        // Create a new keyframe
        KeyFrame *pKF = new KeyFrame(mCurrentFrame, mpMap);
        mpReferenceKF = pKF;
        mCurrentFrame.mpReferenceKF = pKF;

        // We sort points by the measured depth by the RGBD sensor.
        // We create all those MapPoints whose depth < mThDepth.
        // If there are less than 100 close points we create the 100 closest.
        vector<pair<float, int>> vDepthIdx;
        vDepthIdx.reserve(mCurrentFrame.N);
        for (int i = 0; i < mCurrentFrame.N; i++)
        {
            float z = mCurrentFrame.mvDepth[i];
            if (z > 0)
            {
                vDepthIdx.push_back(make_pair(z, i));
            }
        }

        if (!vDepthIdx.empty())
        {
            sort(vDepthIdx.begin(), vDepthIdx.end());

            int nPoints = 0;
            for (size_t j = 0; j < vDepthIdx.size(); j++)
            {
                int i = vDepthIdx[j].second;

                bool bCreateNew = false;

                MapPoint *pMP = mCurrentFrame.mvpMapPoints[i];
                if (!pMP)
                    bCreateNew = true;
                else if (pMP->Observations() < 1)
                {
                    bCreateNew = true;
                    mCurrentFrame.mvpMapPoints[i] = static_cast<MapPoint *>(NULL);
                }

                if (bCreateNew)
                {
                    Eigen::Vector3d x3D;
                    cv::KeyPoint kp = mCurrentFrame.mvKeys[i];
                    const float x = kp.pt.x;
                    const float y = kp.pt.y;
                    const float z = mCurrentFrame.mvDepth[i];
                    x3D(0) = (x - Frame::cx) * Frame::invfx * z;
                    x3D(1) = (y - Frame::cy) * Frame::invfy * z;
                    x3D(2) = z;
                    x3D = pKF->mTcw * x3D;
                    MapPoint *pNewMP = new MapPoint(x3D, pKF, mpMap);
                    pKF->mvpMapPoints[i] = pNewMP;
                    mpMap->AddMapPoint(pNewMP);
                }
                else
                {
                    nPoints++;
                }

                if (vDepthIdx[j].first > mThDepth && nPoints > 100)
                    break;
            }
        }
    }

    //   mpLocalMapper->InsertKeyFrame(pKF);
}

void Tracking::SetViewer(Viewer *pViewer)
{
    mpViewer = pViewer;
}

void Tracking::Reset()
{
    meState = TrackingState::NOTInit;
    mLastFrame.mTcw = Sophus::SE3d();
}