#include "Tracking.h"
#include "ORBMatcher.h"
#include "Optimizer.h"
#include "Sim3Solver.h"

namespace SLAM
{

Tracking::Tracking(System *pSystem, ORBVocabulary *pVoc, Map *pMap, KeyFrameDatabase *pKFDB)
    : mState(SYSTEM_NOT_READY), mpORBVocabulary(pVoc), mpKeyFrameDB(pKFDB), mpSystem(pSystem),
      mpCurrentKeyFrame(NULL), mpLastKeyFrame(NULL), mpMap(pMap), mnLastRelocFrameId(0)
{
    int w = g_width[0];
    int h = g_height[0];
    Eigen::Matrix3f calib = g_calib[0];

    mpTracker = new RGBDTracking(w, h, calib.cast<double>(), g_bUseColour, g_bUseDepth);
    mpMeshEngine = new MeshEngine(20000000);
    mpRayTraceEngine = new RayTraceEngine(w, h, g_calib[0]);
    mpORBExtractor = new ORBextractor(g_ORBNFeatures, g_ORBScaleFactor, g_ORBNLevels, g_ORBIniThFAST, g_ORBMinThFAST);
}

void Tracking::SetLocalMapper(LocalMapping *pLocalMapper)
{
    mpLocalMapper = pLocalMapper;
}

void Tracking::SetLoopClosing(LoopClosing *pLoopClosing)
{
    mpLoopClosing = pLoopClosing;
}

void Tracking::SetViewer(Viewer *pViewer)
{
    mpViewer = pViewer;
}

void Tracking::GrabImageRGBD(cv::Mat ImgGray, cv::Mat ImgDepth, const double TimeStamp)
{
    mCurrentFrame = Frame(ImgGray, ImgDepth, TimeStamp, mpORBExtractor, mpORBVocabulary);

    Track();
}

void Tracking::Track()
{
    bool bOK = false;
    switch (mState)
    {
    case SYSTEM_NOT_READY:
        StereoInitialization();
        if (mState != OK)
            return;
        break;

    case OK:
        bOK = TrackRGBD();

        if (bOK)
        {
            if (NeedNewKeyFrame())
                CreateNewKeyFrame();
        }
        else
        {
            bOK = Relocalization();
            if (!bOK)
            {
                mState = LOST;
            }
        }
        break;

    case LOST:
        bOK = Relocalization();

        if (bOK)
        {
            mState = OK;
            break;
        }
        else
            return;
    }

    mLastFrame = Frame(mCurrentFrame);
}

void Tracking::StereoInitialization()
{
    mCurrentFrame.ExtractORB();
    if (mCurrentFrame.N > 500)
    {
        // Set Frame pose to the origin
        mCurrentFrame.SetPose(Sophus::SE3d(Eigen::Matrix4d::Identity()));

        // Create KeyFrame
        KeyFrame *pKFini = new KeyFrame(mCurrentFrame, mpMap, mpKeyFrameDB);

        // Insert KeyFrame in the map
        mpMap->AddKeyFrame(pKFini);

        // Create MapPoints and asscoiate to KeyFrame
        for (int i = 0; i < mCurrentFrame.N; i++)
        {
            float z = mCurrentFrame.mvDepth[i];
            if (z > 0)
            {
                Eigen::Vector3d x3D = pKFini->UnprojectStereo(i);
                MapPoint *pNewMP = new MapPoint(x3D, pKFini, mpMap);
                pNewMP->AddObservation(pKFini, i);
                pKFini->AddMapPoint(pNewMP, i);
                pNewMP->ComputeDistinctiveDescriptors();
                pNewMP->UpdateNormalAndDepth();
                mpMap->AddMapPoint(pNewMP);

                mCurrentFrame.mvpMapPoints[i] = pNewMP;
            }
        }

        std::cout << "New map created with " << mpMap->MapPointsInMap() << " points" << std::endl;

        mpLocalMapper->InsertKeyFrame(pKFini);

        mLastFrame = Frame(mCurrentFrame);
        mpLastKeyFrame = pKFini;

        mvpLocalKeyFrames.push_back(pKFini);
        mvpLocalMapPoints = mpMap->GetAllMapPoints();
        mpReferenceKF = pKFini;
        mCurrentFrame.mpReferenceKF = pKFini;

        mpMap->SetReferenceMapPoints(mvpLocalMapPoints);

        mpMap->mvpKeyFrameOrigins.push_back(pKFini);

        mState = OK;

        // Create dense map struct
        mpCurrentMapStruct = new MapStruct(g_calib[0]);
        mpCurrentMapStruct->SetMeshEngine(mpMeshEngine);
        mpCurrentMapStruct->SetRayTraceEngine(mpRayTraceEngine);
        mpCurrentMapStruct->create(5000, 4000, 4500, 0.01, 0.03);
        mpCurrentMapStruct->Reset();

        pKFini->mpVoxelStruct = mpCurrentMapStruct;
        pKFini->mbVoxelStructMarginalized = false;

        // Set up Dense Tracker
        mpTracker->SetReferenceImage(mCurrentFrame.mImGray);
        mpTracker->SetReferenceDepth(mCurrentFrame.mImDepth);

        // Fuse the first depth image
        mRawDepth.upload(mCurrentFrame.mImDepth);
        mpCurrentMapStruct->Fuse(mRawDepth, mCurrentFrame.mTcw);

        mpViewer->setKeyFrameImage(mCurrentFrame.mImGray,
                                   mCurrentFrame.mvKeys);
    }
}

bool Tracking::TrackRGBD()
{
    Sophus::SE3d Tmw = mpCurrentMapStruct->mTcw;
    Sophus::SE3d Tcm = Tmw.inverse() * mLastFrame.mTcw;

    mpCurrentMapStruct->RayTrace(Tcm);
    auto vmap = mpCurrentMapStruct->GetRayTracingResult();
    mpTracker->SetReferenceMap(vmap);

    // Set tracking frames
    mpTracker->SetTrackingImage(mCurrentFrame.mImGray);
    mpTracker->SetTrackingDepth(mCurrentFrame.mImDepth);

    // Calculate the relateive transformation
    Sophus::SE3d DT = mpTracker->GetTransform(Sophus::SE3d(), true);

    mCurrentFrame.mTcw = mLastFrame.mTcw * DT.inverse();
    mCurrentFrame.mTcp = mLastFrame.mTcp * DT.inverse();

    mRawDepth.upload(mCurrentFrame.mImDepth);
    mpCurrentMapStruct->Fuse(mRawDepth, mCurrentFrame.mTcp);
    g_nTrackedFrame++;

    if (mpViewer)
        mpViewer->setLivePose(mCurrentFrame.mTcw.matrix());

    return true;
}

bool Tracking::Relocalization()
{
    // Compute Bag of Words Vector
    mCurrentFrame.ComputeBoW();

    // Relocalization is performed when tracking is lost
    // Track Lost: Query KeyFrame Database for keyframe candidates for relocalisation
    std::vector<KeyFrame *> vpCandidateKFs = mpKeyFrameDB->DetectRelocalizationCandidates(&mCurrentFrame);

    if (vpCandidateKFs.empty())
        return false;

    mCurrentFrame.CreateRelocalisationPoints();

    const int nKFs = vpCandidateKFs.size();

    // We perform first an ORB matching with each candidate
    // If enough matches are found we setup a PnP solver
    ORBMatcher matcher(0.75, true);

    std::vector<Sim3Solver *> vpSim3Solvers(nKFs);
    std::vector<std::vector<MapPoint *>> vvpMapPointsMatches(nKFs);
    std::vector<bool> vbDiscarded(nKFs);

    int nCandidates = 0;
    for (int iKF = 0; iKF < nKFs; ++iKF)
    {
        KeyFrame *pKF = vpCandidateKFs[iKF];
        if (pKF->isBad())
        {
            vbDiscarded[iKF] = true;
        }
        else
        {
            int nmatches = matcher.SearchByBoW(pKF, mCurrentFrame, vvpMapPointsMatches[iKF]);

            if (nmatches < 30)
            {
                vbDiscarded[iKF] = true;
                continue;
            }
            else
            {
                //  TODO: set up a ransac pose solver
                Sim3Solver *pSolver = new Sim3Solver(&mCurrentFrame, pKF, vvpMapPointsMatches[iKF]);
                pSolver->SetRansacParameters(0.99, 20, 300);
                vpSim3Solvers[iKF] = pSolver;
            }

            nCandidates++;
        }
    }

    bool bMatch = false;

    // for (int i = 0; i < nCandidates; i++)
    // {
    //     if (vbDiscarded[i])
    //         continue;

    //     // Perform 5 Ransac Iterations
    //     std::vector<bool> vbInliers;
    //     int nInliers;
    //     bool bNoMore;

    //     Sim3Solver *pSolver = vpSim3Solvers[i];

    //     Sophus::SE3d T12;
    //     bool found = pSolver->iterate(5, bNoMore, vbInliers, nInliers, T12);

    //     // If Ransac reachs max. iterations discard keyframe
    //     if (bNoMore)
    //     {
    //         vbDiscarded[i] = true;
    //         nCandidates--;
    //     }

    //     // If RANSAC returns a Sim3, perform a guided matching and optimize with all correspondences
    //     if (found)
    //     {
    //         std::vector<MapPoint *> vpMapPointMatches(vvpMapPointMatches[i].size(), static_cast<MapPoint *>(NULL));
    //         for (size_t j = 0, jend = vbInliers.size(); j < jend; j++)
    //         {
    //             if (vbInliers[j])
    //                 vpMapPointMatches[j] = vvpMapPointMatches[i][j];
    //         }

    //         matcher.SearchBySim3(mpCurrentKF, pKF, vpMapPointMatches, T12, 7.5);

    //         // gScm here should be the inverse of T12, i.e. 2->1
    //         Sophus::SE3d T21 = T12.inverse();
    //         g2o::Sim3 gScm(T21.rotationMatrix(), T21.translation(), 1.0);

    //         const int nInliers = Optimizer::OptimizeSim3(mpCurrentKF, pKF, vpMapPointMatches, gScm, 10, true);

    //         // If optimization is succesful stop ransacs and continue
    //         if (nInliers >= 20)
    //         {
    //             bMatch = true;
    //             mpMatchedKF = pKF;
    //             Sophus::SE3d T21(gScm.rotation(), gScm.translation());
    //             Sophus::SE3d Twc = pKF->GetPoseInverse();

    //             mTcwNew = pKF->GetPose() * T21.inverse();

    //             mvpCurrentMatchedPoints = vpMapPointMatches;
    //             break;
    //         }
    //     }
    // }
}

bool Tracking::TrackLocalMap()
{
    // retrieve the local map and try to
    // find matches to points in the local map.
    UpdateLocalMap();

    SearchLocalPoints();

    // Optimize Pose
    Optimizer::PoseOptimization(mCurrentFrame);
    int mnMatchesInliers = 0;

    // Update MapPoints Statistics
    for (int i = 0; i < mCurrentFrame.N; i++)
    {
        if (mCurrentFrame.mvpMapPoints[i])
        {
            if (!mCurrentFrame.mvbOutlier[i])
            {
                mCurrentFrame.mvpMapPoints[i]->IncreaseFound();
                mCurrentFrame.mvpMapPoints[i]->mnLastFrameSeen = mCurrentFrame.mnId;
                // if (!mbOnlyTracking)
                // {
                //     if (mCurrentFrame.mvpMapPoints[i]->Observations() > 0)
                //         mnMatchesInliers++;
                // }
                // else
                mnMatchesInliers++;
            }
            else
            {
                mCurrentFrame.mvbOutlier[i] = false;
                mCurrentFrame.mvpMapPoints[i] = nullptr;
            }
        }
    }

    // Decide if the tracking was succesful
    // More restrictive if there was a relocalization recently
    if (mCurrentFrame.mnId < mnLastRelocFrameId + mMaxFrames && mnMatchesInliers < 50)
        return false;

    if (mnMatchesInliers < 30)
        return false;
    else
        return true;
}

void Tracking::SearchLocalPoints()
{
    int nToMatch = 0;

    // Project points in frame and check its visibility
    for (auto vit = mvpLocalMapPoints.begin(), vend = mvpLocalMapPoints.end(); vit != vend; vit++)
    {
        MapPoint *pMP = *vit;

        if (pMP->isBad())
            continue;
        // Project (this fills MapPoint variables for matching)
        if (mCurrentFrame.isInFrustum(pMP, 0.5))
        {
            pMP->IncreaseVisible();
            nToMatch++;
        }
    }

    if (nToMatch > 0)
    {
        ORBMatcher matcher(0.8);
        int th = 1;
        // If the camera has been relocalised recently, perform a coarser search
        if (mCurrentFrame.mnId < mnLastRelocFrameId + 2)
            th = 3;

        matcher.SearchByProjection(mCurrentFrame, mvpLocalMapPoints, th);
    }
}

void Tracking::UpdateLocalMap()
{
    // Update
    UpdateLocalKeyFrames();
    UpdateLocalPoints();

    // This is for visualization
    mpMap->SetReferenceMapPoints(mvpLocalMapPoints);
}

void Tracking::UpdateLocalPoints()
{
    mvpLocalMapPoints.clear();

    for (std::vector<KeyFrame *>::const_iterator itKF = mvpLocalKeyFrames.begin(), itEndKF = mvpLocalKeyFrames.end(); itKF != itEndKF; itKF++)
    {
        KeyFrame *pKF = *itKF;
        const std::vector<MapPoint *> vpMPs = pKF->GetMapPointMatches();

        for (std::vector<MapPoint *>::const_iterator itMP = vpMPs.begin(), itEndMP = vpMPs.end(); itMP != itEndMP; itMP++)
        {
            MapPoint *pMP = *itMP;
            if (!pMP)
                continue;
            if (pMP->mnTrackReferenceForFrame == mCurrentFrame.mnId)
                continue;
            if (!pMP->isBad())
            {
                mvpLocalMapPoints.push_back(pMP);
                pMP->mnTrackReferenceForFrame = mCurrentFrame.mnId;
            }
        }
    }
}

void Tracking::UpdateLocalKeyFrames()
{
    // Each map point vote for the keyframes in which it has been observed
    std::map<KeyFrame *, int> keyframeCounter;
    auto PointSeeds = mpReferenceKF->GetMapPointMatches();
    for (int i = 0; i < PointSeeds.size(); i++)
    {
        if (PointSeeds[i])
        {
            MapPoint *pMP = PointSeeds[i];
            if (!pMP->isBad())
            {
                const std::map<KeyFrame *, size_t> observations = pMP->GetObservations();
                for (auto it = observations.begin(), itend = observations.end(); it != itend; it++)
                    keyframeCounter[it->first]++;
            }
            else
                mpReferenceKF->AddMapPoint(nullptr, i);
        }
    }

    if (keyframeCounter.empty())
        return;

    int max = 0;
    KeyFrame *pKFmax = static_cast<KeyFrame *>(NULL);

    mvpLocalKeyFrames.clear();
    mvpLocalKeyFrames.reserve(3 * keyframeCounter.size());

    // All keyframes that observe a map point are included in the local map. Also check which keyframe shares most points
    for (auto it = keyframeCounter.begin(), itEnd = keyframeCounter.end(); it != itEnd; it++)
    {
        KeyFrame *pKF = it->first;

        if (pKF->isBad())
            continue;

        if (it->second > max)
        {
            max = it->second;
            pKFmax = pKF;
        }

        mvpLocalKeyFrames.push_back(it->first);
        pKF->mnTrackReferenceForFrame = mCurrentFrame.mnId;
    }

    // Include also some not-already-included keyframes that are neighbors to already-included keyframes
    for (auto itKF = mvpLocalKeyFrames.begin(), itEndKF = mvpLocalKeyFrames.end(); itKF != itEndKF; itKF++)
    {
        // Limit the number of keyframes
        if (mvpLocalKeyFrames.size() > 80)
            break;

        KeyFrame *pKF = *itKF;

        const auto vNeighs = pKF->GetBestCovisibilityKeyFrames(10);

        for (auto itNeighKF = vNeighs.begin(), itEndNeighKF = vNeighs.end(); itNeighKF != itEndNeighKF; itNeighKF++)
        {
            KeyFrame *pNeighKF = *itNeighKF;
            if (!pNeighKF->isBad())
            {
                if (pNeighKF->mnTrackReferenceForFrame != mCurrentFrame.mnId)
                {
                    mvpLocalKeyFrames.push_back(pNeighKF);
                    pNeighKF->mnTrackReferenceForFrame = mCurrentFrame.mnId;
                    break;
                }
            }
        }

        const std::set<KeyFrame *> spChilds = pKF->GetChilds();
        for (auto sit = spChilds.begin(), send = spChilds.end(); sit != send; sit++)
        {
            KeyFrame *pChildKF = *sit;
            if (!pChildKF->isBad())
            {
                if (pChildKF->mnTrackReferenceForFrame != mCurrentFrame.mnId)
                {
                    mvpLocalKeyFrames.push_back(pChildKF);
                    pChildKF->mnTrackReferenceForFrame = mCurrentFrame.mnId;
                    break;
                }
            }
        }

        KeyFrame *pParent = pKF->GetParent();
        if (pParent)
        {
            if (pParent->mnTrackReferenceForFrame != mCurrentFrame.mnId)
            {
                mvpLocalKeyFrames.push_back(pParent);
                pParent->mnTrackReferenceForFrame = mCurrentFrame.mnId;
                break;
            }
        }
    }

    if (pKFmax)
    {
        mpReferenceKF = pKFmax;
        mCurrentFrame.mpReferenceKF = mpReferenceKF;
    }
}

bool Tracking::NeedNewKeyFrame()
{
    bool bCreateNew = false;
    Sophus::SE3d DT = mCurrentFrame.mTcp;

    if (DT.log().topRows<3>().norm() > 0.15)
        bCreateNew = true;
    else if (DT.log().bottomRows<3>().norm() > 0.15)
        bCreateNew = true;

    return bCreateNew;
}

void Tracking::CreateNewKeyFrame()
{
    if (!mpLocalMapper->SetNotStop(true))
        return;

    mCurrentFrame.mImDepth = cv::Mat(mpCurrentMapStruct->GetRayTracingResultDepth());
    mCurrentFrame.ExtractORB();
    mCurrentFrame.mTcw = mpReferenceKF->GetPose() * mCurrentFrame.mTcp;

    if (!TrackLocalMap())
        return;

    KeyFrame *pKF = new KeyFrame(mCurrentFrame, mpMap, mpKeyFrameDB);

    mpReferenceKF = pKF;
    mCurrentFrame.mpReferenceKF = pKF;

    // We sort points by the measured depth by the stereo/RGBD sensor.
    // We create all those MapPoints whose depth < mThDepth.
    // If there are less than 100 close points we create the 100 closest.
    std::vector<std::pair<float, int>> vDepthIdx;
    vDepthIdx.reserve(mCurrentFrame.N);
    for (int i = 0; i < mCurrentFrame.N; i++)
    {
        float z = mCurrentFrame.mvDepth[i];
        if (z > 0)
        {
            vDepthIdx.push_back(std::make_pair(z, i));
        }
    }

    if (!vDepthIdx.empty())
    {
        std::sort(vDepthIdx.begin(), vDepthIdx.end());

        int nPoints = 0;
        int nCreated = 0;
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
                mCurrentFrame.mvpMapPoints[i] = nullptr;
            }

            if (bCreateNew)
            {
                Eigen::Vector3d x3D = pKF->UnprojectStereo(i);
                MapPoint *pNewMP = new MapPoint(x3D, pKF, mpMap);
                pNewMP->AddObservation(pKF, i);
                pKF->AddMapPoint(pNewMP, i);
                pNewMP->ComputeDistinctiveDescriptors();
                pNewMP->UpdateNormalAndDepth();
                mpMap->AddMapPoint(pNewMP);

                mCurrentFrame.mvpMapPoints[i] = pNewMP;
                nPoints++;
                nCreated++;
            }
            else
            {
                nPoints++;
            }

            if (vDepthIdx[j].first > g_thDepth && nPoints > 100)
                break;
        }
    }

    mpLocalMapper->InsertKeyFrame(pKF);

    mpLocalMapper->SetNotStop(false);

    mnLastKeyFrameId = mCurrentFrame.mnId;
    mpLastKeyFrame = pKF;

    // Update Pose References
    mReferenceFramePose = mCurrentFrame.mTcw;
    mpCurrentMapStruct->SetActiveFlag(false);
    mpMap->AddMapStruct(mpCurrentMapStruct);

    // Create a new MapStruct
    mpCurrentMapStruct = new MapStruct(g_calib[0]);
    mpCurrentMapStruct->SetMeshEngine(mpMeshEngine);
    mpCurrentMapStruct->SetRayTraceEngine(mpRayTraceEngine);
    mpCurrentMapStruct->create(5000, 4000, 4500, 0.01, 0.03);
    mpCurrentMapStruct->Reset();
    mpCurrentMapStruct->SetPose(mCurrentFrame.mTcw);

    pKF->mpVoxelStruct = mpCurrentMapStruct;
    pKF->mbVoxelStructMarginalized = false;
    mpCurrentMapStruct->mTcw = pKF->GetPose();

    mCurrentFrame.mTcp = Sophus::SE3d();

    mpViewer->setKeyFrameImage(mCurrentFrame.mImGray,
                               mCurrentFrame.mvKeys);
}

void Tracking::reset()
{
    mState = SYSTEM_NOT_READY;
}

} // namespace SLAM