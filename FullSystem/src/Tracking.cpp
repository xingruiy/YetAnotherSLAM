#include "Tracking.h"
#include "ORBMatcher.h"
#include "Optimizer.h"
#include "PoseSolver.h"
#include "ImageProc.h"
#include "LocalBundler.h"

namespace SLAM
{

Tracking::Tracking(System *pSystem, ORBVocabulary *pVoc, Map *pMap, KeyFrameDatabase *pKFDB)
    : mState(SYSTEM_NOT_READY), mpORBVocabulary(pVoc), mpKeyFrameDB(pKFDB), mpSystem(pSystem),
      mpCurrentKeyFrame(nullptr), mpLastKeyFrame(nullptr), mpMap(pMap), mnLastRelocFrameId(0)
{
    int w = g_width[0];
    int h = g_height[0];
    Eigen::Matrix3f calib = g_calib[0];

    mpTracker = new RGBDTracking(w, h, calib.cast<double>(), g_bUseColour, g_bUseDepth);
    mpMeshEngine = new MeshEngine(20000000);
    mpRayTraceEngine = new RayTraceEngine(w, h, g_calib[0]);
    mpORBExtractor = new ORBextractor(g_ORBNFeatures, g_ORBScaleFactor, g_ORBNLevels, g_ORBIniThFAST, g_ORBMinThFAST);

    bundler = new LocalBundler(w, h, calib);
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
        std::cout << "tracking failed, trying to relocalise..." << std::endl;
        bOK = Relocalization();

        if (bOK)
        {
            mState = OK;
            break;
        }
    }

    mLastProcessedState = mState;
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

        bundler->AddKeyFrame(pKFini->mnId, mCurrentFrame.mImDepth, mCurrentFrame.mImGray, mCurrentFrame.mTcw);
        bundler->AllocatePoints();
    }
}

bool Tracking::TrackRGBD()
{
    Sophus::SE3d Tmw = mpCurrentMapStruct->mTcw;
    Sophus::SE3d Tcm = Tmw.inverse() * mLastFrame.mTcw;

    mpCurrentMapStruct->RayTrace(Tcm);
    auto vmap = mpCurrentMapStruct->GetRayTracingResult();
    mpTracker->SetReferenceModel(vmap);

    // Set tracking frames
    mpTracker->SetTrackingImage(mCurrentFrame.mImGray);
    mpTracker->SetTrackingDepth(mCurrentFrame.mImDepth);

    // Calculate the relateive transformation
    Sophus::SE3d DT = mpTracker->GetTransform(Sophus::SE3d(), true);

    if (DT.translation().norm() > 0.1)
    {
        std::cout << DT.matrix3x4() << std::endl;
        std::cout << "frame id: " << mCurrentFrame.mnId << std::endl;
        mpTracker->WriteDebugImages();
        return false;
    }

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
    // Extract Features
    mCurrentFrame.ExtractORB();
    if (mCurrentFrame.N < 500)
    {
        std::cout << "Relocalisation failed, Not enough features..." << std::endl;
        return false;
    }

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
    ORBMatcher matcher(0.75, false);

    std::vector<PoseSolver *> vpSim3Solvers(nKFs);
    std::vector<std::vector<MapPoint *>> vvpMapPointMatches(nKFs);
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
            int nmatches = matcher.SearchByBoW(mCurrentFrame, pKF, vvpMapPointMatches[iKF]);
            std::cout << "matches for kf: " << iKF << " : " << nmatches << std::endl;

            if (nmatches < 20)
            {
                vbDiscarded[iKF] = true;
                continue;
            }
            else
            {
                PoseSolver *pSolver = new PoseSolver(&mCurrentFrame, pKF, vvpMapPointMatches[iKF]);
                pSolver->SetRansacParameters(0.99, 20, 300);
                vpSim3Solvers[iKF] = pSolver;
            }

            nCandidates++;
        }
    }

    bool bMatch = false;
    KeyFrame *pMatchedKF = nullptr;

    ORBMatcher matcher2(0.9, true);
    while (nCandidates > 0 && !bMatch)
    {
        for (int i = 0; i < nKFs; i++)
        {
            if (vbDiscarded[i])
                continue;

            KeyFrame *pKF = vpCandidateKFs[i];

            if (!pKF || pKF->isBad())
            {
                vbDiscarded[i] = true;
                nCandidates--;
            }

            std::vector<bool> vbInliers;
            int nInliers;
            bool bNoMore = false;
            PoseSolver *pSolver = vpSim3Solvers[i];

            Sophus::SE3d Tcw;
            bool found = pSolver->iterate(5, bNoMore, vbInliers, nInliers, Tcw);

            // If Ransac reachs max. iterations discard keyframe
            if (bNoMore)
            {
                vbDiscarded[i] = true;
                nCandidates--;
            }

            // If RANSAC returns a SE3, perform a guided matching and optimize with all correspondences
            if (found)
            {
                std::set<MapPoint *> sFound;
                const int np = vbInliers.size();

                for (int j = 0; j < np; j++)
                {
                    if (vbInliers[j])
                    {
                        mCurrentFrame.mvpMapPoints[j] = vvpMapPointMatches[i][j];
                        sFound.insert(vvpMapPointMatches[i][j]);
                    }
                    else
                        mCurrentFrame.mvpMapPoints[j] = NULL;
                }

                int nGood = Optimizer::PoseOptimization(mCurrentFrame);
                std::cout << "after inital pose optimization: " << nGood << std::endl;
                if (nGood < 10)
                    continue;

                for (int io = 0; io < mCurrentFrame.N; io++)
                    if (mCurrentFrame.mvbOutlier[io])
                        mCurrentFrame.mvpMapPoints[io] = nullptr;

                // If few inliers, search by projection in a coarse window and optimize again
                if (nGood < 50)
                {
                    int nadditional = matcher2.SearchByProjection(mCurrentFrame, vpCandidateKFs[i], sFound, 10, 100);

                    if (nadditional + nGood >= 50)
                    {
                        nGood = Optimizer::PoseOptimization(mCurrentFrame);
                        std::cout << "nGood after 1st optimization...: " << nGood << std::endl;

                        // If many inliers but still not enough, search by projection again in a narrower window
                        // the camera has been already optimized with many points
                        if (nGood > 30 && nGood < 50)
                        {
                            sFound.clear();
                            for (int ip = 0; ip < mCurrentFrame.N; ip++)
                                if (mCurrentFrame.mvpMapPoints[ip])
                                    sFound.insert(mCurrentFrame.mvpMapPoints[ip]);
                            nadditional = matcher2.SearchByProjection(mCurrentFrame, vpCandidateKFs[i], sFound, 3, 64);
                            std::cout << "nGood before further optimization...: " << nGood + nadditional << std::endl;

                            // Final optimization
                            if (nGood + nadditional >= 50)
                            {
                                nGood = Optimizer::PoseOptimization(mCurrentFrame);

                                for (int io = 0; io < mCurrentFrame.N; io++)
                                    if (mCurrentFrame.mvbOutlier[io])
                                        mCurrentFrame.mvpMapPoints[io] = NULL;

                                std::cout << "nGood after further optimization...: " << nGood << std::endl;
                            }
                        }
                    }
                }

                // If the pose is supported by enough inliers stop ransacs and continue
                if (nGood >= 50)
                {
                    bMatch = true;
                    pMatchedKF = pKF;
                    break;
                }
            }
        }
    }

    if (!bMatch)
    {
        return false;
    }
    else
    {
        std::cout << "relocalisation success! " << std::endl;
        mpReferenceKF = pMatchedKF;
        mpCurrentMapStruct = pMatchedKF->mpVoxelStruct;
        mpCurrentMapStruct->DeleteMesh();
        mnLastRelocFrameId = mCurrentFrame.mnId;
        mCurrentFrame.mTcp = pMatchedKF->GetPoseInverse() * mCurrentFrame.mTcw;
        return true;
    }
}

bool Tracking::TrackLocalMap()
{
    // retrieve the local map and try to
    // find matches to points in the local map.
    UpdateLocalMap();

    SearchLocalPoints();

    // Optimize Pose
    // Optimizer::PoseOptimization(mCurrentFrame);
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
        if (mCurrentFrame.mnId - mnLastRelocFrameId <= 1)
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
    KeyFrame *pKFmax = nullptr;

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

    bundler->AddKeyFrame(pKF->mnId, mCurrentFrame.mImDepth, mCurrentFrame.mImGray, mCurrentFrame.mTcw);
    bundler->BundleAdjust();
    bundler->AllocatePoints();
    pKF->SetPose(bundler->GetLastKeyFramePose());
}

void Tracking::reset()
{
    mState = SYSTEM_NOT_READY;
}

} // namespace SLAM