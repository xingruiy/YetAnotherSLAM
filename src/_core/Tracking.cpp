#include "Tracking.h"
#include "ORBMatcher.h"
#include "Optimizer.h"
#include "PoseSolver.h"
#include "Map.h"

namespace slam
{

Tracking::Tracking(System *pSystem, ORBVocabulary *pVoc, Map *mpMap, KeyFrameDatabase *pKFDB)
    : mState(SYSTEM_NOT_READY), ORBVoc(pVoc), mpKeyFrameDB(pKFDB), mpSystem(pSystem),
      mpLastKeyFrame(nullptr), mpMap(mpMap), mnLastSuccessRelocFrameId(0)
{
    int w = g_width[0];
    int h = g_height[0];
    Eigen::Matrix3f calib = g_calib[0];

    mpTracker = new CoarseTracking(w, h, calib, g_bUseColour, g_bUseDepth);
    mpMeshEngine = new MeshEngine(20000000);
    rayTracer = new RayTracer(w, h, g_calib[0]);
    mpORBExtractor = new ORBextractor(g_ORBNFeatures, g_ORBScaleFactor, g_ORBNLevels, g_ORBIniThFAST, g_ORBMinThFAST);
}

void Tracking::GrabImageRGBD(cv::Mat ImgGray, cv::Mat ImgDepth, const double TimeStamp)
{
    currFrame = Frame(ImgGray, ImgDepth, TimeStamp, mpORBExtractor, ORBVoc);
    // std::cout << "start tracking process.." << std::endl;
    Track();
}

void Tracking::Track()
{
    bool bOK = false;
    switch (mState)
    {
    case SYSTEM_NOT_READY:
        // std::cout << "initialize system" << std::endl;
        initSystem();
        if (mState != OK)
            return;

        mpTracker->setKeyFrame(currFrame.mImGray, currFrame.mImDepth);
        break;

    case OK:
        bOK = takeNewFrame();

        if (bOK)
        {
            if (NeedNewKeyFrame())
                CreateNewKeyFrame();
        }
        else
        {

            mState = LOST;
            return;
        }

        break;

    case LOST:
        std::cout << "tracking failed, trying to relocalise..." << std::endl;
        bOK = Relocalization();

        if (bOK)
        {
            mState = OK;
        }
        break;
    }

    mLastProcessedState = mState;
    lastFrame = Frame(currFrame);

    // std::cout << "strat writing pose" << std::endl;
    Sophus::SE3d Tcr = mpReferenceKF->GetPoseInverse() * currFrame.mTcw;
    mlRelativeFramePoses.push_back(Tcr);
    mlpReferences.push_back(mpReferenceKF);
    mlFrameTimes.push_back(currFrame.mTimeStamp);
    mlbLost.push_back(mState == LOST);
    // std::cout << "end writing pose" << std::endl;
}

void Tracking::initSystem()
{
    if (currFrame.detectFeaturesInFrame() > 500)
    {
        currFrame.mTcw = Sophus::SE3d();
        KeyFrame *pKFini = new KeyFrame(currFrame, mpMap, mpKeyFrameDB);

        // Insert KeyFrame in the map
        mpMap->AddKeyFrame(pKFini);

        // Create MapPoints and asscoiate to KeyFrame
        for (int i = 0; i < currFrame.N; i++)
        {
            float z = currFrame.mvDepth[i];
            if (z > 0)
            {
                Eigen::Vector3d x3D = pKFini->UnprojectStereo(i);
                MapPoint *pNewMP = new MapPoint(x3D, pKFini, mpMap);
                pNewMP->AddObservation(pKFini, i);
                pKFini->AddMapPoint(pNewMP, i);
                pNewMP->ComputeDistinctiveDescriptors();
                pNewMP->UpdateNormalAndDepth();
                mpMap->AddMapPoint(pNewMP);

                currFrame.mvpMapPoints[i] = pNewMP;
            }
        }

        if (mpMap->MapPointsInMap() < 300)
        {
            mState = SYSTEM_NOT_READY;
            delete mpMap;
            return;
        }

        mpLocalMapper->InsertKeyFrame(pKFini);

        lastFrame = Frame(currFrame);
        mpLastKeyFrame = pKFini;

        mvpLocalKeyFrames.push_back(pKFini);
        mvpLocalMapPoints = mpMap->GetAllMapPoints();
        mpReferenceKF = pKFini;
        currFrame.mpReferenceKF = pKFini;

        mpMap->SetReferenceMapPoints(mvpLocalMapPoints);

        mpMap->mvpKeyFrameOrigins.push_back(pKFini);

        mState = OK;

        // Create dense map struct
        mpCurrVoxelMap = new MapStruct(g_calib[0]);
        mpCurrVoxelMap->SetMeshEngine(mpMeshEngine);
        mpCurrVoxelMap->SetRayTraceEngine(rayTracer);
        mpCurrVoxelMap->create(15000, 12000, 12500, 0.007, 0.03);
        mpCurrVoxelMap->Reset();

        pKFini->mpVoxelStruct = mpCurrVoxelMap;

        // Set up Dense Tracker
        mpTracker->SetReferenceImage(currFrame.mImGray);
        mpTracker->SetReferenceDepth(currFrame.mImDepth);

        // Fuse the first depth image
        mRawDepth.upload(currFrame.mImDepth);
        mpCurrVoxelMap->Fuse(mRawDepth, currFrame.mTcw);

        if (mpViewer)
            mpViewer->setKeyFrameImage(currFrame.mImGray, currFrame.mvKeys);
    }
}

bool Tracking::takeNewFrame()
{
    // Sophus::SE3d Tmw = mpCurrVoxelMap->mTcw;
    // Sophus::SE3d Tcm = Tmw.inverse() * lastFrame.mTcw;

    mpCurrVoxelMap->RayTrace(lastFrame.mTcp);
    auto vmap = mpCurrVoxelMap->GetRayTracingResult();
    mpTracker->SetReferenceModel(vmap);

    // Set tracking frames
    mpTracker->SetTrackingImage(currFrame.mImGray);
    mpTracker->SetTrackingDepth(currFrame.mImDepth);

    // Calculate the relateive transformation
    Sophus::SE3d DT = mpTracker->GetTransform(Sophus::SE3d(), false);

    // if (DT.translation().norm() > 0.3)
    // {
    //     std::cout << DT.translation().norm() << std::endl;
    //     std::cout << DT.matrix3x4() << std::endl;
    //     std::cout << "Tracking lost, frame id: " << currFrame.mnId << std::endl;
    //     // mpTracker->WriteDebugImages();
    //     return false;
    // }

    // std::cout << DT.matrix3x4() << std::endl;

    mpTracker->SwapFrameBuffer();
    currFrame.mTcw = lastFrame.mTcw * DT.inverse();
    currFrame.mTcp = lastFrame.mTcp * DT.inverse();

    mRawDepth.upload(currFrame.mImDepth);
    mpCurrVoxelMap->Fuse(mRawDepth, currFrame.mTcp);
    g_nTrackedFrame++;

    if (mpViewer)
    {
        Sophus::SE3d Tcw = mpReferenceKF->GetPose() * currFrame.mTcp;
        mpViewer->setLivePose(Tcw.matrix());
    }

    return true;
}

bool Tracking::Relocalization()
{
    // Extract Features
    currFrame.detectFeaturesInFrame();
    if (currFrame.N < 500)
    {
        std::cout << "Relocalisation failed, Not enough features..." << std::endl;
        return false;
    }

    // Compute Bag of Words Vector
    currFrame.ComputeBoW();

    // Relocalization is performed when tracking is lost
    // Track Lost: Query KeyFrame Database for keyframe candidates for relocalisation
    std::vector<KeyFrame *> vpCandidateKFs = mpKeyFrameDB->DetectRelocalizationCandidates(&currFrame);

    if (vpCandidateKFs.empty())
        return false;

    currFrame.CreateRelocalisationPoints();

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
            int nmatches = matcher.SearchByBoW(currFrame, pKF, vvpMapPointMatches[iKF]);
            std::cout << "matches for kf: " << iKF << " : " << nmatches << std::endl;

            if (nmatches < 20)
            {
                vbDiscarded[iKF] = true;
                continue;
            }
            else
            {
                PoseSolver *pSolver = new PoseSolver(&currFrame, pKF, vvpMapPointMatches[iKF]);
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
                        currFrame.mvpMapPoints[j] = vvpMapPointMatches[i][j];
                        sFound.insert(vvpMapPointMatches[i][j]);
                    }
                    else
                        currFrame.mvpMapPoints[j] = NULL;
                }

                int nGood = Optimizer::PoseOptimization(currFrame);
                std::cout << "after inital pose optimization: " << nGood << std::endl;
                if (nGood < 10)
                    continue;

                for (int io = 0; io < currFrame.N; io++)
                    if (currFrame.mvbOutlier[io])
                        currFrame.mvpMapPoints[io] = nullptr;

                // If few inliers, search by projection in a coarse window and optimize again
                if (nGood < 50)
                {
                    int nadditional = matcher2.SearchByProjection(currFrame, vpCandidateKFs[i], sFound, 10, 100);

                    if (nadditional + nGood >= 50)
                    {
                        nGood = Optimizer::PoseOptimization(currFrame);
                        std::cout << "nGood after 1st optimization...: " << nGood << std::endl;

                        // If many inliers but still not enough, search by projection again in a narrower window
                        // the camera has been already optimized with many points
                        if (nGood > 30 && nGood < 50)
                        {
                            sFound.clear();
                            for (int ip = 0; ip < currFrame.N; ip++)
                                if (currFrame.mvpMapPoints[ip])
                                    sFound.insert(currFrame.mvpMapPoints[ip]);
                            nadditional = matcher2.SearchByProjection(currFrame, vpCandidateKFs[i], sFound, 3, 64);
                            std::cout << "nGood before further optimization...: " << nGood + nadditional << std::endl;

                            // Final optimization
                            if (nGood + nadditional >= 50)
                            {
                                nGood = Optimizer::PoseOptimization(currFrame);

                                for (int io = 0; io < currFrame.N; io++)
                                    if (currFrame.mvbOutlier[io])
                                        currFrame.mvpMapPoints[io] = NULL;

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
        mpCurrVoxelMap = pMatchedKF->mpVoxelStruct;
        mpCurrVoxelMap->DeleteMesh();
        mnLastSuccessRelocFrameId = currFrame.mnId;
        currFrame.mTcp = pMatchedKF->GetPoseInverse() * currFrame.mTcw;
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
    Optimizer::PoseOptimization(currFrame);
    int mnMatchesInliers = 0;

    // Update MapPoints Statistics
    for (int i = 0; i < currFrame.N; i++)
    {
        if (currFrame.mvpMapPoints[i])
        {
            if (!currFrame.mvbOutlier[i])
            {
                currFrame.mvpMapPoints[i]->IncreaseFound();
                currFrame.mvpMapPoints[i]->mnLastFrameSeen = currFrame.mnId;
                // if (!mbOnlyTracking)
                // {
                //     if (currFrame.mvpMapPoints[i]->Observations() > 0)
                //         mnMatchesInliers++;
                // }
                // else
                mnMatchesInliers++;
            }
            else
            {
                currFrame.mvbOutlier[i] = false;
                currFrame.mvpMapPoints[i] = nullptr;
            }
        }
    }

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
        if (currFrame.isInFrustum(pMP, 0.5))
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
        if (currFrame.mnId - mnLastSuccessRelocFrameId <= 1)
            th = 3;

        matcher.SearchByProjection(currFrame, mvpLocalMapPoints, th);
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
            if (pMP->mnTrackReferenceForFrame == currFrame.mnId)
                continue;
            if (!pMP->isBad())
            {
                mvpLocalMapPoints.push_back(pMP);
                pMP->mnTrackReferenceForFrame = currFrame.mnId;
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
        pKF->mnTrackReferenceForFrame = currFrame.mnId;
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
                if (pNeighKF->mnTrackReferenceForFrame != currFrame.mnId)
                {
                    mvpLocalKeyFrames.push_back(pNeighKF);
                    pNeighKF->mnTrackReferenceForFrame = currFrame.mnId;
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
                if (pChildKF->mnTrackReferenceForFrame != currFrame.mnId)
                {
                    mvpLocalKeyFrames.push_back(pChildKF);
                    pChildKF->mnTrackReferenceForFrame = currFrame.mnId;
                    break;
                }
            }
        }

        KeyFrame *pParent = pKF->GetParent();
        if (pParent)
        {
            if (pParent->mnTrackReferenceForFrame != currFrame.mnId)
            {
                mvpLocalKeyFrames.push_back(pParent);
                pParent->mnTrackReferenceForFrame = currFrame.mnId;
                break;
            }
        }
    }

    if (pKFmax)
    {
        mpReferenceKF = pKFmax;
        currFrame.mpReferenceKF = mpReferenceKF;
    }
}

bool Tracking::NeedNewKeyFrame()
{
    bool bCreateNew = false;
    Sophus::SE3d DT = currFrame.mTcp;

    if (DT.log().topRows<3>().norm() > 0.08)
        bCreateNew = true;
    else if (DT.log().bottomRows<3>().norm() > 0.1)
        bCreateNew = true;

    return bCreateNew;
}

void Tracking::CreateNewMapPoints()
{
    // We sort points by the measured depth by the stereo/RGBD sensor.
    // We create all those MapPoints whose depth < mThDepth.
    // If there are less than 100 close points we create the 100 closest.
    std::vector<std::pair<float, int>> vDepthIdx;
    vDepthIdx.reserve(currFrame.N);
    for (int i = 0; i < currFrame.N; i++)
    {
        float z = currFrame.mvDepth[i];
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

            MapPoint *pMP = currFrame.mvpMapPoints[i];

            if (!pMP)
                bCreateNew = true;
            else if (pMP->Observations() < 1)
            {
                bCreateNew = true;
                currFrame.mvpMapPoints[i] = nullptr;
            }

            if (bCreateNew)
            {
                Eigen::Vector3d x3D = mpReferenceKF->UnprojectStereo(i);
                MapPoint *pNewMP = new MapPoint(x3D, mpReferenceKF, mpMap);
                pNewMP->AddObservation(mpReferenceKF, i);
                mpReferenceKF->AddMapPoint(pNewMP, i);
                pNewMP->ComputeDistinctiveDescriptors();
                pNewMP->UpdateNormalAndDepth();
                mpMap->AddMapPoint(pNewMP);

                currFrame.mvpMapPoints[i] = pNewMP;
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
}

void Tracking::CreateNewKeyFrame()
{
    if (!mpLocalMapper->SetNotStop(true))
        return;

    mpCurrVoxelMap->RayTrace(currFrame.mTcp);
    currFrame.mImDepth = cv::Mat(mpCurrVoxelMap->GetRayTracingResultDepth());
    cv::imshow("currFrame.mImDepth", currFrame.mImDepth);
    currFrame.detectFeaturesInFrame();
    currFrame.mTcw = mpReferenceKF->GetPose() * currFrame.mTcp;

    if (!TrackLocalMap())
        return;

    mpTracker->setKeyFrame(currFrame.mImGray, currFrame.mImDepth);
    mpReferenceKF = new KeyFrame(currFrame, mpMap, mpKeyFrameDB);

    CreateNewMapPoints();

    currFrame.mpReferenceKF = mpReferenceKF;

    currFrame.mTcp = Sophus::SE3d();
    mpLocalMapper->InsertKeyFrame(mpReferenceKF);
    mpLocalMapper->SetNotStop(false);

    if (mpViewer)
        mpViewer->setKeyFrameImage(currFrame.mImGray, currFrame.mvKeys);

    createNewVoxelMap();
}

void Tracking::reset()
{
    mState = SYSTEM_NOT_READY;
    mpReferenceKF = 0;
    mpLastKeyFrame = 0;
    mpCurrVoxelMap = 0;
}

bool Tracking::needNewVoxelMap()
{
    return true;
}

void Tracking::createNewVoxelMap()
{
    mpMap->AddMapStruct(mpCurrVoxelMap);
    mpCurrVoxelMap->SetActiveFlag(false);
    mpCurrVoxelMap->Hibernate();

    // Create a new MapStruct
    mpCurrVoxelMap = new MapStruct(g_calib[0]);
    mpCurrVoxelMap->SetActiveFlag(true);
    mpCurrVoxelMap->SetMeshEngine(mpMeshEngine);
    mpCurrVoxelMap->SetRayTraceEngine(rayTracer);
    mpCurrVoxelMap->create(15000, 12000, 12500, 0.007, 0.03);
    mpCurrVoxelMap->Reset();
    mpReferenceKF->mpVoxelStruct = mpCurrVoxelMap;
    mpCurrVoxelMap->mTcw = mpReferenceKF->GetPose();
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

} // namespace slam