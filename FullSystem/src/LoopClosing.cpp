#include "LoopClosing.h"
#include "ORBMatcher.h"
#include "Sim3Solver.h"
#include "Optimizer.h"
#include "Converter.h"

namespace SLAM
{

LoopClosing::LoopClosing(Map *pMap, KeyFrameDatabase *pDB, ORBVocabulary *pVoc)
    : mpMap(pMap), mpKeyFrameDB(pDB), mpORBVocabulary(pVoc), mLastLoopKFid(0),
      mbFixScale(true), mpThreadGBA(nullptr), mbRunningGBA(false)
{
    mnCovisibilityConsistencyTh = 3;
}

void LoopClosing::SetLocalMapper(LocalMapping *pLocalMapper)
{
    mpLocalMapper = pLocalMapper;
}

void LoopClosing::Run()
{
    while (!g_bSystemKilled)
    {
        if (CheckNewKeyFrames())
        {
            if (DetectLoop())
            {
                if (ComputeSim3())
                    // Perform loop fusion and pose graph optimization
                    CorrectLoop();
            }
        }
    }
}

void LoopClosing::InsertKeyFrame(KeyFrame *pKF)
{
    std::unique_lock<std::mutex> lock(mMutexLoopQueue);
    if (pKF->mnId != 0)
        mlpLoopKeyFrameQueue.push_back(pKF);
}

bool LoopClosing::CheckNewKeyFrames()
{
    std::unique_lock<std::mutex> lock(mMutexLoopQueue);
    return !mlpLoopKeyFrameQueue.empty();
}

bool LoopClosing::DetectLoop()
{
    {
        std::unique_lock<std::mutex> lock(mMutexLoopQueue);
        mpCurrentKF = mlpLoopKeyFrameQueue.front();
        mlpLoopKeyFrameQueue.pop_front();
        // Avoid that a keyframe can be erased while it is being process by this thread
        mpCurrentKF->SetNotErase();
    }

    //If the map contains less than 10 KF or less than 10 KF have passed from last loop detection
    if (mpCurrentKF->mnId < mLastLoopKFid + 10)
    {
        mpKeyFrameDB->add(mpCurrentKF);
        mpCurrentKF->SetErase();
        return false;
    }

    // Compute reference BoW similarity score
    // This is the lowest score to a connected keyframe in the covisibility graph
    // We will impose loop candidates to have a higher similarity than this
    const auto vpConnectedKeyFrames = mpCurrentKF->GetVectorCovisibleKeyFrames();
    const DBoW2::BowVector &CurrentBowVec = mpCurrentKF->mBowVec;
    float minScore = 1;
    for (size_t i = 0; i < vpConnectedKeyFrames.size(); i++)
    {
        KeyFrame *pKF = vpConnectedKeyFrames[i];
        if (!pKF || pKF->isBad())
            continue;
        const DBoW2::BowVector &BowVec = pKF->mBowVec;

        float score = mpORBVocabulary->score(CurrentBowVec, BowVec);

        if (score < minScore)
            minScore = score;
    }

    // Query the database imposing the minimum score
    auto vpCandidateKFs = mpKeyFrameDB->DetectLoopCandidates(mpCurrentKF, minScore);

    // If there are no loop candidates, just add new keyframe and return false
    if (vpCandidateKFs.empty())
    {
        mpKeyFrameDB->add(mpCurrentKF);
        mvConsistentGroups.clear();
        mpCurrentKF->SetErase();
        return false;
    }

    // For each loop candidate check consistency with previous loop candidates
    // Each candidate expands a covisibility group (keyframes connected to the loop candidate in the covisibility graph)
    // A group is consistent with a previous group if they share at least a keyframe
    // We must detect a consistent loop in several consecutive keyframes to accept it
    mvpEnoughConsistentCandidates.clear();

    std::vector<ConsistentGroup> vCurrentConsistentGroups;
    std::vector<bool> vbConsistentGroup(mvConsistentGroups.size(), false);
    for (size_t i = 0, iend = vpCandidateKFs.size(); i < iend; i++)
    {
        KeyFrame *pCandidateKF = vpCandidateKFs[i];

        std::set<KeyFrame *> spCandidateGroup = pCandidateKF->GetConnectedKeyFrames();
        spCandidateGroup.insert(pCandidateKF);

        bool bEnoughConsistent = false;
        bool bConsistentForSomeGroup = false;
        for (size_t iG = 0, iendG = mvConsistentGroups.size(); iG < iendG; iG++)
        {
            std::set<KeyFrame *> sPreviousGroup = mvConsistentGroups[iG].first;

            bool bConsistent = false;
            for (auto sit = spCandidateGroup.begin(), send = spCandidateGroup.end(); sit != send; sit++)
            {
                if (sPreviousGroup.count(*sit))
                {
                    bConsistent = true;
                    bConsistentForSomeGroup = true;
                    break;
                }
            }

            if (bConsistent)
            {
                int nPreviousConsistency = mvConsistentGroups[iG].second;
                int nCurrentConsistency = nPreviousConsistency + 1;
                if (!vbConsistentGroup[iG])
                {
                    ConsistentGroup cg = make_pair(spCandidateGroup, nCurrentConsistency);
                    vCurrentConsistentGroups.push_back(cg);
                    vbConsistentGroup[iG] = true; //this avoid to include the same group more than once
                }
                if (nCurrentConsistency >= mnCovisibilityConsistencyTh && !bEnoughConsistent)
                {
                    mvpEnoughConsistentCandidates.push_back(pCandidateKF);
                    bEnoughConsistent = true; //this avoid to insert the same candidate more than once
                }
            }
        }

        // If the group is not consistent with any previous group insert with consistency counter set to zero
        if (!bConsistentForSomeGroup)
        {
            ConsistentGroup cg = make_pair(spCandidateGroup, 0);
            vCurrentConsistentGroups.push_back(cg);
        }
    }

    // Update Covisibility Consistent Groups
    mvConsistentGroups = vCurrentConsistentGroups;

    // Add Current Keyframe to database
    mpKeyFrameDB->add(mpCurrentKF);

    if (mvpEnoughConsistentCandidates.empty())
    {
        mpCurrentKF->SetErase();
        return false;
    }
    else
    {
        return true;
    }

    mpCurrentKF->SetErase();
    return false;
}

bool LoopClosing::ComputeSim3()
{
    // For each consistent loop candidate we try to compute a Sim3
    const int nInitialCandidates = mvpEnoughConsistentCandidates.size();

    // We compute first ORB matches for each candidate
    // If enough matches are found, we setup a Sim3Solver
    ORBMatcher matcher(0.75, true);

    std::vector<Sim3Solver *> vpSim3Solvers;
    vpSim3Solvers.resize(nInitialCandidates);

    std::vector<std::vector<MapPoint *>> vvpMapPointMatches;
    vvpMapPointMatches.resize(nInitialCandidates);

    std::vector<bool> vbDiscarded;
    vbDiscarded.resize(nInitialCandidates);

    int nCandidates = 0; //candidates with enough matches

    for (int i = 0; i < nInitialCandidates; i++)
    {
        KeyFrame *pKF = mvpEnoughConsistentCandidates[i];

        // avoid that local mapping erase it while it is being processed in this thread
        pKF->SetNotErase();

        if (pKF->isBad())
        {
            vbDiscarded[i] = true;
            continue;
        }

        int nmatches = matcher.SearchByBoW(mpCurrentKF, pKF, vvpMapPointMatches[i]);

        if (nmatches < 20)
        {
            vbDiscarded[i] = true;
            continue;
        }
        else
        {
            Sim3Solver *pSolver = new Sim3Solver(mpCurrentKF, pKF, vvpMapPointMatches[i], true);
            pSolver->SetRansacParameters(0.99, 20, 300);
            vpSim3Solvers[i] = pSolver;
        }

        nCandidates++;
    }

    bool bMatch = false;

    // Perform alternatively RANSAC iterations for each candidate
    // until one is succesful or all fail
    while (nCandidates > 0 && !bMatch)
    {
        for (int i = 0; i < nInitialCandidates; i++)
        {
            if (vbDiscarded[i])
                continue;

            KeyFrame *pKF = mvpEnoughConsistentCandidates[i];

            // Perform 5 Ransac Iterations
            std::vector<bool> vbInliers;
            int nInliers;
            bool bNoMore;

            Sim3Solver *pSolver = vpSim3Solvers[i];

            Sophus::SE3d T12;
            bool found = pSolver->iterate(5, bNoMore, vbInliers, nInliers, T12);

            // If Ransac reachs max. iterations discard keyframe
            if (bNoMore)
            {
                vbDiscarded[i] = true;
                nCandidates--;
            }

            // If RANSAC returns a Sim3, perform a guided matching and optimize with all correspondences
            if (found)
            {
                std::vector<MapPoint *> vpMapPointMatches(vvpMapPointMatches[i].size(), static_cast<MapPoint *>(NULL));
                for (size_t j = 0, jend = vbInliers.size(); j < jend; j++)
                {
                    if (vbInliers[j])
                        vpMapPointMatches[j] = vvpMapPointMatches[i][j];
                }

                matcher.SearchBySim3(mpCurrentKF, pKF, vpMapPointMatches, T12, 7.5);

                // gScm here should be the inverse of T12, i.e. 2->1
                Sophus::SE3d T21 = T12.inverse();
                g2o::Sim3 gScm(T21.rotationMatrix(), T21.translation(), 1.0);

                const int nInliers = Optimizer::OptimizeSim3(mpCurrentKF, pKF, vpMapPointMatches, gScm, 10, mbFixScale);

                // If optimization is succesful stop ransacs and continue
                if (nInliers >= 20)
                {
                    bMatch = true;
                    mpMatchedKF = pKF;
                    Sophus::SE3d T21(gScm.rotation(), gScm.translation());
                    Sophus::SE3d Twc = pKF->GetPoseInverse();

                    mTcwNew = pKF->GetPose() * T21.inverse();

                    mvpCurrentMatchedPoints = vpMapPointMatches;
                    break;
                }
            }
        }
    }

    if (!bMatch)
    {
        for (int i = 0; i < nInitialCandidates; i++)
            mvpEnoughConsistentCandidates[i]->SetErase();
        mpCurrentKF->SetErase();
        return false;
    }

    // Retrieve MapPoints seen in Loop Keyframe and neighbors
    std::vector<KeyFrame *> vpLoopConnectedKFs = mpMatchedKF->GetVectorCovisibleKeyFrames();
    vpLoopConnectedKFs.push_back(mpMatchedKF);
    mvpLoopMapPoints.clear();
    for (auto vit = vpLoopConnectedKFs.begin(); vit != vpLoopConnectedKFs.end(); vit++)
    {
        KeyFrame *pKF = *vit;
        std::vector<MapPoint *> vpMapPoints = pKF->GetMapPointMatches();
        for (size_t i = 0, iend = vpMapPoints.size(); i < iend; i++)
        {
            MapPoint *pMP = vpMapPoints[i];
            if (pMP)
            {
                if (!pMP->isBad() && pMP->mnLoopPointForKF != mpCurrentKF->mnId)
                {
                    mvpLoopMapPoints.push_back(pMP);
                    pMP->mnLoopPointForKF = mpCurrentKF->mnId;
                }
            }
        }
    }

    // Find more matches projecting with the computed Sim3
    matcher.SearchByProjection(mpCurrentKF, mTcwNew, mvpLoopMapPoints, mvpCurrentMatchedPoints, 10);

    // If enough matches accept Loop
    int nTotalMatches = 0;
    for (size_t i = 0; i < mvpCurrentMatchedPoints.size(); i++)
    {
        if (mvpCurrentMatchedPoints[i])
            nTotalMatches++;
    }

    if (nTotalMatches >= 40)
    {
        for (int i = 0; i < nInitialCandidates; i++)
            if (mvpEnoughConsistentCandidates[i] != mpMatchedKF)
                mvpEnoughConsistentCandidates[i]->SetErase();
        return true;
    }
    else
    {
        for (int i = 0; i < nInitialCandidates; i++)
            mvpEnoughConsistentCandidates[i]->SetErase();
        mpCurrentKF->SetErase();
        return false;
    }
}

void LoopClosing::CorrectLoop()
{
    std::cout << "Loop detected!" << std::endl;

    // Send a stop signal to Local Mapping
    // Avoid new keyframes are inserted while correcting the loop
    mpLocalMapper->RequestStop();

    // If a Global Bundle Adjustment is running, abort it
    if (isRunningGBA())
    {
        std::unique_lock<std::mutex> lock(mMutexGBA);
        mbStopGBA = true;

        mnFullBAIdx++;

        if (mpThreadGBA)
        {
            mpThreadGBA->detach();
            delete mpThreadGBA;
        }
    }

    // Wait until Local Mapping has effectively stopped
    while (!mpLocalMapper->isStopped())
    {
        usleep(1000);
    }

    // Ensure current keyframe is updated
    mpCurrentKF->UpdateConnections();

    // Retrive keyframes connected to the current keyframe and compute corrected Sim3 pose by propagation
    mvpCurrentConnectedKFs = mpCurrentKF->GetVectorCovisibleKeyFrames();
    mvpCurrentConnectedKFs.push_back(mpCurrentKF);

    KeyFrameAndPose CorrectedSim3, NonCorrectedSim3;
    CorrectedSim3[mpCurrentKF] = mTcwNew;
    Sophus::SE3d Twc1 = mpCurrentKF->GetPoseInverse();

    {
        // Get Map Mutex
        std::unique_lock<std::mutex> lock(mpMap->mMutexMapUpdate);

        for (auto vit = mvpCurrentConnectedKFs.begin(), vend = mvpCurrentConnectedKFs.end(); vit != vend; vit++)
        {
            KeyFrame *pKFi = *vit;

            Sophus::SE3d Tcw2 = pKFi->GetPose();

            if (pKFi != mpCurrentKF)
            {
                //Pose corrected with the Sim3 of the loop closure
                Sophus::SE3d T21 = Twc1 * Tcw2;
                Sophus::SE3d CorrectedScw = mTcwNew * T21;
                CorrectedSim3[pKFi] = CorrectedScw;
            }

            //Pose without correction
            NonCorrectedSim3[pKFi] = Tcw2;
        }

        // Correct all MapPoints obsrved by current keyframe and neighbors, so that they align with the other side of the loop
        for (auto mit = CorrectedSim3.begin(), mend = CorrectedSim3.end(); mit != mend; mit++)
        {
            KeyFrame *pKFi = mit->first;

            Sophus::SE3d CorrectedTcw = mit->second;
            Sophus::SE3d CorrectedTwc = CorrectedTcw.inverse();

            Sophus::SE3d NonCorrectedTcw = NonCorrectedSim3[pKFi];

            std::vector<MapPoint *> vpMPsi = pKFi->GetMapPointMatches();
            for (size_t iMP = 0, endMPi = vpMPsi.size(); iMP < endMPi; iMP++)
            {
                MapPoint *pMPi = vpMPsi[iMP];
                if (!pMPi)
                    continue;
                if (pMPi->isBad())
                    continue;
                if (pMPi->mnCorrectedByKF == mpCurrentKF->mnId)
                    continue;

                // Project with non-corrected pose and project back with corrected pose
                Eigen::Vector3d eigP3Dw = pMPi->GetWorldPos();
                Eigen::Vector3d eigCorrectedP3Dw = CorrectedTwc * NonCorrectedTcw * eigP3Dw;

                pMPi->SetWorldPos(eigCorrectedP3Dw);
                pMPi->mnCorrectedByKF = mpCurrentKF->mnId;
                pMPi->mnCorrectedReference = pKFi->mnId;
                pMPi->UpdateNormalAndDepth();
            }

            pKFi->SetPose(CorrectedTcw);

            // Make sure connections are updated
            pKFi->UpdateConnections();
        }

        // Start Loop Fusion
        // Update matched map points and replace if duplicated
        for (size_t i = 0; i < mvpCurrentMatchedPoints.size(); i++)
        {
            if (mvpCurrentMatchedPoints[i])
            {
                MapPoint *pLoopMP = mvpCurrentMatchedPoints[i];
                MapPoint *pCurMP = mpCurrentKF->GetMapPoint(i);
                if (pCurMP)
                    pCurMP->Replace(pLoopMP);
                else
                {
                    mpCurrentKF->AddMapPoint(pLoopMP, i);
                    pLoopMP->AddObservation(mpCurrentKF, i);
                    pLoopMP->ComputeDistinctiveDescriptors();
                }
            }
        }
    }

    // Project MapPoints observed in the neighborhood of the loop keyframe
    // into the current keyframe and neighbors using corrected poses.
    // Fuse duplications.
    SearchAndFuse(CorrectedSim3);

    // After the MapPoint fusion, new links in the covisibility graph will appear attaching both sides of the loop
    std::map<KeyFrame *, std::set<KeyFrame *>> LoopConnections;

    for (auto vit = mvpCurrentConnectedKFs.begin(), vend = mvpCurrentConnectedKFs.end(); vit != vend; vit++)
    {
        KeyFrame *pKFi = *vit;
        std::vector<KeyFrame *> vpPreviousNeighbors = pKFi->GetVectorCovisibleKeyFrames();

        // Update connections. Detect new links.
        pKFi->UpdateConnections();
        LoopConnections[pKFi] = pKFi->GetConnectedKeyFrames();
        for (auto vit_prev = vpPreviousNeighbors.begin(), vend_prev = vpPreviousNeighbors.end(); vit_prev != vend_prev; vit_prev++)
        {
            LoopConnections[pKFi].erase(*vit_prev);
        }
        for (auto vit2 = mvpCurrentConnectedKFs.begin(), vend2 = mvpCurrentConnectedKFs.end(); vit2 != vend2; vit2++)
        {
            LoopConnections[pKFi].erase(*vit2);
        }
    }

    // Optimize graph
    Optimizer::OptimizeEssentialGraph(mpMap, mpMatchedKF, mpCurrentKF, NonCorrectedSim3, CorrectedSim3, LoopConnections, mbFixScale);

    mpMap->InformNewBigChange();

    // Add loop edge
    mpMatchedKF->AddLoopEdge(mpCurrentKF);
    mpCurrentKF->AddLoopEdge(mpMatchedKF);

    // Launch a new thread to perform Global Bundle Adjustment
    mbRunningGBA = true;
    mbFinishedGBA = false;
    mbStopGBA = false;
    mpThreadGBA = new std::thread(&LoopClosing::RunGlobalBundleAdjustment, this, mpCurrentKF->mnId);

    // Loop closed. Release Local Mapping.
    mpLocalMapper->Release();

    mLastLoopKFid = mpCurrentKF->mnId;
}

void LoopClosing::SearchAndFuse(const KeyFrameAndPose &CorrectedPosesMap)
{
    ORBMatcher matcher(0.8);

    for (auto mit = CorrectedPosesMap.begin(), mend = CorrectedPosesMap.end(); mit != mend; mit++)
    {
        KeyFrame *pKF = mit->first;
        Sophus::SE3d CorrectedTcw = mit->second;

        std::vector<MapPoint *> vpReplacePoints(mvpLoopMapPoints.size(), static_cast<MapPoint *>(NULL));
        matcher.Fuse(pKF, CorrectedTcw, mvpLoopMapPoints, 4, vpReplacePoints);

        // Get Map Mutex
        std::unique_lock<std::mutex> lock(mpMap->mMutexMapUpdate);
        const int nLP = mvpLoopMapPoints.size();
        for (int i = 0; i < nLP; i++)
        {
            MapPoint *pRep = vpReplacePoints[i];
            if (pRep)
            {
                pRep->Replace(mvpLoopMapPoints[i]);
            }
        }
    }
}

void LoopClosing::RunGlobalBundleAdjustment(unsigned long nLoopKF)
{
    std::cout << "Starting Global Bundle Adjustment" << std::endl;

    int idx = mnFullBAIdx;
    Optimizer::GlobalBundleAdjustemnt(mpMap, 10, &mbStopGBA, nLoopKF, false);

    // Update all MapPoints and KeyFrames
    // Local Mapping was active during BA, that means that there might be new keyframes
    // not included in the Global BA and they are not consistent with the updated map.
    // We need to propagate the correction through the spanning tree
    {
        std::unique_lock<std::mutex> lock(mMutexGBA);
        if (idx != mnFullBAIdx)
            return;

        if (!mbStopGBA)
        {
            std::cout << "Global Bundle Adjustment finished" << std::endl;
            std::cout << "Updating map ..." << std::endl;
            mpLocalMapper->RequestStop();
            // Wait until Local Mapping has effectively stopped

            while (!mpLocalMapper->isStopped() && !mpLocalMapper->isFinished())
            {
                usleep(1000);
            }

            // Get Map Mutex
            std::unique_lock<std::mutex> lock(mpMap->mMutexMapUpdate);

            // Correct keyframes starting at map first keyframe
            std::list<KeyFrame *> lpKFtoCheck(mpMap->mvpKeyFrameOrigins.begin(), mpMap->mvpKeyFrameOrigins.end());

            while (!lpKFtoCheck.empty())
            {
                KeyFrame *pKF = lpKFtoCheck.front();
                const std::set<KeyFrame *> sChilds = pKF->GetChilds();
                Sophus::SE3d Twc = pKF->GetPoseInverse();
                for (auto sit = sChilds.begin(); sit != sChilds.end(); sit++)
                {
                    KeyFrame *pChild = *sit;
                    if (pChild->mnBAGlobalForKF != nLoopKF)
                    {
                        Sophus::SE3d Tchildc = Twc * pChild->GetPose();
                        pChild->mTcwGBA = pKF->mTcwGBA * Tchildc; //*Tcorc*pKF->mTcwGBA;
                        pChild->mnBAGlobalForKF = nLoopKF;
                    }

                    lpKFtoCheck.push_back(pChild);
                }

                pKF->mTcwBefGBA = pKF->GetPoseInverse();
                pKF->SetPose(pKF->mTcwGBA);
                lpKFtoCheck.pop_front();
            }

            // Correct MapPoints
            const auto vpMPs = mpMap->GetAllMapPoints();

            for (size_t i = 0; i < vpMPs.size(); i++)
            {
                MapPoint *pMP = vpMPs[i];

                if (pMP->isBad())
                    continue;

                if (pMP->mnBAGlobalForKF == nLoopKF)
                {
                    // If optimized by Global BA, just update
                    pMP->SetWorldPos(pMP->mPosGBA);
                }
                else
                {
                    // Update according to the correction of its reference keyframe
                    KeyFrame *pRefKF = pMP->GetReferenceKeyFrame();

                    if (pRefKF->mnBAGlobalForKF != nLoopKF)
                        continue;

                    // Map to non-corrected camera
                    Eigen::Vector3d Xc = pRefKF->mTcwBefGBA * pMP->GetWorldPos();

                    // Backproject using corrected camera
                    pMP->SetWorldPos(pRefKF->GetPose() * Xc);
                }
            }

            mpMap->InformNewBigChange();

            mpLocalMapper->Release();

            std::cout << "Map updated!" << std::endl;
        }

        mbFinishedGBA = true;
        mbRunningGBA = false;
    }
}

} // namespace SLAM