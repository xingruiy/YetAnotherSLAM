#include "Optimizer/Optimizer.h"
#include <g2o/core/sparse_optimizer.h>
#include <g2o/core/block_solver.h>
#include <g2o/core/linear_solver.h>
#include <g2o/solvers/dense/linear_solver_dense.h>
#include <g2o/core/optimization_algorithm_levenberg.h>
#include <list>
#include <memory>

std::vector<std::shared_ptr<KeyFrame>> GetCovisibleKeyFrames(std::shared_ptr<KeyFrame> KF)
{
}

void LocalBundleAdjustment(std::shared_ptr<KeyFrame> pKF, Map *map)
{
    std::list<std::shared_ptr<KeyFrame>> lLocalKeyFrames;

    lLocalKeyFrames.push_back(pKF);
    pKF->localReferenceId = pKF->KFId;

    const std::vector<std::shared_ptr<KeyFrame>> vNeighKFs;
    // Big TODO
    // vNeighKFs = pKF->GetVectorCovisibleKeyFrames();

    // Collect co-visibly key frames
    for (int i = 0, iend = vNeighKFs.size(); i < iend; i++)
    {
        const auto pKFi = vNeighKFs[i];
        pKFi->localReferenceId = pKF->KFId;
        // if (!pKFi->isBad())
        lLocalKeyFrames.push_back(pKFi);
    }

    // Local MapPoints seen in Local KeyFrames
    std::list<std::shared_ptr<MapPoint>> lLocalMapPoints;
    for (const auto &KFi : lLocalKeyFrames)
    {
        for (auto &pMP : KFi->mapPoints)
        {
            if (pMP && !pMP->setToRemove && pMP->localReferenceId != KFi->KFId)
            {
                lLocalMapPoints.push_back(pMP);
                pMP->localReferenceId = pKF->KFId;
            }
        }
    }

    // Fixed Keyframes. Keyframes that see Local MapPoints but that are not Local Keyframes
    std::list<std::shared_ptr<KeyFrame>> lFixedCameras;
    for (auto &mp : lLocalMapPoints)
    {
        for (const auto obs : mp->observations)
        {
            auto pKFi = obs.first;
            if (pKFi->localReferenceId != pKF->KFId && pKFi->localReferenceId2 != pKF->KFId)
            {
                pKFi->localReferenceId2 = pKF->KFId;
                lFixedCameras.push_back(pKFi);
            }
        }
    }

    // Setup optimizer
    g2o::SparseOptimizer optimizer;
    std::unique_ptr<g2o::BlockSolver_6_3::LinearSolverType> linearSolver;
    linearSolver = g2o::make_unique<g2o::LinearSolverDense<g2o::BlockSolver_6_3::PoseMatrixType>>();
    g2o::OptimizationAlgorithmLevenberg *solver = new g2o::OptimizationAlgorithmLevenberg(
        g2o::make_unique<g2o::BlockSolver_6_3>(std::move(linearSolver)));
    optimizer.setAlgorithm(solver);

    // if (pbStopFlag)
    //     optimizer.setForceStopFlag(pbStopFlag);

    unsigned long maxKFid = 0;

    // Set Local KeyFrame vertices
    for (auto &pKFi : lLocalKeyFrames)
    {
        g2o::VertexSE3Expmap *vSE3 = new g2o::VertexSE3Expmap();
        vSE3->setEstimate(Converter::toSE3Quat(pKFi->RT);
        vSE3->setId(pKFi->KFId);
        vSE3->setFixed(pKFi->KFId == 0);
        optimizer.addVertex(vSE3);
        if (pKFi->KFId > maxKFid)
            maxKFid = pKFi->KFId;
    }

    // Set Fixed KeyFrame vertices
    for (auto &pKFi : lFixedCameras)
    {
        g2o::VertexSE3Expmap *vSE3 = new g2o::VertexSE3Expmap();
        vSE3->setEstimate(Converter::toSE3Quat(pKFi->RT));
        vSE3->setId(pKFi->KFId);
        vSE3->setFixed(true);
        optimizer.addVertex(vSE3);
        if (pKFi->KFId > maxKFid)
            maxKFid = pKFi->KFId;
    }

    // Set MapPoint vertices
    const int nExpectedSize = (lLocalKeyFrames.size() + lFixedCameras.size()) * lLocalMapPoints.size();

    std::vector<g2o::EdgeSE3ProjectXYZ *> vpEdgesMono;
    vpEdgesMono.reserve(nExpectedSize);

    std::vector<std::shared_ptr<KeyFrame>> vpEdgeKFMono;
    vpEdgeKFMono.reserve(nExpectedSize);

    std::vector<std::shared_ptr<MapPoint>> vpMapPointEdgeMono;
    vpMapPointEdgeMono.reserve(nExpectedSize);

    std::vector<g2o::EdgeStereoSE3ProjectXYZ *> vpEdgesStereo;
    vpEdgesStereo.reserve(nExpectedSize);

    std::vector<std::shared_ptr<KeyFrame>> vpEdgeKFStereo;
    vpEdgeKFStereo.reserve(nExpectedSize);

    std::vector<std::shared_ptr<MapPoint>> vpMapPointEdgeStereo;
    vpMapPointEdgeStereo.reserve(nExpectedSize);

    const float thHuberStereo = sqrt(7.815);

    for (auto &pMP : lLocalMapPoints)
    {
        g2o::VertexSBAPointXYZ *vPoint = new g2o::VertexSBAPointXYZ();
        vPoint->setEstimate(Converter::toVector3d(pMP->pos));
        int id = pMP->mpId + maxKFid + 1;
        vPoint->setId(id);
        vPoint->setMarginalized(true);
        optimizer.addVertex(vPoint);

        //Set edges
        for (auto &mit : pMP->observations)
        {
            // if (!pKFi->isBad())
            // {
            // auto &pKFi = mit.first;
            // const cv::KeyPoint &kpUn = pKFi->keyPoints[mit.second];

            // Eigen::Matrix<double, 2, 1> obs;
            // obs << kpUn.pt.x, kpUn.pt.y;

            // g2o::EdgeStereoSE3ProjectXYZ *e = new g2o::EdgeStereoSE3ProjectXYZ();

            // e->setVertex(0, dynamic_cast<g2o::OptimizableGraph::Vertex *>(optimizer.vertex(id)));
            // e->setVertex(1, dynamic_cast<g2o::OptimizableGraph::Vertex *>(optimizer.vertex(pKFi->KFId)));
            // e->setMeasurement(obs);
            // // const float &invSigma2 = pKFi->mvInvLevelSigma2[kpUn.octave];
            // // Eigen::Matrix3d Info = Eigen::Matrix3d::Identity() * invSigma2;
            // // e->setInformation(Info);

            // g2o::RobustKernelHuber *rk = new g2o::RobustKernelHuber;
            // e->setRobustKernel(rk);
            // rk->setDelta(thHuberStereo);

            // e->fx = pKFi->fx;
            // e->fy = pKFi->fy;
            // e->cx = pKFi->cx;
            // e->cy = pKFi->cy;
            // e->bf = pKFi->mbf;

            // optimizer.addEdge(e);
            // vpEdgesStereo.push_back(e);
            // vpEdgeKFStereo.push_back(pKFi);
            // vpMapPointEdgeStereo.push_back(pMP);
        }
        // }
    }

    // if (pbStopFlag)
    //     if (*pbStopFlag)
    //         return;

    optimizer.initializeOptimization();
    optimizer.optimize(5);

    // bool bDoMore = true;

    // if (pbStopFlag)
    //     if (*pbStopFlag)
    //         bDoMore = false;

    // if (bDoMore)
    // {

    //     // Check inlier observations
    //     for (size_t i = 0, iend = vpEdgesMono.size(); i < iend; i++)
    //     {
    //         g2o::EdgeSE3ProjectXYZ *e = vpEdgesMono[i];
    //         MapPoint *pMP = vpMapPointEdgeMono[i];

    //         if (pMP->isBad())
    //             continue;

    //         if (e->chi2() > 5.991 || !e->isDepthPositive())
    //         {
    //             e->setLevel(1);
    //         }

    //         e->setRobustKernel(0);
    //     }

    //     for (size_t i = 0, iend = vpEdgesStereo.size(); i < iend; i++)
    //     {
    //         g2o::EdgeStereoSE3ProjectXYZ *e = vpEdgesStereo[i];
    //         MapPoint *pMP = vpMapPointEdgeStereo[i];

    //         if (pMP->isBad())
    //             continue;

    //         if (e->chi2() > 7.815 || !e->isDepthPositive())
    //         {
    //             e->setLevel(1);
    //         }

    //         e->setRobustKernel(0);
    //     }

    //     // Optimize again without the outliers

    //     optimizer.initializeOptimization(0);
    //     optimizer.optimize(10);
    // }

    // std::vector<std::pair<std::shared_ptr<KeyFrame>, std::shared_ptr<MapPoint>>> vToErase;
    // vToErase.reserve(vpEdgesMono.size() + vpEdgesStereo.size());

    // // Check inlier observations
    // for (size_t i = 0, iend = vpEdgesMono.size(); i < iend; i++)
    // {
    //     g2o::EdgeSE3ProjectXYZ *e = vpEdgesMono[i];
    //     auto &pMP = vpMapPointEdgeMono[i];

    //     if (pMP->setToRemove)
    //         continue;

    //     if (e->chi2() > 5.991 || !e->isDepthPositive())
    //     {
    //         auto &pKFi = vpEdgeKFMono[i];
    //         vToErase.push_back(make_pair(pKFi, pMP));
    //     }
    // }

    // for (size_t i = 0, iend = vpEdgesStereo.size(); i < iend; i++)
    // {
    //     g2o::EdgeStereoSE3ProjectXYZ *e = vpEdgesStereo[i];
    //     MapPoint *pMP = vpMapPointEdgeStereo[i];

    //     if (pMP->isBad())
    //         continue;

    //     if (e->chi2() > 7.815 || !e->isDepthPositive())
    //     {
    //         KeyFrame *pKFi = vpEdgeKFStereo[i];
    //         vToErase.push_back(make_pair(pKFi, pMP));
    //     }
    // }

    // // Get Map Mutex
    // unique_lock<mutex> lock(pMap->mMutexMapUpdate);

    // if (!vToErase.empty())
    // {
    //     for (size_t i = 0; i < vToErase.size(); i++)
    //     {
    //         KeyFrame *pKFi = vToErase[i].first;
    //         MapPoint *pMPi = vToErase[i].second;
    //         pKFi->EraseMapPointMatch(pMPi);
    //         pMPi->EraseObservation(pKFi);
    //     }
    // }

    // // Recover optimized data

    // //Keyframes
    // for (list<KeyFrame *>::iterator lit = lLocalKeyFrames.begin(), lend = lLocalKeyFrames.end(); lit != lend; lit++)
    // {
    //     KeyFrame *pKF = *lit;
    //     g2o::VertexSE3Expmap *vSE3 = static_cast<g2o::VertexSE3Expmap *>(optimizer.vertex(pKF->mnId));
    //     g2o::SE3Quat SE3quat = vSE3->estimate();
    //     pKF->SetPose(Converter::toCvMat(SE3quat));
    // }

    // //Points
    // for (list<MapPoint *>::iterator lit = lLocalMapPoints.begin(), lend = lLocalMapPoints.end(); lit != lend; lit++)
    // {
    //     MapPoint *pMP = *lit;
    //     g2o::VertexSBAPointXYZ *vPoint = static_cast<g2o::VertexSBAPointXYZ *>(optimizer.vertex(pMP->mnId + maxKFid + 1));
    //     pMP->SetWorldPos(Converter::toCvMat(vPoint->estimate()));
    //     pMP->UpdateNormalAndDepth();
    // }
}