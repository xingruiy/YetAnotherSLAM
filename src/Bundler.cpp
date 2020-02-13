#include <Thirdparty/g2o/g2o/core/block_solver.h>
#include <Thirdparty/g2o/g2o/core/optimizable_graph.h>
#include <Thirdparty/g2o/g2o/core/optimization_algorithm_levenberg.h>
#include <Thirdparty/g2o/g2o/solvers/linear_solver_eigen.h>
#include <Thirdparty/g2o/g2o/types/types_six_dof_expmap.h>
#include <Thirdparty/g2o/g2o/core/robust_kernel_impl.h>
#include <Thirdparty/g2o/g2o/solvers/linear_solver_dense.h>
#include <Thirdparty/g2o/g2o/types/types_seven_dof_expmap.h>
#include <mutex>
#include <unistd.h>
#include <Eigen/StdVector>

#include "Bundler.h"
#include "Converter.h"

namespace SLAM
{

int Bundler::PoseOptimization(KeyFrame *pKF)
{
    g2o::SparseOptimizer optimizer;
    g2o::BlockSolver_6_3::LinearSolverType *linearSolver;

    linearSolver = new g2o::LinearSolverDense<g2o::BlockSolver_6_3::PoseMatrixType>();

    g2o::BlockSolver_6_3 *solver_ptr = new g2o::BlockSolver_6_3(linearSolver);

    g2o::OptimizationAlgorithmLevenberg *solver = new g2o::OptimizationAlgorithmLevenberg(solver_ptr);
    optimizer.setAlgorithm(solver);
    int nInitialCorrespondences = 0;

    // Set keyframe vertex
    g2o::VertexSE3Expmap *vSE3 = new g2o::VertexSE3Expmap();
    Eigen::Matrix<double, 3, 3> R = pKF->mTcw.inverse().rotationMatrix();
    Eigen::Matrix<double, 3, 1> t = pKF->mTcw.inverse().translation();
    g2o::SE3Quat estimate(R, t);
    vSE3->setEstimate(estimate);
    vSE3->setId(0);
    vSE3->setFixed(false);
    optimizer.addVertex(vSE3);

    // Set MapPoint vertices
    const int N = pKF->N;
    std::vector<g2o::EdgeStereoSE3ProjectXYZOnlyPose *> vpEdgesStereo;
    std::vector<size_t> vnIndexEdgeStereo;
    vpEdgesStereo.reserve(N);
    vnIndexEdgeStereo.reserve(N);

    const float delta = sqrt(7.815);
    {
        std::unique_lock<std::mutex> lock(MapPoint::mGlobalMutex);

        for (int i = 0; i < N; i++)
        {
            MapPoint *pMP = pKF->mvpMapPoints[i];
            if (pMP)
            {
                nInitialCorrespondences++;
                pKF->mvbOutlier[i] = false;

                //SET EDGE
                Eigen::Matrix<double, 3, 1> obs;
                const cv::KeyPoint &kpUn = pKF->mvKeysUn[i];
                const float &kp_ur = pKF->mvuRight[i];
                obs << kpUn.pt.x, kpUn.pt.y, kp_ur;

                g2o::EdgeStereoSE3ProjectXYZOnlyPose *e = new g2o::EdgeStereoSE3ProjectXYZOnlyPose();

                e->setVertex(0, dynamic_cast<g2o::OptimizableGraph::Vertex *>(optimizer.vertex(0)));
                e->setMeasurement(obs);
                const float invSigma2 = pKF->mvInvLevelSigma2[kpUn.octave];
                Eigen::Matrix3d Info = Eigen::Matrix3d::Identity() * invSigma2;
                e->setInformation(Info);

                g2o::RobustKernelHuber *rk = new g2o::RobustKernelHuber;
                e->setRobustKernel(rk);
                rk->setDelta(delta);

                e->fx = pKF->fx;
                e->fy = pKF->fy;
                e->cx = pKF->cx;
                e->cy = pKF->cy;
                e->bf = pKF->mbf;
                e->Xw = pMP->GetWorldPos();

                optimizer.addEdge(e);
                vpEdgesStereo.push_back(e);
                vnIndexEdgeStereo.push_back(i);
            }
        }
    }

    if (nInitialCorrespondences < 3)
        return 0;

    // We perform 4 optimizations, after each optimization we classify observation as inlier/outlier
    // At the next optimization, outliers are not included, but at the end they can be classified as inliers again.
    const float chi2Th[4] = {7.815, 7.815, 7.815, 7.815};
    const int its[4] = {10, 10, 10, 10};

    int nBad = 0;
    for (size_t it = 0; it < 4; it++)
    {
        Eigen::Matrix<double, 3, 3> R = pKF->mTcw.inverse().rotationMatrix();
        Eigen::Matrix<double, 3, 1> t = pKF->mTcw.inverse().translation();
        g2o::SE3Quat estimate(R, t);
        vSE3->setEstimate(estimate);
        optimizer.initializeOptimization(0);
        optimizer.optimize(its[it]);

        nBad = 0;
        for (size_t i = 0, iend = vpEdgesStereo.size(); i < iend; i++)
        {
            g2o::EdgeStereoSE3ProjectXYZOnlyPose *e = vpEdgesStereo[i];

            const size_t idx = vnIndexEdgeStereo[i];

            if (pKF->mvbOutlier[idx])
            {
                e->computeError();
            }

            const float chi2 = e->chi2();

            if (chi2 > chi2Th[it])
            {
                pKF->mvbOutlier[idx] = true;
                e->setLevel(1);
                nBad++;
            }
            else
            {
                e->setLevel(0);
                pKF->mvbOutlier[idx] = false;
            }

            if (it == 2)
                e->setRobustKernel(0);
        }

        if (optimizer.edges().size() < 10)
            break;
    }

    // Recover optimized pose and return number of inliers
    g2o::VertexSE3Expmap *vSE3_recov = static_cast<g2o::VertexSE3Expmap *>(optimizer.vertex(0));
    g2o::SE3Quat SE3quat_recov = vSE3_recov->estimate();
    cv::Mat pose = ORB_SLAM2::Converter::toCvMat(SE3quat_recov);
    pKF->SetPose(pose);

    return nInitialCorrespondences - nBad;
}

void Bundler::LocalBundleAdjustment(KeyFrame *pKF, bool *pbStopFlag, Map *pMap)
{
    // Local KeyFrames: First Breath Search from Current Keyframe
    std::list<KeyFrame *> lLocalKeyFrames;

    lLocalKeyFrames.push_back(pKF);
    pKF->mnBALocalForKF = pKF->mnId;

    const auto vNeighKFs = pKF->GetVectorCovisibleKeyFrames();
    for (int i = 0, iend = vNeighKFs.size(); i < iend; i++)
    {
        KeyFrame *pKFi = vNeighKFs[i];
        pKFi->mnBALocalForKF = pKF->mnId;
        if (!pKFi->isBad())
            lLocalKeyFrames.push_back(pKFi);
    }

    // Local MapPoints seen in Local KeyFrames
    std::list<MapPoint *> lLocalMapPoints;
    for (auto lit = lLocalKeyFrames.begin(), lend = lLocalKeyFrames.end(); lit != lend; lit++)
    {
        std::vector<MapPoint *> vpMPs = (*lit)->GetMapPointMatches();
        for (auto vit = vpMPs.begin(), vend = vpMPs.end(); vit != vend; vit++)
        {
            MapPoint *pMP = *vit;
            if (pMP)
                if (!pMP->isBad())
                    if (pMP->mnBALocalForKF != pKF->mnId)
                    {
                        lLocalMapPoints.push_back(pMP);
                        pMP->mnBALocalForKF = pKF->mnId;
                    }
        }
    }

    // Fixed Keyframes.
    // Keyframes that see Local MapPoints but that are not Local Keyframes
    std::list<KeyFrame *> lFixedCameras;
    for (auto lit = lLocalMapPoints.begin(), lend = lLocalMapPoints.end(); lit != lend; lit++)
    {
        auto observations = (*lit)->GetObservations();
        for (auto mit = observations.begin(), mend = observations.end(); mit != mend; mit++)
        {
            KeyFrame *pKFi = mit->first;

            if (pKFi->mnBALocalForKF != pKF->mnId && pKFi->mnBAFixedForKF != pKF->mnId)
            {
                pKFi->mnBAFixedForKF = pKF->mnId;
                if (!pKFi->isBad())
                    lFixedCameras.push_back(pKFi);
            }
        }
    }

    // Setup optimizer
    g2o::SparseOptimizer optimizer;
    g2o::BlockSolver_6_3::LinearSolverType *linearSolver;

    linearSolver = new g2o::LinearSolverEigen<g2o::BlockSolver_6_3::PoseMatrixType>();

    g2o::BlockSolver_6_3 *solver_ptr = new g2o::BlockSolver_6_3(linearSolver);

    g2o::OptimizationAlgorithmLevenberg *solver = new g2o::OptimizationAlgorithmLevenberg(solver_ptr);
    optimizer.setAlgorithm(solver);

    if (pbStopFlag)
        optimizer.setForceStopFlag(pbStopFlag);

    // unsigned long maxKFid = 0;

    // // Set Local KeyFrame vertices
    // for (list<KeyFrame *>::iterator lit = lLocalKeyFrames.begin(), lend = lLocalKeyFrames.end(); lit != lend; lit++)
    // {
    //     KeyFrame *pKFi = *lit;
    //     g2o::VertexSE3Expmap *vSE3 = new g2o::VertexSE3Expmap();
    //     vSE3->setEstimate(ORB_SLAM2::Converter::toSE3Quat(pKFi->GetPose()));
    //     vSE3->setId(pKFi->mnId);
    //     vSE3->setFixed(pKFi->mnId == 0);
    //     optimizer.addVertex(vSE3);
    //     if (pKFi->mnId > maxKFid)
    //         maxKFid = pKFi->mnId;
    // }

    // // Set Fixed KeyFrame vertices
    // for (list<KeyFrame *>::iterator lit = lFixedCameras.begin(), lend = lFixedCameras.end(); lit != lend; lit++)
    // {
    //     KeyFrame *pKFi = *lit;
    //     g2o::VertexSE3Expmap *vSE3 = new g2o::VertexSE3Expmap();
    //     vSE3->setEstimate(ORB_SLAM2::Converter::toSE3Quat(pKFi->GetPose()));
    //     vSE3->setId(pKFi->mnId);
    //     vSE3->setFixed(true);
    //     optimizer.addVertex(vSE3);
    //     if (pKFi->mnId > maxKFid)
    //         maxKFid = pKFi->mnId;
    // }

    // // Set MapPoint vertices
    // const int nExpectedSize = (lLocalKeyFrames.size() + lFixedCameras.size()) * lLocalMapPoints.size();

    // vector<g2o::EdgeSE3ProjectXYZ *> vpEdgesMono;
    // vpEdgesMono.reserve(nExpectedSize);

    // vector<KeyFrame *> vpEdgeKFMono;
    // vpEdgeKFMono.reserve(nExpectedSize);

    // vector<MapPoint *> vpMapPointEdgeMono;
    // vpMapPointEdgeMono.reserve(nExpectedSize);

    // vector<g2o::EdgeStereoSE3ProjectXYZ *> vpEdgesStereo;
    // vpEdgesStereo.reserve(nExpectedSize);

    // vector<KeyFrame *> vpEdgeKFStereo;
    // vpEdgeKFStereo.reserve(nExpectedSize);

    // vector<MapPoint *> vpMapPointEdgeStereo;
    // vpMapPointEdgeStereo.reserve(nExpectedSize);

    // const float thHuberMono = sqrt(5.991);
    // const float thHuberStereo = sqrt(7.815);

    // for (list<MapPoint *>::iterator lit = lLocalMapPoints.begin(), lend = lLocalMapPoints.end(); lit != lend; lit++)
    // {
    //     MapPoint *pMP = *lit;
    //     g2o::VertexSBAPointXYZ *vPoint = new g2o::VertexSBAPointXYZ();
    //     vPoint->setEstimate(ORB_SLAM2::Converter::toVector3d(pMP->GetWorldPos()));
    //     int id = pMP->mnId + maxKFid + 1;
    //     vPoint->setId(id);
    //     vPoint->setMarginalized(true);
    //     optimizer.addVertex(vPoint);

    //     const map<KeyFrame *, size_t> observations = pMP->GetObservations();

    //     //Set edges
    //     for (map<KeyFrame *, size_t>::const_iterator mit = observations.begin(), mend = observations.end(); mit != mend; mit++)
    //     {
    //         KeyFrame *pKFi = mit->first;

    //         if (!pKFi->isBad())
    //         {
    //             const cv::KeyPoint &kpUn = pKFi->mvKeysUn[mit->second];

    //             // Monocular observation
    //             if (pKFi->mvuRight[mit->second] < 0)
    //             {
    //                 Eigen::Matrix<double, 2, 1> obs;
    //                 obs << kpUn.pt.x, kpUn.pt.y;

    //                 g2o::EdgeSE3ProjectXYZ *e = new g2o::EdgeSE3ProjectXYZ();

    //                 e->setVertex(0, dynamic_cast<g2o::OptimizableGraph::Vertex *>(optimizer.vertex(id)));
    //                 e->setVertex(1, dynamic_cast<g2o::OptimizableGraph::Vertex *>(optimizer.vertex(pKFi->mnId)));
    //                 e->setMeasurement(obs);
    //                 const float &invSigma2 = pKFi->mvInvLevelSigma2[kpUn.octave];
    //                 e->setInformation(Eigen::Matrix2d::Identity() * invSigma2);

    //                 g2o::RobustKernelHuber *rk = new g2o::RobustKernelHuber;
    //                 e->setRobustKernel(rk);
    //                 rk->setDelta(thHuberMono);

    //                 e->fx = pKFi->fx;
    //                 e->fy = pKFi->fy;
    //                 e->cx = pKFi->cx;
    //                 e->cy = pKFi->cy;

    //                 optimizer.addEdge(e);
    //                 vpEdgesMono.push_back(e);
    //                 vpEdgeKFMono.push_back(pKFi);
    //                 vpMapPointEdgeMono.push_back(pMP);
    //             }
    //             else // Stereo observation
    //             {
    //                 Eigen::Matrix<double, 3, 1> obs;
    //                 const float kp_ur = pKFi->mvuRight[mit->second];
    //                 obs << kpUn.pt.x, kpUn.pt.y, kp_ur;

    //                 g2o::EdgeStereoSE3ProjectXYZ *e = new g2o::EdgeStereoSE3ProjectXYZ();

    //                 e->setVertex(0, dynamic_cast<g2o::OptimizableGraph::Vertex *>(optimizer.vertex(id)));
    //                 e->setVertex(1, dynamic_cast<g2o::OptimizableGraph::Vertex *>(optimizer.vertex(pKFi->mnId)));
    //                 e->setMeasurement(obs);
    //                 const float &invSigma2 = pKFi->mvInvLevelSigma2[kpUn.octave];
    //                 Eigen::Matrix3d Info = Eigen::Matrix3d::Identity() * invSigma2;
    //                 e->setInformation(Info);

    //                 g2o::RobustKernelHuber *rk = new g2o::RobustKernelHuber;
    //                 e->setRobustKernel(rk);
    //                 rk->setDelta(thHuberStereo);

    //                 e->fx = pKFi->fx;
    //                 e->fy = pKFi->fy;
    //                 e->cx = pKFi->cx;
    //                 e->cy = pKFi->cy;
    //                 e->bf = pKFi->mbf;

    //                 optimizer.addEdge(e);
    //                 vpEdgesStereo.push_back(e);
    //                 vpEdgeKFStereo.push_back(pKFi);
    //                 vpMapPointEdgeStereo.push_back(pMP);
    //             }
    //         }
    //     }
    // }

    // if (pbStopFlag)
    //     if (*pbStopFlag)
    //         return;

    // optimizer.initializeOptimization();
    // optimizer.optimize(5);

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

    // vector<pair<KeyFrame *, MapPoint *>> vToErase;
    // vToErase.reserve(vpEdgesMono.size() + vpEdgesStereo.size());

    // // Check inlier observations
    // for (size_t i = 0, iend = vpEdgesMono.size(); i < iend; i++)
    // {
    //     g2o::EdgeSE3ProjectXYZ *e = vpEdgesMono[i];
    //     MapPoint *pMP = vpMapPointEdgeMono[i];

    //     if (pMP->isBad())
    //         continue;

    //     if (e->chi2() > 5.991 || !e->isDepthPositive())
    //     {
    //         KeyFrame *pKFi = vpEdgeKFMono[i];
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
    // std::unique_lock<std::mutex> lock(pMap->mMutexMapUpdate);

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
    //     pKF->SetPose(ORB_SLAM2::Converter::toCvMat(SE3quat));
    // }

    // //Points
    // for (list<MapPoint *>::iterator lit = lLocalMapPoints.begin(), lend = lLocalMapPoints.end(); lit != lend; lit++)
    // {
    //     MapPoint *pMP = *lit;
    //     g2o::VertexSBAPointXYZ *vPoint = static_cast<g2o::VertexSBAPointXYZ *>(optimizer.vertex(pMP->mnId + maxKFid + 1));
    //     pMP->SetWorldPos(ORB_SLAM2::Converter::toCvMat(vPoint->estimate()));
    //     pMP->UpdateNormalAndDepth();
    // }
}

} // namespace SLAM