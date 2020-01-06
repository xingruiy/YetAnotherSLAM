#include <g2o/g2o/core/block_solver.h>
#include <g2o/g2o/core/optimizable_graph.h>
#include <g2o/g2o/core/optimization_algorithm_levenberg.h>
#include <g2o/g2o/solvers/linear_solver_eigen.h>
#include <g2o/g2o/types/types_six_dof_expmap.h>
#include <g2o/g2o/core/robust_kernel_impl.h>
#include <g2o/g2o/solvers/linear_solver_dense.h>
#include <g2o/g2o/types/types_seven_dof_expmap.h>

#include "Optimizer.h"
#include "Converter.h"

#include <Eigen/Core>

int Optimizer::PoseOptimization(KeyFrame *pKF)
{
    g2o::SparseOptimizer optimizer;
    g2o::BlockSolver_6_3::LinearSolverType *linearSolver;
    linearSolver = new g2o::LinearSolverDense<g2o::BlockSolver_6_3::PoseMatrixType>();
    g2o::BlockSolver_6_3 *solver_ptr = new g2o::BlockSolver_6_3(linearSolver);
    g2o::OptimizationAlgorithmLevenberg *solver = new g2o::OptimizationAlgorithmLevenberg(solver_ptr);
    optimizer.setAlgorithm(solver);

    int nInitialCorrespondences = 0;

    // Set KeyFrame vertex
    g2o::VertexSE3Expmap *vSE3 = new g2o::VertexSE3Expmap();
    vSE3->setEstimate(ORB_SLAM2::Converter::toSE3Quat(pKF->mTcw.matrix()));
    vSE3->setId(0);
    vSE3->setFixed(false);
    optimizer.addVertex(vSE3);

    // Set MapPoint vertices
    const int N = pKF->N;

    vector<g2o::EdgeStereoSE3ProjectXYZOnlyPose *> vpEdgesStereo;
    vector<size_t> vnIndexEdgeStereo;
    vpEdgesStereo.reserve(N);
    vnIndexEdgeStereo.reserve(N);

    const float deltaStereo = sqrt(7.815);

    {
        unique_lock<mutex> lock(MapPoint::mGlobalMutex);

        for (int i = 0; i < N; i++)
        {
            MapPoint *pMP = pKF->mvpMapPoints[i];
            if (pMP)
            {
                nInitialCorrespondences++;
                pKF->mvbOutlier[i] = false;

                //SET EDGE
                Eigen::Matrix<double, 3, 1> obs;
                const cv::KeyPoint &kpUn = pKF->mvKeys[i];
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
                rk->setDelta(deltaStereo);

                e->fx = Frame::fx;
                e->fy = Frame::fy;
                e->cx = Frame::cx;
                e->cy = Frame::cy;
                e->bf = pKF->mbf;
                Eigen::Vector3d Xw = pMP->mWorldPos;
                e->Xw[0] = Xw(0);
                e->Xw[1] = Xw(1);
                e->Xw[2] = Xw(2);

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
    const float chi2Stereo[4] = {7.815, 7.815, 7.815, 7.815};
    const int its[4] = {10, 10, 10, 10};

    int nBad = 0;
    for (size_t it = 0; it < 4; it++)
    {

        vSE3->setEstimate(ORB_SLAM2::Converter::toSE3Quat(pKF->mTcw.matrix()));
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

            if (chi2 > chi2Stereo[it])
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
    pKF->mTcw = Sophus::SE3d(SE3quat_recov.to_homogeneous_matrix());

    return nInitialCorrespondences - nBad;
}