#include "globalMapper/ceresSolver.h"

CeresSolver::CeresSolver(Mat33d &K)
    : intrinsics(K), camLocalSE3(new LocalParameterizationSE3())
{
    this->K[0] = K(0, 0);
    this->K[1] = K(1, 1);
    this->K[2] = K(0, 2);
    this->K[3] = K(1, 2);
}

void CeresSolver::addCamera(const size_t camIdx, const SE3 &T, bool fixed)
{
    if (cameras.find(camIdx) != cameras.end())
    {
        printf("camera block already exist, id: %lu\n", camIdx);
        return;
    }

    cameras.insert(std::pair<size_t, SE3>(camIdx, T));
}

void CeresSolver::removeCamera(const size_t camIdx)
{
    if (cameras.find(camIdx) == cameras.end())
    {
        printf("cannot find camera block, id: %lu\n", camIdx);
        return;
    }

    cameras.erase(camIdx);
}

void CeresSolver::addWorldPoint(const size_t ptIdx, const Vec3d &pos, bool fixed)
{
    if (mapPoints.find(ptIdx) != mapPoints.end())
    {
        printf("point block already exist, id: %lu\n", ptIdx);
        return;
    }

    mapPoints.insert(std::pair<size_t, Vec3d>(ptIdx, pos));
}

void CeresSolver::removeWorldPoint(const size_t ptIdx)
{
    if (mapPoints.find(ptIdx) == mapPoints.end())
    {
        printf("cannot find point block, id: %lu\n", ptIdx);
        return;
    }

    if (observations.find(ptIdx) != observations.end())
    {
        observations.erase(ptIdx);
    }

    mapPoints.erase(ptIdx);
}

void CeresSolver::addObservation(const size_t ptIdx, const size_t camIdx, const Vec2d &obs)
{
    if (cameras.find(camIdx) == cameras.end())
    {
        printf("cannot find camera block, id: %lu\n", camIdx);
        return;
    }

    if (mapPoints.find(ptIdx) == mapPoints.end())
    {
        printf("cannot find point block, id: %lu\n", ptIdx);
        return;
    }

    observations[ptIdx].insert(std::pair<size_t, Vec2d>(camIdx, obs));
}

void CeresSolver::removeObservation(const size_t ptIdx, const size_t camIdx)
{
    if (observations.find(ptIdx) == observations.end())
    {
        printf("cannot find observations, id: %lu\n", ptIdx);
        return;
    }

    observations[ptIdx].erase(camIdx);
}

void CeresSolver::optimize(const int maxiter, const size_t oldestKFId, const size_t newestKFId)
{
    ceres::Problem problem;
    size_t numResidualBlock = 0;

    auto *se3Local = new LocalParameterizationSE3();
    for (auto kf : cameras)
    {
        problem.AddParameterBlock(kf.second.data(), SE3::num_parameters, se3Local);

        if (kf.first != newestKFId || kf.first == oldestKFId)
        {
            // printf("kf: %lu, fixedKF: %lu\n", kf.first, fixedKfId);
            problem.AddParameterBlock(&K[0], 4);
            problem.SetParameterBlockConstant(&K[0]);
            problem.SetParameterBlockConstant(kf.second.data());
        }
    }

    for (auto iter = observations.begin(), iend = observations.end(); iter != iend; ++iter)
    {
        auto ptId = iter->first;
        if (observations[ptId].size() < 2)
            continue;

        problem.AddParameterBlock(mapPoints[ptId].data(), 3);
        problem.SetParameterBlockConstant(mapPoints[ptId].data());

        for (auto iter2 = iter->second.begin(), iend2 = iter->second.end(); iter2 != iend2; ++iter2)
        {
            auto kfId = iter2->first;
            auto obs = iter2->second;

            problem.AddResidualBlock(
                ReprojectionErrorFunctor::create(obs(0), obs(1)),
                NULL,
                &K[0],
                cameras[kfId].data(),
                mapPoints[ptId].data());

            numResidualBlock++;
        }
    }

    printf("total num of %lu residuals to be optimized.\n", numResidualBlock);

    ceres::Solver::Options options;
    ceres::Solver::Summary summary;
    // options.eta
    options.linear_solver_type = ceres::SPARSE_SCHUR;
    // options.max_num_iterations = maxiter;
    Solve(options, &problem, &summary);
    std::cout << summary.BriefReport() << std::endl;
}

bool CeresSolver::hasCamera(const size_t camIdx)
{
    return (cameras.find(camIdx) != cameras.end());
}

SE3 CeresSolver::getCamera(const size_t camIdx)
{
    return cameras[camIdx].inverse();
}
