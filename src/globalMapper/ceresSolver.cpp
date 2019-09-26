#include "globalMapper/ceresSolver.h"

CeresSolver::~CeresSolver()
{
    for (auto resMap : resBlockMap)
    {
        for (auto res : resMap.second)
        {
            // delete res.second.costFunction;
            // delete res.second.lossFunction;
        }
    }
}

CeresSolver::CeresSolver(Mat33d &K)
{
    this->K[0] = K(0, 0);
    this->K[1] = K(1, 1);
    this->K[2] = K(0, 2);
    this->K[3] = K(1, 2);

    // problemOptions.cost_function_ownership = ceres::Ownership::DO_NOT_TAKE_OWNERSHIP;
    // problemOptions.loss_function_ownership = ceres::Ownership::DO_NOT_TAKE_OWNERSHIP;
    solver = std::make_shared<ceres::Problem>(problemOptions);

    solverOptions.linear_solver_type = ceres::SPARSE_SCHUR;
}

void CeresSolver::addCamera(const size_t camId, const SE3 &T, bool fixed)
{
    if (hasCamera(camId))
    {
        printf("camera block already exists, id: %lu\n", camId);
        return;
    }

    // cameras.insert(std::pair<size_t, SE3>(camId, T));
    auto camera = std::make_shared<CameraBlock>(camId, T);
    camBlockMap.insert(std::make_pair(camId, camera));
    // solver->AddParameterBlock(
    //     camera->optimizationBuffer.data(),
    //     SE3::num_parameters,
    //     new LocalParameterizationSE3());
}

SE3 CeresSolver::getCamPoseOptimized(const size_t camId) const
{
    if (hasCamera(camId))
        return getCameraBlock(camId)->lastSuccessOptimized;
    return SE3();
}

Mat33d CeresSolver::getCamParamOptimized() const
{
    Mat33d Kmat = Mat33d::Identity();
    Kmat(0, 0) = K[0];
    Kmat(1, 1) = K[1];
    Kmat(0, 2) = K[2];
    Kmat(1, 2) = K[3];
    return Kmat;
}

std::shared_ptr<PointBlock> CeresSolver::getPointBlock(const size_t ptId) const
{
    return ptBlockMap.find(ptId)->second;
}

std::shared_ptr<CameraBlock> CeresSolver::getCameraBlock(const size_t camId) const
{
    return camBlockMap.find(camId)->second;
}

void CeresSolver::addResidualBlock(
    // std::shared_ptr<ceres::CostFunction> costFunction,
    // std::shared_ptr<ceres::LossFunction> lossFunction,
    std::shared_ptr<CameraBlock> cameraBlock,
    std::shared_ptr<PointBlock> pointBlock,
    ResidualBlock &resBlock)
{
    auto resId = solver->AddResidualBlock(
        resBlock.costFunction,
        resBlock.lossFunction,
        &K[0],
        cameraBlock->optimizationBuffer.data(),
        pointBlock->optimizationBuffer.data());

    resBlock.residualBlockId = resId;
}

void CeresSolver::removeResidualBlock(ceres::ResidualBlockId id)
{
    solver->RemoveResidualBlock(id);
}

void CeresSolver::removeResidualBlock(ResidualBlock &block)
{
    delete block.costFunction;
    delete block.lossFunction;
    // std::cout << solver->NumResidualBlocks() << std::endl;
    // std::cout << block.residualBlockId << std::endl;
    solver->RemoveResidualBlock(block.residualBlockId);
}

void CeresSolver::removeCamera(const size_t camId)
{
    if (!hasCamera(camId))
    {
        printf("cannot find camera block, id: %lu\n", camId);
        return;
    }

    auto block = camBlockMap[camId];
    // solver->RemoveParameterBlock(block->optimizationBuffer.data());
    camBlockMap.erase(camId);
}

void CeresSolver::setCameraBlockConstant(const size_t camId)
{
    if (hasCamera(camId))
    {
        auto camBlockIter = camBlockMap.find(camId);
        solver->SetParameterBlockConstant(camBlockIter->second->optimizationBuffer.data());
        camBlockMap.erase(camBlockIter);
    }
}

void CeresSolver::addWorldPoint(const size_t ptId, const Vec3d &pos, bool fixed)
{
    if (hasPoint(ptId))
    {
        printf("point block already exists, id: %lu\n", ptId);
        return;
    }

    auto point = std::make_shared<PointBlock>(ptId, pos);
    ptBlockMap.insert(std::make_pair(ptId, point));
    // solver->AddParameterBlock(point->optimizationBuffer.data(), 3);
    // solver->SetParameterBlockConstant(point->optimizationBuffer.data());
}

void CeresSolver::removeWorldPoint(const size_t ptId)
{
    if (!hasPoint(ptId))
    {
        printf("cannot find point block, id: %lu\n", ptId);
        return;
    }

    // std::cout << ptId << std::endl;
    // if (ptId == 0)
    //     return;

    // auto obsMap = resBlockMap.find(ptId);
    // if (obsMap != resBlockMap.end())
    // {
    //     for (auto obs : obsMap->second)
    //     {
    //         // if (obs.first == 0)
    //         //     continue;
    //         // std::cout << "pt: " << ptId << "cam: " << obs.first << std::endl;
    //         removeResidualBlock(obs.second);
    //     }

    //     obsMap->second.clear();
    // }

    auto blockId = ptBlockMap[ptId];
    // solver->RemoveParameterBlock(blockId->optimizationBuffer.data());
    ptBlockMap.erase(ptId);
    auto obsMap = resBlockMap.find(ptId);
    if (obsMap != resBlockMap.end())
    {
        obsMap->second.clear();
    }
}

void CeresSolver::addObservation(const size_t ptId, const size_t camId, const Vec2d &obs)
{
    // if (!hasCamera(camId))
    // {
    //     printf("cannot find camera block, id: %lu\n", camId);
    //     return;
    // }

    // if (!hasPoint(ptId))
    // {
    //     printf("cannot find point block, id: %lu\n", ptId);
    //     return;
    // }

    auto camIter = camBlockMap.find(camId);
    auto ptIter = ptBlockMap.find(ptId);
    if (camIter != camBlockMap.end() && ptIter != ptBlockMap.end())
    {
        // ResidualBlock resBlock;
        // resBlock.lossFunction = NULL;
        // resBlock.costFunction = ReprojectionErrorFunctor::create(obs(0), obs(1));

        // resBlock.residualBlockId = solver->AddResidualBlock(
        //     resBlock.costFunction,
        //     resBlock.lossFunction,
        //     &K[0],
        //     camIter->second->optimizationBuffer.data(),
        //     ptIter->second->optimizationBuffer.data());

        // std::cout << resBlock.residualBlockId << std::endl;

        // resBlock.residualBlockId = resId;

        // addResidualBlock(camIter->second, ptIter->second, resBlock);
        resBlockMap[ptId].insert(std::make_pair(camId, obs));
        // if (ptId < 10)
        //     for (auto ob : resBlockMap[ptId])
        //         std::cout << "pt: " << ptId << "cam: " << ob.first << std::endl;
    }
}

bool CeresSolver::hasResidualBlock(const size_t ptId, const size_t camId) const
{
    auto obs = resBlockMap.find(ptId)->second;
    return obs.find(camId) != obs.end();
}

void CeresSolver::removeObservation(const size_t ptId, const size_t camId)
{
    if (!hasResidualBlock(ptId, camId))
    {
        printf("cannot find residual block, id: %lu\n", ptId);
        return;
    }

    auto obsMap = resBlockMap.find(ptId);
    if (obsMap != resBlockMap.end())
    {
        auto obs = obsMap->second.find(camId);
        if (obs != obsMap->second.end())
        {
            // removeResidualBlock(obs->second);
            obsMap->second.erase(obs);
        }
    }
}

void CeresSolver::initializeParameters()
{
    for (auto camBlock : camBlockMap)
        camBlock.second->optimizationBuffer = camBlock.second->lastSuccessOptimized;

    for (auto ptBlock : ptBlockMap)
        ptBlock.second->optimizationBuffer = ptBlock.second->lastSuccessOptimized;
}

void CeresSolver::optimize(const int maxIter)
{
    // solverOptions.max_num_iterations = maxIter;

    initializeParameters();

    ceres::Problem problem;
    size_t smallestKFId = std::numeric_limits<size_t>::max();

    if (camBlockMap.size() < 5)
        return;

    ceres::LossFunction *robustLoss = new ceres::HuberLoss(100);

    problem.AddParameterBlock(&K[0], 4);
    problem.SetParameterBlockConstant(&K[0]);

    for (auto camBlock : camBlockMap)
    {
        if (camBlock.first < smallestKFId)
            smallestKFId = camBlock.first;

        problem.AddParameterBlock(camBlock.second->optimizationBuffer.data(), SE3::num_parameters, new LocalParameterizationSE3());
        // problem.SetParameterBlockConstant(camBlock.second->optimizationBuffer.data());
    }

    std::cout << smallestKFId << std::endl;
    problem.SetParameterBlockConstant(camBlockMap[smallestKFId]->optimizationBuffer.data());

    // std::cout << camBlockMap.size() << std::endl;

    for (auto ptBlock : ptBlockMap)
    {
        auto resBlock = resBlockMap.find(ptBlock.first);
        // if (resBlock->second.size() >= 2)
        if (resBlock != resBlockMap.end() && resBlock->second.size() != 0)
        {
            problem.AddParameterBlock(ptBlock.second->optimizationBuffer.data(), 3);
            problem.SetParameterBlockConstant(ptBlock.second->optimizationBuffer.data());
        }
        // problem.SetParameterBlockConstant(ptBlock.second->optimizationBuffer.data());
    }

    for (auto resBlock : resBlockMap)
    {
        auto ptIter = ptBlockMap.find(resBlock.first);
        // if (resBlock.second.size() >= 2)
        for (auto obs : resBlock.second)
        {
            auto camIter = camBlockMap.find(obs.first);
            if (ptIter != ptBlockMap.end() && camIter != camBlockMap.end())
            {
                // if (obs.second.size() > 1)
                // {
                problem.AddResidualBlock(
                    ReprojectionErrorFunctor::create(obs.second(0), obs.second(1)),
                    robustLoss,
                    &K[0],
                    camIter->second->optimizationBuffer.data(),
                    ptIter->second->optimizationBuffer.data());
                // }
                // else if (obs.second.size() > 0)
                // {
                //     SE3 camPose = camIter->second->lastSuccessOptimized;
                //     Vec3d ptPos = ptIter->second->lastSuccessOptimized;
                //     Vec3d ptTransformed = camPose * ptPos;
                //     problem.AddResidualBlock(
                //         PointToPointErrorFunctor::create(ptTransformed(0), ptTransformed(1), ptTransformed(2)),
                //         robustLoss,
                //         camIter->second->optimizationBuffer.data(),
                //         ptIter->second->optimizationBuffer.data());
                // }
            }
        }
    }

    // Solve(solverOptions, solver.get(), &solverLog);
    Solve(solverOptions, &problem, &solverLog);
    std::cout << solverLog.BriefReport() << std::endl;

    if (solverLog.termination_type == ceres::CONVERGENCE)
    {
        for (auto camBlock : camBlockMap)
            camBlock.second->lastSuccessOptimized = camBlock.second->optimizationBuffer;

        for (auto ptBlock : ptBlockMap)
        {
            // auto &p1 = ptBlock.second->lastSuccessOptimized;
            // auto &p2 = ptBlock.second->optimizationBuffer;
            // if ((p1 - p2).norm() > 0.1f)
            // {
            //     ptBlock.second->potentialOutlier = true;
            // }

            // p1 = p2;
            ptBlock.second->lastSuccessOptimized = ptBlock.second->optimizationBuffer;
        }
    }

    // return;
    // ceres::Problem problem;
    // size_t numResidualBlock = 0;

    // auto *se3Local = new LocalParameterizationSE3();
    // for (auto kf : cameras)
    // {
    //     if (kf.first < oldestKFId || kf.first > newestKFId)
    //         continue;

    //     problem.AddParameterBlock(kf.second.data(), SE3::num_parameters, se3Local);

    //     if (kf.first != newestKFId || kf.first == oldestKFId)
    //     {
    //         // printf("kf: %lu, fixedKF: %lu\n", kf.first, fixedKfId);
    //         problem.AddParameterBlock(&K[0], 4);
    //         problem.SetParameterBlockConstant(&K[0]);
    //         problem.SetParameterBlockConstant(kf.second.data());
    //     }
    // }

    // for (auto iter = observations.begin(), iend = observations.end(); iter != iend; ++iter)
    // {
    //     auto ptId = iter->first;
    //     if (observations[ptId].size() < 2)
    //         continue;

    //     problem.AddParameterBlock(mapPoints[ptId].data(), 3);
    //     problem.SetParameterBlockConstant(mapPoints[ptId].data());

    //     for (auto iter2 = iter->second.begin(), iend2 = iter->second.end(); iter2 != iend2; ++iter2)
    //     {
    //         auto kfId = iter2->first;
    //         auto obs = iter2->second;

    //         problem.AddResidualBlock(
    //             ReprojectionErrorFunctor::create(obs(0), obs(1)),
    //             NULL,
    //             &K[0],
    //             cameras[kfId].data(),
    //             mapPoints[ptId].data());

    //         numResidualBlock++;
    //     }
    // }

    // printf("total num of %lu residuals to be optimized.\n", numResidualBlock);

    // ceres::Solver::Options options;
    // ceres::Solver::Summary summary;
    // options.linear_solver_type = ceres::SPARSE_SCHUR;
    // options.max_num_iterations = maxiter;
    // Solve(options, &problem, &summary);
    // std::cout << summary.BriefReport() << std::endl;
}

Vec3d CeresSolver::getPtPosOptimized(const size_t ptId) const
{
    auto ptBlock = ptBlockMap.find(ptId);
    if (ptBlock != ptBlockMap.end())
        return ptBlockMap.find(ptId)->second->lastSuccessOptimized;
}

void CeresSolver::reset()
{
    camBlockMap.clear();
    ptBlockMap.clear();
    resBlockMap.clear();
}

bool CeresSolver::hasCamera(const size_t camId) const
{
    return camBlockMap.find(camId) != camBlockMap.end();
}

bool CeresSolver::hasPoint(const size_t ptId) const
{
    return ptBlockMap.find(ptId) != ptBlockMap.end();
}
