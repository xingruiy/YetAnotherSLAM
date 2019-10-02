#include "loopCloser.h"
#include "optimizer/costFunctors.h"

LoopCloser::LoopCloser(const Mat33d &K, std::shared_ptr<Map> map)
    : shouldQuit(false), map(map), K(K)
{
}

void LoopCloser::loop()
{
    while (!shouldQuit)
    {
        auto frame = map->getLoopClosingKeyframe();
        if (frame == NULL)
            continue;

        auto kpAll = map->getMapPointsAll();
        }
}

void LoopCloser::setShouldQuit()
{
    shouldQuit = true;
}

void LoopCloser::setMap(std::shared_ptr<Map> map)
{
    this->map = map;
}

void LoopCloser::optimize(
    std::vector<std::shared_ptr<Frame>> kfs,
    std::vector<std::shared_ptr<MapPoint>> pts,
    const int maxIter)
{
    ceres::Problem problem;
    for (int i = 0; i < kfs.size(); ++i)
    {
        problem.AddParameterBlock(
            kfs[i]->getParameterBlock(),
            SE3::num_parameters,
            new LocalParameterizationSE3());

        if (i == 0)
            problem.SetParameterBlockConstant(kfs[i]->getParameterBlock());
    }

    double KBlock[4] = {K(0, 0), K(1, 1), K(0, 2), K(1, 2)};
    ceres::LossFunction *lossFunc = new ceres::HuberLoss(10);

    size_t numResidualBlocks = 0;
    for (auto pt : pts)
    {
        if (!pt || pt->getNumObservations() == 0)
            continue;

        for (auto obs : pt->getObservations())
        {
            //     problem.AddResidualBlock(
            //         ReprojectionErrorFunctor::create(obs.second(0), obs.second(1)),
            //         NULL,
            //         &KBlock[0],
            //         obs.first->getParameterBlock(),
            //         pt->getParameterBlock());

            problem.AddResidualBlock(
                ReprojectionError3DFunctor::create(obs.second),
                lossFunc,
                &KBlock[0],
                obs.first->getParameterBlock(),
                pt->getParameterBlock());

            numResidualBlocks++;
        }
    }

    if (numResidualBlocks == 0)
        return;

    problem.SetParameterBlockConstant(&KBlock[0]);

    std::cout << "start bundle adjustment with keyframes: " << kfs.size() << " points : " << pts.size() << " residual blocks : " << numResidualBlocks << std::endl;
    ceres::Solver::Options options;
    options.linear_solver_type = ceres::ITERATIVE_SCHUR;
    options.preconditioner_type = ceres::SCHUR_JACOBI;
    options.use_explicit_schur_complement = false;
    ceres::Solver::Summary summary;
    Solve(options, &problem, &summary);
    std::cout << summary.BriefReport() << std::endl;
}
