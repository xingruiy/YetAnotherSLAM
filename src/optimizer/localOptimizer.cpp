#include "optimizer/localOptimizer.h"
#include "optimizer/costFunctors.h"

LocalOptimizer::LocalOptimizer(
    Mat33d &K,
    int localWinSize,
    std::shared_ptr<Map> map)
    : K(K),
      shouldQuit(false),
      map(map),
      pauseMapping(false)
{
    matcher = std::make_shared<FeatureMatcher>(PointType::ORB, DescType::ORB);
}

void LocalOptimizer::reset()
{
}

void LocalOptimizer::setMap(std::shared_ptr<Map> map)
{
    this->map = map;
}

void LocalOptimizer::setShouldQuit()
{
    shouldQuit = true;
}

void LocalOptimizer::loop()
{
    while (!shouldQuit)
    {
        std::shared_ptr<Frame> frame = map->getUnprocessedKeyframe();

        if (frame == NULL)
            continue;

        Mat image = frame->getImage();
        Mat depth = frame->getDepth();
        Mat intensity = frame->getIntensity();

        auto refKF = frame->getReferenceKF();
        if (refKF)
        {
            auto dT = frame->getTrackingResult();
            auto refT = refKF->getPoseInGlobalMap();
            auto T = refT * dT;
            frame->setOptimizationResult(T);
        }

        matcher->detect(
            image, depth,
            frame->cvKeyPoints,
            frame->pointDesc,
            frame->depthVec);

        int numDetectedPoints = frame->cvKeyPoints.size();

        Mat displayImage;
        image.copyTo(displayImage);

        if (numDetectedPoints == 0)
        {
            printf("Error: no features detected! keyframe not accepted.\n");
            return;
        }

        frame->mapPoints.resize(numDetectedPoints);
        size_t numMatchedPoints = 0;

        if (!pauseMapping)
        {

            std::set<std::shared_ptr<MapPoint>> mapPointTemp;
            auto localKFs = map->getLastNKeyframes(5);
            for (auto kf : localKFs)
            {
                for (auto pt : kf->mapPoints)
                    if (pt && !pt->isBad())
                        mapPointTemp.insert(pt);
            }

            auto mapPointAll = std::vector<std::shared_ptr<MapPoint>>(
                mapPointTemp.begin(), mapPointTemp.end());

            std::vector<cv::DMatch> matches;
            matcher->matchByProjection2NN(mapPointAll, frame, K, matches, NULL);

            for (auto match : matches)
            {
                auto &pt = mapPointAll[match.queryIdx];
                auto &framePt3d = frame->mapPoints[match.trainIdx];
                auto &framePt = frame->cvKeyPoints[match.trainIdx];
                auto z = frame->depthVec[match.trainIdx];

                if (!pt || pt == framePt3d)
                    continue;

                if (!framePt3d)
                {
                    pt->addObservation(frame, Vec3d(framePt.pt.x, framePt.pt.y, z));
                }
                else if ((framePt3d->getPosWorld() - pt->getPosWorld()).norm() < 0.05)
                {
                    pt->fusePoint(framePt3d);
                }

                frame->mapPoints[match.trainIdx] = pt;
                numMatchedPoints++;
            }

            auto framePose = frame->getPoseInGlobalMap();
            size_t numCreatedPoints = 0;
            for (int i = 0; i < numDetectedPoints; ++i)
            {
                if ((numMatchedPoints + numCreatedPoints) > 400)
                    break;

                if (frame->mapPoints[i] != NULL)
                {
                    cv::drawMarker(displayImage, frame->cvKeyPoints[i].pt, cv::Scalar(0, 0, 255), cv::MARKER_SQUARE);
                    continue;
                }
                else
                    cv::drawMarker(displayImage, frame->cvKeyPoints[i].pt, cv::Scalar(0, 255, 0), cv::MARKER_CROSS);

                const auto &kp = frame->cvKeyPoints[i];
                const auto &desc = frame->pointDesc.row(i);
                const auto &z = frame->depthVec[i];

                if (z > FLT_EPSILON)
                {
                    auto pt3d = std::make_shared<MapPoint>();

                    pt3d->setHost(frame);
                    pt3d->setPosWorld(framePose * (K.inverse() * Vec3d(kp.pt.x, kp.pt.y, 1.0) * z));
                    pt3d->setDescriptor(desc);
                    pt3d->addObservation(frame, Vec3d(kp.pt.x, kp.pt.y, z));
                    frame->mapPoints[i] = pt3d;
                    if (!pauseMapping)
                        map->addMapPoint(pt3d);
                    numCreatedPoints++;
                }
            }

            cv::imshow("features", displayImage);
            cv::waitKey(1);

            map->addKeyFrame(frame);
            optimize(localKFs, mapPointAll, 50);
        }

        map->addLoopClosingKeyframe(frame);
    }
}

void LocalOptimizer::optimize(
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

        // if (kfs[i]->getId() == 0)
        //     problem.SetParameterBlockConstant(kfs[i]->getParameterBlock());

        kfs[i]->kfIdLocalRoot = kfs[0]->getId();
    }

    std::set<std::shared_ptr<Frame>> fixedKFs;
    for (auto pt : pts)
    {
        if (!pt || pt->isBad() || pt->getNumObservations() == 0)
            continue;

        for (auto obs : pt->getObservations())
        {
            if (obs.first->kfIdLocalRoot != kfs[0]->getId())
                fixedKFs.insert(obs.first);
        }
    }

    if (fixedKFs.size() == 0)
        return;

    for (auto kf : fixedKFs)
    {
        problem.AddParameterBlock(
            kf->getParameterBlock(),
            SE3::num_parameters,
            new LocalParameterizationSE3());
        problem.SetParameterBlockConstant(kf->getParameterBlock());
    }

    double KBlock[4] = {K(0, 0), K(1, 1), K(0, 2), K(1, 2)};
    ceres::LossFunction *lossFunc = new ceres::HuberLoss(10);

    size_t numResidualBlocks = 0;
    for (auto pt : pts)
    {
        if (!pt || pt->isBad() || pt->getNumObservations() == 0)
            continue;

        for (auto obs : pt->getObservations())
        {
            problem.AddResidualBlock(
                ReprojectionErrorFunctor::create(obs.second(0), obs.second(1)),
                NULL,
                &KBlock[0],
                obs.first->getParameterBlock(),
                pt->getParameterBlock());

            // problem.AddResidualBlock(
            //     ReprojectionError3DFunctor::create(obs.second),
            //     lossFunc,
            //     &KBlock[0],
            //     obs.first->getParameterBlock(),
            //     pt->getParameterBlock());

            numResidualBlocks++;
        }

        // problem.SetParameterBlockConstant(pt->getParameterBlock());
    }

    if (numResidualBlocks == 0)
        return;

    problem.SetParameterBlockConstant(&KBlock[0]);

    // K << KBlock[0], 0, KBlock[2],
    //     0, KBlock[1], KBlock[3],
    //     0, 0, 1;
    // Kinv = K.inverse();

    // std::cout << "start bundleAdjustment with keyframes: " << kfs.size() << " points : " << pts.size() << " residual blocks : " << numResidualBlocks << std::endl;
    ceres::Solver::Options options;
    options.max_num_iterations = maxIter;
    options.update_state_every_iteration = true;
    options.linear_solver_type = ceres::SPARSE_SCHUR;
    ceres::Solver::Summary summary;
    Solve(options, &problem, &summary);
    std::cout << summary.BriefReport() << std::endl;
}
