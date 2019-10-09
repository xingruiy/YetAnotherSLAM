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

void LocalOptimizer::optimize(std::shared_ptr<Frame> kf)
{
    ceres::Problem problem;
    problem.AddParameterBlock(kf->getParameterBlock(), SE3::num_parameters, new LocalParameterizationSE3);
    // problem.SetParameterBlockConstant(kf->getParameterBlock());

    double KBlock[4] = {K(0, 0), K(1, 1), K(0, 2), K(1, 2)};
    problem.AddParameterBlock(&KBlock[0], 4);
    problem.SetParameterBlockConstant(&KBlock[0]);

    std::vector<Vec3d> before;
    for (auto pt : kf->mapPoints)
    {
        if (pt)
            before.push_back(pt->getPosWorld());
        else
            before.push_back(Vec3d());
    }

    std::set<std::shared_ptr<Frame>> fixedKFs;
    for (auto pt : kf->mapPoints)
    {
        if (pt && pt->getNumObservations() > 1)
        {
            for (auto obs : pt->getObservations())
            {
                if (obs.first != kf && fixedKFs.find(obs.first) == fixedKFs.end())
                    fixedKFs.insert(obs.first);
            }
        }
    }

    std::cout << fixedKFs.size() << std::endl;

    for (auto frame : fixedKFs)
    {
        problem.AddParameterBlock(frame->getParameterBlock(), SE3::num_parameters, new LocalParameterizationSE3);
        problem.SetParameterBlockConstant(frame->getParameterBlock());
    }

    size_t numResidualBlocks = 0;
    for (auto pt : kf->mapPoints)
    {
        if (pt && pt->getNumObservations() > 1)
        {
            for (auto obs : pt->getObservations())
            {
                problem.AddResidualBlock(
                    ReprojectionErrorFunctor::create(obs.second(0), obs.second(1)),
                    NULL,
                    &KBlock[0],
                    obs.first->getParameterBlock(),
                    pt->getParameterBlock());

                numResidualBlocks++;
            }

            problem.SetParameterBlockConstant(pt->getParameterBlock());
        }
    }
    // std::cout << numResidualBlocks << std::endl;

    ceres::Solver::Options options;
    ceres::Solver::Summary summary;
    options.linear_solver_type = ceres::DENSE_SCHUR;
    Solve(options, &problem, &summary);
    std::cout << summary.BriefReport() << std::endl;

    // for (int i = 0; i < kf->mapPoints.size(); ++i)
    // {
    //     auto pt = kf->mapPoints[i];
    //     if (!pt)
    //         continue;
    //     auto b = before[i];
    //     auto c = pt->getPosWorld();
    //     auto diff = (b - c).norm();
    //     if (diff > 0.4)
    //     {
    //         std::cout << "pt before : " << b << std::endl;
    //         std::cout << "pt after : " << c << std::endl;

    //         auto obs = pt->getObservations();
    //         std::cout << "Num obs: " << obs.size() << std::endl;
    //         for (auto ob : obs)
    //         {
    //             auto &kf = ob.first;
    //             auto &d = ob.second;
    //             std::cout << kf->getPoseInGlobalMap().matrix3x4() << std::endl;
    //             std::cout << d << std::endl;
    //             Vec2d val = d.head<2>();

    //             auto e = kf->getPoseInGlobalMap().inverse() * b;
    //             Vec2d proj = {KBlock[0] * e(0) / e(2) + KBlock[2], KBlock[1] * e(1) / e(2) + KBlock[3]};

    //             std::cout << "before : " << (proj - val).norm() << std::endl;
    //             std::cout << proj << std::endl;

    //             e = kf->getPoseInGlobalMap().inverse() * pt->getPosWorld();
    //             proj = {KBlock[0] * e(0) / e(2) + KBlock[2], KBlock[1] * e(1) / e(2) + KBlock[3]};
    //             std::cout << "after: " << (proj - val).norm() << std::endl;
    //             std::cout << proj << std::endl;
    //         }
    //     }
    // }
}

void LocalOptimizer::optimizePoints(std::shared_ptr<Frame> kf)
{
    ceres::Problem problem;
    problem.AddParameterBlock(kf->getParameterBlock(), SE3::num_parameters, new LocalParameterizationSE3);
    problem.SetParameterBlockConstant(kf->getParameterBlock());

    double KBlock[4] = {K(0, 0), K(1, 1), K(0, 2), K(1, 2)};
    problem.AddParameterBlock(&KBlock[0], 4);
    problem.SetParameterBlockConstant(&KBlock[0]);

    std::vector<Vec3d> before;
    for (auto pt : kf->mapPoints)
    {
        if (pt)
            before.push_back(pt->getPosWorld());
        else
            before.push_back(Vec3d());
    }

    std::set<std::shared_ptr<Frame>> fixedKFs;
    for (auto pt : kf->mapPoints)
    {
        if (pt && pt->getNumObservations() > 1)
        {
            for (auto obs : pt->getObservations())
            {
                if (obs.first != kf && fixedKFs.find(obs.first) == fixedKFs.end())
                    fixedKFs.insert(obs.first);
            }
        }
    }

    std::cout << fixedKFs.size() << std::endl;

    for (auto frame : fixedKFs)
    {
        problem.AddParameterBlock(frame->getParameterBlock(), SE3::num_parameters, new LocalParameterizationSE3);
        problem.SetParameterBlockConstant(frame->getParameterBlock());
    }

    size_t numResidualBlocks = 0;
    for (auto pt : kf->mapPoints)
    {
        if (pt && pt->getNumObservations() > 1)
        {
            for (auto obs : pt->getObservations())
            {
                problem.AddResidualBlock(
                    ReprojectionErrorFunctor::create(obs.second(0), obs.second(1)),
                    NULL,
                    &KBlock[0],
                    obs.first->getParameterBlock(),
                    pt->getParameterBlock());

                numResidualBlocks++;
            }
        }
    }

    ceres::Solver::Options options;
    ceres::Solver::Summary summary;
    options.linear_solver_type = ceres::DENSE_SCHUR;
    Solve(options, &problem, &summary);
    std::cout << summary.BriefReport() << std::endl;

    // for (int i = 0; i < kf->mapPoints.size(); ++i)
    // {
    //     auto pt = kf->mapPoints[i];
    //     if (!pt)
    //         continue;
    //     auto b = before[i];
    //     auto c = pt->getPosWorld();
    //     auto diff = (b - c).norm();
    //     if (diff > 0.4)
    //     {
    //         std::cout << "pt before : " << b << std::endl;
    //         std::cout << "pt after : " << c << std::endl;

    //         auto obs = pt->getObservations();
    //         std::cout << "Num obs: " << obs.size() << std::endl;
    //         for (auto ob : obs)
    //         {
    //             auto &kf = ob.first;
    //             auto &d = ob.second;
    //             std::cout << kf->getPoseInGlobalMap().matrix3x4() << std::endl;
    //             std::cout << d << std::endl;
    //             Vec2d val = d.head<2>();

    //             auto e = kf->getPoseInGlobalMap().inverse() * b;
    //             Vec2d proj = {KBlock[0] * e(0) / e(2) + KBlock[2], KBlock[1] * e(1) / e(2) + KBlock[3]};

    //             std::cout << "before : " << (proj - val).norm() << std::endl;
    //             std::cout << proj << std::endl;

    //             e = kf->getPoseInGlobalMap().inverse() * pt->getPosWorld();
    //             proj = {KBlock[0] * e(0) / e(2) + KBlock[2], KBlock[1] * e(1) / e(2) + KBlock[3]};
    //             std::cout << "after: " << (proj - val).norm() << std::endl;
    //             std::cout << proj << std::endl;
    //         }
    //     }
    // }
}

void LocalOptimizer::loop()
{
    while (!shouldQuit)
    {
        auto frame = map->getUnprocessedKeyframe();

        if (frame == NULL)
            continue;

        auto refKF = frame->getReferenceKF();
        if (refKF)
        {
            auto dT = frame->getTrackingResult();
            auto refT = refKF->getPoseInGlobalMap();
            auto T = refT * dT;
            frame->setOptimizationResult(T);
        }

        frame->detectKeyPoints(matcher);
        int numDetectedPoints = frame->cvKeyPoints.size();
        if (numDetectedPoints == 0)
        {
            printf("Error: no features detected! keyframe not accepted.\n");
            return;
        }

        std::vector<cv::DMatch> matches;

        auto kf = map->getCurrentKeyframe();
        if (kf != NULL)
        {
            std::vector<bool> matchesFound(numDetectedPoints);
            std::fill(matchesFound.begin(), matchesFound.end(), false);
            matcher->matchByProjection2NN(kf, frame, K, matches, NULL);
            Mat outImg;
            cv::drawMatches(kf->getImage(), kf->cvKeyPoints, frame->getImage(), frame->cvKeyPoints, matches, outImg);
            cv::imshow("img", outImg);
            cv::waitKey(1);
        }

        for (auto match : matches)
        {
            frame->mapPoints[match.trainIdx] = kf->mapPoints[match.queryIdx];
            frame->mapPoints[match.trainIdx]->addObservation(frame, Vec3d(frame->cvKeyPoints[match.trainIdx].pt.x, frame->cvKeyPoints[match.trainIdx].pt.y, frame->depthVec[match.trainIdx]));
        }

        optimize(frame);
        optimizePoints(frame);

        for (int i = 0; i < numDetectedPoints; ++i)
        {
            // if (frame->mapPoints[i] != NULL)
            //     continue;

            const auto &z = frame->depthVec[i];
            if (z > FLT_EPSILON)
            {
                const auto &kp = frame->cvKeyPoints[i];
                Vec3d pos = frame->getPoseInGlobalMap() * (K.inverse() * Vec3d(kp.pt.x, kp.pt.y, 1.0) * z);

                auto pt = std::make_shared<MapPoint>();
                pt->setHost(frame);
                pt->setPosWorld(pos);
                pt->setDescriptor(frame->pointDesc.row(i));
                pt->addObservation(frame, Vec3d(kp.pt.x, kp.pt.y, z));
                frame->mapPoints[i] = pt;
                map->addMapPoint(pt);
            }
        }

        map->addKeyFrame(frame);
        map->setCurrentKeyframe(frame);
        map->addLoopClosingKeyframe(frame);
    }
}

// void LocalOptimizer::loop()
// {
//     while (!shouldQuit)
//     {
//         auto frame = map->getUnprocessedKeyframe();
//         if (frame == NULL)
//             continue;

//         Mat image = frame->getImage();
//         Mat depth = frame->getDepth();
//         Mat intensity = frame->getIntensity();

//         auto refKF = frame->getReferenceKF();
//         if (refKF)
//         {
//             auto dT = frame->getTrackingResult();
//             auto refT = refKF->getPoseInGlobalMap();
//             auto T = refT * dT;
//             frame->setOptimizationResult(T);
//         }

//         frame->detectKeyPoints(matcher);
//         int numDetectedPoints = frame->cvKeyPoints.size();

//         Mat displayImage;
//         image.copyTo(displayImage);

//         if (numDetectedPoints == 0)
//         {
//             printf("Error: no features detected! keyframe not accepted.\n");
//             return;
//         }

//         frame->mapPoints.resize(numDetectedPoints);
//         size_t numMatchedPoints = 0;

//         std::set<std::shared_ptr<MapPoint>> mapPointTemp;
//         auto localKFs = map->getLastNKeyframes(1);
//         for (auto kf : localKFs)
//         {
//             for (auto pt : kf->mapPoints)
//                 if (pt && !pt->isBad())
//                     mapPointTemp.insert(pt);
//         }

//         auto mapPointAll = std::vector<std::shared_ptr<MapPoint>>(
//             mapPointTemp.begin(), mapPointTemp.end());

//         std::vector<cv::DMatch> matches;
//         matcher->matchByProjection2NN(mapPointAll, frame, K, matches, NULL);

//         for (auto match : matches)
//         {
//             auto &pt = mapPointAll[match.queryIdx];
//             auto &framePt3d = frame->mapPoints[match.trainIdx];
//             auto &framePt = frame->cvKeyPoints[match.trainIdx];
//             auto z = frame->depthVec[match.trainIdx];

//             if (!pt || pt == framePt3d)
//                 continue;

//             if (!framePt3d)
//             {
//                 pt->addObservation(frame, Vec3d(framePt.pt.x, framePt.pt.y, z));
//             }
//             // else if ((framePt3d->getPosWorld() - pt->getPosWorld()).norm() < 0.05)
//             // {
//             // pt->fusePoint(framePt3d);
//             // }
//             else
//             {
//                 continue;
//             }

//             // frame->mapPoints[match.trainIdx] = pt;
//             frame->setMapPoint(pt, match.trainIdx);
//             numMatchedPoints++;
//         }

//         // cv::imshow("features", displayImage);
//         // cv::waitKey(1);

//         localKFs.push_back(frame);
//         optimize(localKFs, mapPointAll, 50);

//         auto framePose = frame->getPoseInGlobalMap();
//         size_t numCreatedPoints = 0;
//         for (int i = 0; i < numDetectedPoints; ++i)
//         {
//             // if ((numMatchedPoints + numCreatedPoints) > 400)
//             //     break;

//             if (frame->mapPoints[i] != NULL)
//                 continue;
//             // {
//             //     cv::drawMarker(displayImage, frame->cvKeyPoints[i].pt, cv::Scalar(0, 0, 255), cv::MARKER_SQUARE);
//             //     continue;
//             // }
//             // else
//             //     cv::drawMarker(displayImage, frame->cvKeyPoints[i].pt, cv::Scalar(0, 255, 0), cv::MARKER_CROSS);

//             auto pt = frame->createMapPoint(i);
//             if (pt)
//             {
//                 pt->setHost(frame);
//                 frame->setMapPoint(pt, i);
//                 pt->addObservation(frame, Vec3d(frame->cvKeyPoints[i].pt.x, frame->cvKeyPoints[i].pt.y, frame->depthVec[i]));
//                 map->addMapPoint(pt);
//                 numCreatedPoints++;
//             }
//         }

//         map->addKeyFrame(frame);
//         map->addLoopClosingKeyframe(frame);
//     }
// }

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
            new LocalParameterizationSE3);

        // if (kfs[i]->getId() == 0)
        problem.SetParameterBlockConstant(kfs[i]->getParameterBlock());

        kfs[i]->kfIdLocalRoot = kfs.back()->getId();
    }

    std::set<std::shared_ptr<Frame>> fixedKFs;
    for (auto pt : pts)
    {
        if (!pt || pt->isBad() || pt->getNumObservations() < 2)
            continue;

        for (auto obs : pt->getObservations())
        {
            if (obs.first->kfIdLocalRoot != kfs.back()->getId())
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
            new LocalParameterizationSE3);
        problem.SetParameterBlockConstant(kf->getParameterBlock());
    }

    double KBlock[4] = {K(0, 0), K(1, 1), K(0, 2), K(1, 2)};
    // ceres::LossFunction *lossFunc = new ceres::HuberLoss(10);
    ceres::LossFunction *lossFunc = NULL;

    size_t numResidualBlocks = 0;
    for (auto pt : pts)
    {
        if (!pt || pt->isBad() || pt->getNumObservations() < 2)
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

    for (auto pt : pts)
    {
    }

    if (numResidualBlocks == 0)
        return;

    problem.SetParameterBlockConstant(&KBlock[0]);

    // K << KBlock[0], 0, KBlock[2],
    //     0, KBlock[1], KBlock[3],
    //     0, 0, 1;
    // Kinv = K.inverse();

    std::cout << "start bundleAdjustment with keyframes: " << kfs.size() << " points : " << pts.size() << " residual blocks : " << numResidualBlocks << std::endl;
    ceres::Solver::Options options;
    options.max_num_iterations = maxIter;
    // options.update_state_every_iteration = true;
    options.linear_solver_type = ceres::SPARSE_SCHUR;
    ceres::Solver::Summary summary;
    Solve(options, &problem, &summary);
    std::cout << summary.BriefReport() << std::endl;
}
