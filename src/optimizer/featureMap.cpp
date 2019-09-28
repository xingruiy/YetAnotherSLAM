#include "optimizer/featureMap.h"
#include "optimizer/costFunctors.h"

FeatureMap::FeatureMap(Mat33d &K, int localWinSize)
    : optWinSize(localWinSize), K(K),
      isOptimizing(false), shouldQuit(false), hasNewKF(false)
{
    Kinv = K.inverse();
    matcher = std::make_shared<FeatureMatcher>(PointType::ORB, DescType::ORB);
}

void FeatureMap::reset()
{
    frameHistory.clear();
    keyframeOptWin.clear();
    keyframeHistory.clear();
}

void FeatureMap::addFrameHistory(std::shared_ptr<Frame> frame)
{
    auto refKF = frame->getReferenceKF();
    if (refKF == NULL)
        return;

    frameHistory.push_back(std::make_pair(frame->getTrackingResult(), refKF));
}

void FeatureMap::addReferenceFrame(std::shared_ptr<Frame> frame)
{
    if (frame == NULL)
        return;

    std::unique_lock<std::mutex> lock(bufferMutex);
    newKeyFrameBuffer.push(frame);
}

void FeatureMap::marginalizeOldFrame()
{
    std::shared_ptr<Frame> kf2Del = NULL;

    {
        std::unique_lock<std::mutex> lock(optWinMutex);
        kf2Del = keyframeOptWin.front();
        keyframeOptWin.pop_front();
    }

    kf2Del->inLocalOptimizer = false;

    // for (auto pt : kf2Del->mapPoints)
    // {
    //     if (pt && pt->getHost() == kf2Del)
    //     {
    //         if (pt->observations.size() <= 1)
    //             pt->invalidated = true;
    //     }
    // }

    {
        std::unique_lock<std::mutex> lock(historyMutex);
        keyframeHistory.push_back(kf2Del);
    }
}

std::vector<Vec3f> FeatureMap::getActivePoints()
{
    std::vector<Vec3f> localPoints;

    for (auto kf : keyframeOptWin)
        if (kf)
            for (auto pt : kf->mapPoints)
                if (pt)
                    pt->visited = false;

    for (auto kf : keyframeOptWin)
    {
        std::unique_lock<std::mutex> lock(optimizerMutex);
        if (kf == NULL)
            continue;

        for (auto pt : kf->mapPoints)
            if (pt && !pt->visited)
            {
                pt->visited = true;
                localPoints.push_back(pt->getPosWorld().cast<float>());
            }
    }

    return localPoints;
}

std::vector<Vec3f> FeatureMap::getStablePoints()
{
    std::vector<Vec3f> stablePoints;

    for (auto kf : keyframeHistory)
        if (kf)
            for (auto pt : kf->mapPoints)
                if (pt)
                    pt->visited = false;

    for (auto kf : keyframeHistory)
    {
        if (kf == NULL)
            continue;

        for (auto pt : kf->mapPoints)
            if (pt && !pt->visited)
            {
                pt->visited = true;
                stablePoints.push_back(pt->getPosWorld().cast<float>());
            }
    }

    return stablePoints;
}

std::vector<SE3> FeatureMap::getFrameHistory() const
{
    std::vector<SE3> history;

    for (auto &h : frameHistory)
    {
        auto &dT = h.first;
        auto &refKF = h.second;
        history.push_back(refKF->getPoseInGlobalMap() * dT);
    }

    return history;
}

void FeatureMap::setShouldQuit()
{
    shouldQuit = true;
}

void FeatureMap::optimizationLoop()
{
    while (!shouldQuit)
    {
        std::shared_ptr<Frame> frame = NULL;

        {
            std::unique_lock<std::mutex> lock(bufferMutex);
            if (newKeyFrameBuffer.size() > 0)
            {
                frame = newKeyFrameBuffer.front();
                newKeyFrameBuffer.pop();
            }
        }

        if (frame == NULL)
            continue;

        if (keyframeOptWin.size() >= optWinSize)
            marginalizeOldFrame();

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

        std::vector<float> zVector;
        matcher->detect(image, depth, frame->cvKeyPoints, frame->pointDesc, zVector);
        frame->depthVec = zVector;
        int numDetectedPoints = frame->cvKeyPoints.size();

        Mat displayImage;
        image.copyTo(displayImage);

        if (numDetectedPoints == 0)
        {
            printf("Error: no features detected! keyframe not accepted.\n");
            return;
        }

        std::vector<bool> matchesFound(numDetectedPoints);
        std::fill(matchesFound.begin(), matchesFound.end(), false);
        frame->mapPoints.resize(numDetectedPoints);

        size_t numMatchedPoints = 0;

        for (auto refKF : keyframeOptWin)
        {
            if (refKF == frame)
                continue;

            std::vector<cv::DMatch> matches;
            // matcher->matchByProjection2NN(refKF, frame, K, matches, &matchesFound);
            matcher->matchByProjection2NN(refKF, frame, K, matches, NULL);

            if (matches.size() == 0)
                continue;

            Mat outImg;
            Mat refImg = refKF->getImage();
            cv::drawMatches(refImg, refKF->cvKeyPoints, image, frame->cvKeyPoints, matches, outImg);
            cv::imshow("img", outImg);
            cv::waitKey(1);

            for (auto match : matches)
            {
                auto &pt = refKF->mapPoints[match.queryIdx];
                auto &framePt3d = frame->mapPoints[match.trainIdx];
                auto &framePt = frame->cvKeyPoints[match.trainIdx];
                // auto obs = Vec2d(framePt.pt.x, framePt.pt.y);
                auto z = zVector[match.trainIdx];

                if (!pt || pt == framePt3d)
                    continue;

                if (!framePt3d)
                {
                    frame->mapPoints[match.trainIdx] = pt;
                    // pt->numObservations++;
                    // pt->observations.insert(std::make_pair(frame, ));
                    pt->addObservation(frame, Vec3d(framePt.pt.x, framePt.pt.y, z));
                }
                else
                {
                    // pt->observations.insert(framePt3d->observations.begin(), framePt3d->observations.end());
                    pt->fusePoint(framePt3d);
                    frame->mapPoints[match.trainIdx] = pt;
                }

                numMatchedPoints++;
            }
        }

        std::cout << numMatchedPoints << std::endl;

        auto framePose = frame->getPoseInGlobalMap();
        size_t numCreatedPoints = 0;

        for (int i = 0; i < numDetectedPoints; ++i)
        {
            // if (matchesFound[i])
            //     continue;
            if ((numMatchedPoints + numCreatedPoints) > 300)
                break;
            if (frame->mapPoints[i] != NULL)
            {
                cv::drawMarker(displayImage, frame->cvKeyPoints[i].pt, cv::Scalar(0, 0, 255), cv::MARKER_SQUARE);
                continue;
            }
            else
            {
                // cv::drawMarker(displayImage, frame->cvKeyPoints[i].pt, cv::Scalar(0, 255, 0));
            }

            const auto &kp = frame->cvKeyPoints[i];
            const auto &desc = frame->pointDesc.row(i);
            const auto &z = zVector[i];

            if (z > FLT_EPSILON)
            {
                auto pt3d = std::make_shared<MapPoint>();

                pt3d->setHost(frame);
                pt3d->setPosWorld(framePose * (Kinv * Vec3d(kp.pt.x, kp.pt.y, 1.0) * z));
                pt3d->setDescriptor(desc);
                pt3d->addObservation(frame, Vec3d(kp.pt.x, kp.pt.y, z));
                frame->mapPoints[i] = pt3d;
                numCreatedPoints++;
            }
        }

        cv::imshow("features", displayImage);
        cv::waitKey(1);

        // add to frame history
        {
            std::unique_lock<std::mutex> lock(historyMutex);
            frameHistory.push_back(std::pair<SE3, std::shared_ptr<Frame>>(SE3(), frame));
        }
        // add to optimization window
        {
            std::unique_lock<std::mutex> lock(optWinMutex);
            keyframeOptWin.push_back(frame);
        }

        addToOptimizer(frame);
        windowedOptimization(15);
    }
}

void FeatureMap::findPointCorrespondences(std::shared_ptr<Frame> kf, std::vector<std::shared_ptr<MapPoint>> mapPoints)
{
}

std::vector<std::shared_ptr<Frame>> FeatureMap::findCloseLoopCandidate(std::shared_ptr<Frame> frame)
{
    std::vector<std::shared_ptr<Frame>> candidates;
    const float distTh = 1; // meters
    auto framePosition = frame->getPoseInGlobalMap().translation();

    std::vector<std::shared_ptr<Frame>> keyframeHistoryCopy;
    {
        std::unique_lock<std::mutex> lock(historyMutex);
        keyframeHistoryCopy = keyframeHistory;
    }

    for (auto kf : keyframeHistoryCopy)
    {
        // if (std::abs(frame->getKeyframeId() - kf->getKeyframeId()) < 50)
        //     continue;

        auto kfPosition = kf->getPoseInGlobalMap().translation();
        auto dist = (kfPosition - framePosition).norm();

        if (dist < distTh)
            candidates.push_back(kf);
    }

    return candidates;
}

void FeatureMap::globalConsistencyLoop()
{
    while (!shouldQuit)
    {
        std::shared_ptr<Frame> loopKF = NULL;

        {
            std::unique_lock<std::mutex> lock(loopBufferMutex);
            if (loopKeyFrameBuffer.size() == 0)
                continue;
            loopKF = loopKeyFrameBuffer.front();
            loopKeyFrameBuffer.pop();
        }

        if (loopKF == NULL)
            continue;

        auto candidateClose = findCloseLoopCandidate(loopKF);

        for (auto candidateKF : candidateClose)
        {
            std::vector<cv::DMatch> matches;
            matcher->matchByDescriptor(candidateKF, loopKF, K, matches);

            cv::Mat outImg;
            cv::drawMatches(loopKF->getImage(),
                            loopKF->cvKeyPoints,
                            candidateKF->getImage(),
                            candidateKF->cvKeyPoints,
                            matches, outImg);

            cv::imshow("img2", outImg);
            cv::waitKey(1);

            auto numSuccessMatches = matches.size();

            if (numSuccessMatches < 100)
                continue;
        }
    }
}

void FeatureMap::addToOptimizer(std::shared_ptr<Frame> kf)
{
    kf->inLocalOptimizer = true;
    // solver->addCamera(kf->getKeyframeId(), kf->getPoseInGlobalMap().inverse(), false);
    // for (int i = 0; i < kf->cvKeyPoints.size(); ++i)
    // {
    //     auto pt = kf->mapPoints[i];
    //     if (pt)
    //     {
    //         const auto &kp = kf->cvKeyPoints[i];

    //         if (!pt->inOptimizer)
    //         {
    //             solver->addWorldPoint(pt->id, ptgetPosWorld(), true);
    //             pt->inOptimizer = true;
    //         }

    //         solver->addObservation(pt->id, kf->getKeyframeId(), Vec2d(kp.pt.x, kp.pt.y));
    //     }
    // }
}

void FeatureMap::windowedOptimization(const int maxIteration)
{
    std::vector<std::shared_ptr<Frame>> localKFs;

    {
        std::unique_lock<std::mutex> lock(optWinMutex);
        localKFs = std::vector<std::shared_ptr<Frame>>(keyframeOptWin.begin(), keyframeOptWin.end());
    }

    if (localKFs.size() < optWinSize)
        return;

    ceres::Problem problem;
    std::set<std::shared_ptr<MapPoint>> localPoints;

    for (auto kf : localKFs)
    {
        auto &framePoints = kf->mapPoints;

        for (auto pt : framePoints)
        {
            if (pt)
                localPoints.insert(pt);
        }
    }

    std::set<std::shared_ptr<Frame>> fixedKFs;
    for (auto pt : localPoints)
    {
        if (!pt || pt->getNumObservations() == 0)
            continue;

        for (auto obs : pt->getObservations())
        {
            if (!obs.first->inLocalOptimizer)
                fixedKFs.insert(obs.first);
        }
    }

    if (fixedKFs.size() == 0)
        return;

    for (auto kf : localKFs)
    {
        // printf("local: %lu\n", kf->getKeyframeId());
        problem.AddParameterBlock(kf->getParameterBlock(), SE3::num_parameters, new LocalParameterizationSE3());
    }
    for (auto kf : fixedKFs)
    {
        // printf("fixed: %lu\n", kf->getKeyframeId());
        problem.AddParameterBlock(kf->getParameterBlock(), SE3::num_parameters, new LocalParameterizationSE3());
        problem.SetParameterBlockConstant(kf->getParameterBlock());
    }

    std::cout << "num local: " << localKFs.size() << " num fixed: " << fixedKFs.size() << std::endl;

    double KBlock[4] = {K(0, 0), K(1, 1), K(0, 2), K(1, 2)};
    ceres::LossFunction *lossFunc = new ceres::HuberLoss(10);

    size_t numResidualBlocks = 0;
    for (auto pt : localPoints)
    {
        if (!pt || pt->getNumObservations() == 0)
            continue;

        for (auto obs : pt->getObservations())
        {
            // problem.AddResidualBlock(
            //     ReprojectionErrorFunctor::create(obs.second(0), obs.second(1)),
            //     NULL,
            //     &KBlock[0],
            //     obs.first->getParameterBlock(),
            //     pt->getParameterBlock());

            problem.AddResidualBlock(
                ReprojectionError3DFunctor::create(obs.second),
                lossFunc,
                &KBlock[0],
                obs.first->getParameterBlock(),
                pt->getParameterBlock());

            numResidualBlocks++;
        }

        // problem.SetParameterBlockConstant(pt->getParameterBlock());
    }

    if (numResidualBlocks == 0)
        return;

    problem.SetParameterBlockConstant(&KBlock[0]);

    std::cout << "start optimization with residual blocks: " << numResidualBlocks << std::endl;
    ceres::Solver::Options options;
    options.linear_solver_type = ceres::SPARSE_SCHUR;
    ceres::Solver::Summary summary;
    Solve(options, &problem, &summary);
    std::cout << summary.BriefReport() << std::endl;

    // solver->optimize(maxIteration);

    // {
    //     std::unique_lock<std::mutex> lock(optWinMutex);
    //     std::unique_lock<std::mutex> lock2(optimizerMutex);
    //     for (auto kf : keyframeOptWin)
    //     {
    //         // for (auto pt : kf->mapPoints)
    //         // {
    //         //     if (pt && pt->inOptimizer && !pt->invalidated)
    //         //     {

    //         //         auto rval = solver->getPtPosOptimized(pt->id);
    //         //         if (!rval.isApprox(Vec3d()))
    //         //             ptgetPosWorld() = rval.cast<double>();
    //         //     }
    //         // }

    //         auto poseOpt = solver->getCamPoseOptimized(kf->getKeyframeId()).inverse();
    //         kf->setOptimizationResult(poseOpt);
    //     }

    std::unique_lock<std::mutex> lock3(loopBufferMutex);
    loopKeyFrameBuffer.push(keyframeOptWin.back());
    // }
}

bool FeatureMap::hasUnfinishedWork()
{
    std::unique_lock<std::mutex> lock(bufferMutex);
    return isOptimizing && newKeyFrameBuffer.size() > 0;
}

std::vector<SE3> FeatureMap::getKeyFrameHistory()
{
    std::vector<SE3> history;

    {
        std::unique_lock<std::mutex> lock(historyMutex);
        for (auto kf : keyframeHistory)
        {
            history.push_back(kf->getPoseInGlobalMap());
        }
    }

    {
        std::unique_lock<std::mutex> lock(optWinMutex);
        for (auto kf : keyframeOptWin)
        {
            history.push_back(kf->getPoseInGlobalMap());
        }
    }

    return history;
}