#include "globalMapper/globalMapper.h"
#include <ceres/ceres.h>

GlobalMapper::GlobalMapper(Mat33d &K, int localWinSize)
    : optWinSize(localWinSize), K(K),
      isOptimizing(false), shouldQuit(false), hasNewKF(false)
{
    Kinv = K.inverse();
    solver = std::make_shared<CeresSolver>(K);
    matcher = std::make_shared<FeatureMatcher>(PointType::ORB, DescType::ORB);
}

void GlobalMapper::reset()
{
    frameHistory.clear();
    keyframeOptWin.clear();
    keyframeHistory.clear();
}

void GlobalMapper::addFrameHistory(std::shared_ptr<Frame> frame)
{
    auto refKF = frame->getReferenceKF();
    if (refKF == NULL)
        return;

    frameHistory.push_back(std::make_pair(frame->getTrackingResult(), refKF));
}

void GlobalMapper::addReferenceFrame(std::shared_ptr<Frame> frame)
{
    if (frame == NULL)
        return;

    std::unique_lock<std::mutex> lock(bufferMutex);
    newKeyFrameBuffer.push(frame);
}

void GlobalMapper::marginalizeOldFrame()
{
    std::shared_ptr<Frame> kf2Del = NULL;

    {
        std::unique_lock<std::mutex> lock(optWinMutex);
        kf2Del = keyframeOptWin.front();
        keyframeOptWin.pop_front();
    }

    kf2Del->inLocalOptimizer = false;

    for (auto pt : kf2Del->mapPoints)
    {
        if (pt && pt->hostKF == kf2Del)
        {
            if (pt->observations.size() <= 1)
                pt->invalidated = true;
        }
    }

    // for (auto pt : kf2Del->mapPoints)
    // {
    //     if (pt && pt->inOptimizer && pt->hostKF == kf2Del)
    //     {
    //         if (pt->hostKF == kf2Del)
    //         {
    //             solver->removeWorldPoint(pt->id);
    //             pt->inOptimizer = false;
    //             pt->invalidated = true;
    //             if (pt->numObservations <= 1)
    //                 pt = NULL;
    //         }
    //         else
    //             solver->removeObservation(pt->id, kf2Del->getKeyframeId());
    //     }
    // }

    // solver->removeCamera(kf2Del->getKeyframeId());

    {
        std::unique_lock<std::mutex> lock(historyMutex);
        keyframeHistory.push_back(kf2Del);
    }
}

std::vector<Vec3f> GlobalMapper::getActivePoints()
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
            if (pt && !pt->visited && !pt->invalidated && !pt->isImmature)
            {
                pt->visited = true;
                localPoints.push_back(pt->position.cast<float>());
            }
    }

    return localPoints;
}

std::vector<Vec3f> GlobalMapper::getStablePoints()
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
            if (pt && !pt->visited && !pt->invalidated && !pt->isImmature)
            {
                pt->visited = true;
                stablePoints.push_back(pt->position.cast<float>());
            }
    }

    return stablePoints;
}

std::vector<SE3> GlobalMapper::getFrameHistory() const
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

void GlobalMapper::setShouldQuit()
{
    shouldQuit = true;
}

void GlobalMapper::optimizationLoop()
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

        if (numDetectedPoints == 0)
        {
            printf("Error: no features detected! keyframe not accepted.\n");
            return;
        }

        std::vector<bool> matchesFound(numDetectedPoints);
        std::fill(matchesFound.begin(), matchesFound.end(), false);
        frame->mapPoints.resize(numDetectedPoints);

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
                auto pt = refKF->mapPoints[match.queryIdx];
                auto framePt3d = frame->mapPoints[match.trainIdx];
                auto framePt = frame->cvKeyPoints[match.trainIdx];
                // auto obs = Vec2d(framePt.pt.x, framePt.pt.y);
                auto z = zVector[match.trainIdx];

                if (!pt || pt == framePt3d)
                    continue;

                if (!framePt3d)
                {
                    frame->mapPoints[match.trainIdx] = pt;
                    pt->numObservations++;
                    pt->observations.insert(std::make_pair(frame, Vec3d(framePt.pt.x, framePt.pt.y, z)));
                }
                else
                {
                    // if (matcher->computeMatchingScore(pt->descriptor, framePt3d->descriptor) < 32)
                    // {
                    pt->observations.insert(framePt3d->observations.begin(), framePt3d->observations.end());
                    frame->mapPoints[match.trainIdx] = pt;
                    // }
                }

                // if (framePt3d && framePt3d->isImmature && framePt3d->getNumObservations() > 1)
                // {
                //     framePt3d->position = framePt3d->hostKF->getPoseInGlobalMap() * framePt3d->relativePos;
                //     framePt3d->isImmature = false;
                // }
            }
        }

        auto framePose = frame->getPoseInGlobalMap();
        for (int i = 0; i < numDetectedPoints; ++i)
        {
            // if (matchesFound[i])
            //     continue;
            if (frame->mapPoints[i] != NULL)
                continue;

            const auto &kp = frame->cvKeyPoints[i];
            const auto &desc = frame->pointDesc.row(i);
            const auto &z = zVector[i];

            if (z > FLT_EPSILON)
            {
                auto pt3d = std::make_shared<MapPoint>();

                pt3d->hostKF = frame;
                // pt3d->position = framePose * (Kinv * Vec3d(kp.pt.x, kp.pt.y, 1.0) * z);
                pt3d->position = framePose * (Kinv * Vec3d(kp.pt.x, kp.pt.y, 1.0) * z);
                pt3d->relativePos = Kinv * Vec3d(kp.pt.x, kp.pt.y, 1.0) * z;
                pt3d->descriptor = desc;
                pt3d->isImmature = false;
                pt3d->observations.insert(std::make_pair(frame, Vec3d(kp.pt.x, kp.pt.y, z)));
                frame->mapPoints[i] = pt3d;
            }
        }

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

void GlobalMapper::findPointCorrespondences(std::shared_ptr<Frame> kf, std::vector<std::shared_ptr<MapPoint>> mapPoints)
{
}

std::vector<std::shared_ptr<Frame>> GlobalMapper::findCloseLoopCandidate(std::shared_ptr<Frame> frame)
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
        if (std::abs(frame->getKeyframeId() - kf->getKeyframeId()) < 50)
            continue;

        auto kfPosition = kf->getPoseInGlobalMap().translation();
        auto dist = (kfPosition - framePosition).norm();

        if (dist < distTh)
            candidates.push_back(kf);
    }

    return candidates;
}

void GlobalMapper::globalConsistencyLoop()
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

void GlobalMapper::addToOptimizer(std::shared_ptr<Frame> kf)
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
    //             solver->addWorldPoint(pt->id, pt->position, true);
    //             pt->inOptimizer = true;
    //         }

    //         solver->addObservation(pt->id, kf->getKeyframeId(), Vec2d(kp.pt.x, kp.pt.y));
    //     }
    // }
}

void GlobalMapper::windowedOptimization(const int maxIteration)
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
        if (!pt || pt->observations.size() == 0 || pt->invalidated || pt->isImmature)
            continue;

        for (auto obs : pt->observations)
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
        if (!pt || pt->observations.size() == 0 || pt->invalidated || pt->isImmature)
            continue;

        for (auto obs : pt->observations)
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

        problem.SetParameterBlockConstant(pt->getParameterBlock());
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
    //         //             pt->position = rval.cast<double>();
    //         //     }
    //         // }

    //         auto poseOpt = solver->getCamPoseOptimized(kf->getKeyframeId()).inverse();
    //         kf->setOptimizationResult(poseOpt);
    //     }

    //     std::unique_lock<std::mutex> lock3(loopBufferMutex);
    //     loopKeyFrameBuffer.push(keyframeOptWin.back());
    // }
}

bool GlobalMapper::hasUnfinishedWork()
{
    std::unique_lock<std::mutex> lock(bufferMutex);
    return isOptimizing && newKeyFrameBuffer.size() > 0;
}

std::vector<SE3> GlobalMapper::getKeyFrameHistory()
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