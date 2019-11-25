#include "LocalMapper/localMapper.h"
#include "utils/costFunctors.h"
#include "utils/mapCUDA.h"
#include <numeric>
#include <iostream>
#include <algorithm>

// Bit set count operation from
// http://graphics.stanford.edu/~seander/bithacks.html#CountBitsSetParallel
int descriptorDistance(const cv::Mat &a, const cv::Mat &b)
{
    const int *pa = a.ptr<int32_t>();
    const int *pb = b.ptr<int32_t>();

    int dist = 0;

    for (int i = 0; i < 8; i++, pa++, pb++)
    {
        unsigned int v = *pa ^ *pb;
        v = v - ((v >> 1) & 0x55555555);
        v = (v & 0x33333333) + ((v >> 2) & 0x33333333);
        dist += (((v + (v >> 4)) & 0xF0F0F0F) * 0x1010101) >> 24;
    }

    return dist;
}

LocalMapper::LocalMapper(const Mat33d &K)
    : K(K),
      shouldQuit(false),
      currKeyFrame(NULL),
      lastKeyFrame(NULL),
      updateLocalMap(true)
{
}

void LocalMapper::run()
{
    while (!shouldQuit)
    {
        if (hasNewKeyFrame())
        {
            processNewKeyFrame();

            if (lastKeyFrame)
            {
                updateLocalKeyFrames();

                updateLocalMapPoints();

                int nMatches = matchLocalMapPoints();

                optimizeKeyFramePose();

                if (updateLocalMap)
                    createNewMapPoints();
            }
            else
            {
                createInitMapPoints();
            }

            lastKeyFrame = currKeyFrame;

            if (updateLocalMap)
                loopCloser->testKeyFrame(currKeyFrame);
        }
        else
        {
            usleep(1000);
        }
    }
}

void LocalMapper::addKeyFrame(std::shared_ptr<KeyFrame> KF)
{
    std::unique_lock<std::mutex> lock(mutexKeyFrameQueue);
    keyFrameQueue.push_back(KF);
}

void LocalMapper::createInitMapPoints()
{
    size_t numPointsCreated = 0;
    auto N = currKeyFrame->keyPoints.size();

    for (int i = 0; i < N; ++i)
    {
        const auto &kp = currKeyFrame->keyPoints[i];
        const double x = kp.pt.x;
        const double y = kp.pt.y;
        const double z = currKeyFrame->getDepth(x, y);

        if (z > 0)
        {
            std::shared_ptr<MapPoint> mp(new MapPoint());
            mp->hostKF = currKeyFrame;
            mp->localReferenceId = 0;
            mp->descriptor = currKeyFrame->descriptors.row(i);
            mp->pos = currKeyFrame->RT * (K.inverse() * Vec3d(x, y, 1.0) * z);
            mp->observations[currKeyFrame] = i;

            currKeyFrame->mapPoints[i] = mp;
            map->addMapPoint(mp);

            numPointsCreated++;
        }
    }

    printf("Map initialized with %lu map points.\n", numPointsCreated);
}

void LocalMapper::updateLocalKeyFrames()
{
    std::map<std::shared_ptr<KeyFrame>, int> keyFrameCounter;
    for (auto &mp : lastKeyFrame->mapPoints)
    {
        if (mp != NULL)
        {
            if (!mp->setToRemove)
            {
                for (const auto &obs : mp->observations)
                    keyFrameCounter[obs.first]++;
            }
            else
            {
                mp = NULL;
            }
        }
    }

    if (keyFrameCounter.empty())
        return;

    localKeyFrameSet.clear();
    localKeyFrameSet.reserve(keyFrameCounter.size() * 3);

    int max = 0;
    std::shared_ptr<KeyFrame> KFMax = NULL;

    for (const auto &counter : keyFrameCounter)
    {
        auto &KF = counter.first;

        if (counter.second > max)
        {
            max = counter.second;
            KFMax = KF;
        }

        localKeyFrameSet.push_back(KF);
        KF->localReferenceId = currKeyFrame->KFId;
    }
}

void LocalMapper::updateLocalMapPoints()
{
    localMapPointSet.clear();
    for (const auto &KF : localKeyFrameSet)
    {
        for (auto &mp : KF->mapPoints)
        {
            if (!mp || mp->localReferenceId == currKeyFrame->KFId)
                continue;

            if (!mp->setToRemove)
            {
                localMapPointSet.push_back(mp);
                mp->localReferenceId = currKeyFrame->KFId;
            }
            else
                mp = NULL;
        }
    }
}

int LocalMapper::matchLocalMapPoints()
{
    int totalMatches = 0;
    const auto &RTinv = currKeyFrame->RTinv;

    const double fx = K(0, 0);
    const double fy = K(1, 1);
    const double cx = K(0, 2);
    const double cy = K(1, 2);

    for (int i = 0; i < localMapPointSet.size(); ++i)
    {
        auto &mp = localMapPointSet[i];
        auto pos = RTinv * mp->pos;
        const double z = pos(2);
        const double x = fx * pos(0) / z + cx;
        const double y = fy * pos(1) / z + cy;
        if (x >= 0 && y >= 0 && x < 640 && y < 480)
        {
            int bestDist = 256;
            int bestLevel = -1;
            int bestDist2 = 256;
            int bestLevel2 = -1;
            int bestIdx = -1;

            const auto indices = currKeyFrame->getKeyPointsInArea(x, y, 5);

            for (auto idx : indices)
            {
                if (currKeyFrame->mapPoints[idx])
                    continue;

                const int dist = descriptorDistance(mp->descriptor, currKeyFrame->descriptors.row(idx));

                if (dist < bestDist)
                {
                    bestDist2 = bestDist;
                    bestDist = dist;
                    bestLevel2 = bestLevel;
                    bestLevel = currKeyFrame->keyPoints[idx].octave;
                    bestIdx = idx;
                }
                else if (dist < bestDist2)
                {
                    bestLevel2 = currKeyFrame->keyPoints[idx].octave;
                    bestDist2 = dist;
                }
            }

            if (bestLevel == bestLevel2 && bestDist > 0.8 * bestDist2)
                continue;

            auto ptZ = currKeyFrame->getDepth(x, y);
            if (abs(z - ptZ) > 0.2)
                continue;

            // Choose the best descriptor based on a median filter
            if (!mp->setToRemove)
            {
                currKeyFrame->mapPoints[bestIdx] = mp;
                auto &kp = currKeyFrame->keyPoints[bestIdx];
                mp->observations[currKeyFrame] = bestIdx;

                std::vector<Mat> descriptors;
                for (auto obs : mp->observations)
                {
                    const auto &KF = obs.first;
                    const auto &idx = obs.second;

                    descriptors.push_back(KF->descriptors.row(idx));
                }

                if (descriptors.empty())
                    continue;

                const auto N = descriptors.size();
                float distance[N][N];

                for (int i = 0; i < N; ++i)
                {
                    distance[i][i] = 0;
                    for (int j = i + 1; j < N; ++j)
                    {
                        int distij = descriptorDistance(descriptors[i], descriptors[j]);
                        distance[i][j] = distij;
                        distance[j][i] = distij;
                    }
                }

                int bestMedian = INT_MAX;
                int bestIdx = 0;
                for (int i = 0; i < N; ++i)
                {
                    std::vector<int> dists(distance[i], distance[i] + N);
                    std::sort(dists.begin(), dists.end());
                    int median = dists[0.5 * (N - 1)];

                    if (median < bestMedian)
                    {
                        bestMedian = median;
                        bestIdx = i;
                    }
                }

                mp->descriptor = descriptors[bestIdx].clone();
                // mp->pos = (mp->pos * mp->referenceCounter + currKeyFrame->RT * (K.inverse() * Vec3d(kp.pt.x, kp.pt.y, 1.0) * ptZ)) / (mp->referenceCounter + 1);
                // mp->referenceCounter++;
                totalMatches++;
            }
        }
    }

    return totalMatches;
}

void LocalMapper::processNewKeyFrame()
{
    {
        std::unique_lock<std::mutex> lock(mutexKeyFrameQueue);
        currKeyFrame = keyFrameQueue.front();
        keyFrameQueue.pop_front();
    }

    cv::Mat descriptors;
    std::vector<cv::KeyPoint> keyPoints;

    // ORB_SLAM2::ORBextractor detector(500, 1.2, 8, 20, 7);
    // detector(currKeyFrame->imRGB, Mat(), keyPoints, descriptors);

    cv::Ptr<cv::ORB> detector = cv::ORB::create();
    detector->detect(currKeyFrame->imRGB, keyPoints);
    detector->compute(currKeyFrame->imRGB, keyPoints, descriptors);

    currKeyFrame->keyPoints = keyPoints;
    currKeyFrame->descriptors = descriptors;
    currKeyFrame->mapPoints.resize(keyPoints.size());

    if (updateLocalMap)
        map->addKeyFrame(currKeyFrame);
}

void LocalMapper::createNewMapPoints()
{
    int numPointCreated = 0;

    for (int i = 0; i < currKeyFrame->keyPoints.size(); ++i)
    {
        auto &mp = currKeyFrame->mapPoints[i];
        if (mp && !mp->setToRemove)
            continue;

        const auto &kp = currKeyFrame->keyPoints[i];
        const double x = kp.pt.x;
        const double y = kp.pt.y;
        const double z = currKeyFrame->getDepth(x, y);

        if (z > 0)
        {
            std::shared_ptr<MapPoint> mp(new MapPoint());
            mp->hostKF = currKeyFrame;
            mp->localReferenceId = 0;
            mp->descriptor = currKeyFrame->descriptors.row(i);
            mp->pos = currKeyFrame->RT * (K.inverse() * Vec3d(x, y, 1.0) * z);
            mp->observations[currKeyFrame] = i;

            currKeyFrame->mapPoints[i] = mp;
            map->addMapPoint(mp);

            numPointCreated++;
        }
    }
}

void LocalMapper::optimizeKeyFramePose()
{
    SE3 RTbo = currKeyFrame->RT;
    auto robustLoss = new ceres::HuberLoss(10);

    ceres::Problem problem;
    // problem.AddParameterBlock(currKeyFrame->RT.data(), SE3::num_parameters, new LocalParameterizationSE3);
    double KBlock[4] = {K(0, 0), K(1, 1), K(0, 2), K(1, 2)};
    problem.AddParameterBlock(&KBlock[0], 4);
    problem.SetParameterBlockConstant(&KBlock[0]);

    // Collect Key Frames that are fixed in the optimizer
    std::set<std::shared_ptr<KeyFrame>> lFixedKeyFrames;
    std::set<std::shared_ptr<KeyFrame>> lLocalKeyFrames;
    lLocalKeyFrames.insert(currKeyFrame);
    for (const auto &mp : currKeyFrame->mapPoints)
    {
        if (mp && !mp->setToRemove)
            for (const auto &obs : mp->observations)
            {
                auto KF = obs.first;
                if (KF->KFId != currKeyFrame->KFId)
                    lLocalKeyFrames.insert(obs.first);
            }
    }

    std::set<std::shared_ptr<MapPoint>> fixedMapPointSet;
    for (const auto &KF : lLocalKeyFrames)
    {
        for (const auto &mp : KF->mapPoints)
            fixedMapPointSet.insert(mp);
    }

    for (const auto &mp : fixedMapPointSet)
    {
        if (mp && !mp->setToRemove)
            for (const auto &obs : mp->observations)
            {
                const auto &KF = obs.first;
                if (KF->KFId != currKeyFrame->KFId && lLocalKeyFrames.count(KF) == 0)
                    lFixedKeyFrames.insert(KF);
            }
    }

    for (auto KF : lLocalKeyFrames)
    {
        problem.AddParameterBlock(KF->RT.data(), SE3::num_parameters, new LocalParameterizationSE3);
        // problem.SetParameterBlockConstant(KF->RT.data());
    }

    for (auto KF : lFixedKeyFrames)
    {
        problem.AddParameterBlock(KF->RT.data(), SE3::num_parameters, new LocalParameterizationSE3);
        problem.SetParameterBlockConstant(KF->RT.data());
    }

    size_t numResidualBlocks = 0;
    for (auto mp : currKeyFrame->mapPoints)
    {
        if (!mp)
            continue;

        for (const auto &obs : mp->observations)
        {
            auto KF = obs.first;
            auto idx = obs.second;
            auto &kp = KF->keyPoints[idx];
            auto z = KF->getDepth(kp.pt.x, kp.pt.y);

            problem.AddResidualBlock(
                ReprojectionErrorFunctor::create(kp.pt.x, kp.pt.y),
                // ReprojectionError3DFunctor::create(Vec3d(kp.pt.x, kp.pt.y, z)),
                // robustLoss,
                NULL,
                &KBlock[0],
                obs.first->RT.data(),
                mp->pos.data());

            numResidualBlocks++;
        }

        // if (mp->observations.size() <= 2)
        problem.SetParameterBlockConstant(mp->pos.data());
    }

    if (numResidualBlocks == 0)
        return;

    ceres::Solver::Options options;
    ceres::Solver::Summary summary;
    options.linear_solver_type = ceres::SPARSE_SCHUR;
    // options.function_tolerance = 0.001;
    // options.minimizer_progress_to_stdout = true;
    // options.function_tolerance = 1e-9;
    Solve(options, &problem, &summary);
    std::cout << summary.BriefReport() << std::endl;

    for (auto KF : lLocalKeyFrames)
        KF->setPose(KF->RT);

    if (viewer)
    {
        if (updateLocalMap)
            viewer->addOptimizedKFPose(currKeyFrame->RT);
        viewer->setRTLocalToGlobal(currKeyFrame->RT * RTbo.inverse());
    }
}

int LocalMapper::checkMapPointOutliers()
{
    auto &matchedPoints = currKeyFrame->mapPoints;
    auto &projections = currKeyFrame->keyPoints;
    const double thInPixel = 3;
    const auto fx = K(0, 0);
    const auto fy = K(1, 1);
    const auto cx = K(0, 2);
    const auto cy = K(1, 2);

    int nOutliers = 0;
    const auto nPoints = matchedPoints.size();
    std::vector<bool> outliers(nPoints);
    std::fill(outliers.begin(), outliers.end(), false);
    for (int i = 0; i < nPoints; ++i)
    {
        auto &mp = matchedPoints[i];

        if (!mp || mp->setToRemove)
        {
            mp = NULL;
            continue;
        }

        const auto pos = currKeyFrame->RTinv * mp->pos;
        const double x = fx * pos(0) / pos(2) + cx;
        const double y = fy * pos(1) / pos(2) + cy;

        if (x < 0 || y < 0 || x >= 640 || y >= 480)
        {
            mp = NULL;
            outliers[i] = true;
            nOutliers++;
            continue;
        }

        const auto &kp = projections[i];
        if ((Vec2d(x, y) - Vec2d(kp.pt.x, kp.pt.y)).norm() > thInPixel)
        {
            mp = NULL;
            outliers[i] = true;
            nOutliers++;
        }
    }

    return nOutliers;
}
