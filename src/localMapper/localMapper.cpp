#include "localMapper/localMapper.h"
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
      lastKeyFrame(NULL)
{
}

void LocalMapper::loop()
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

                auto nMatches = matchLocalMapPoints();
                // printf("A total of %i points matched.\n", nMatches);

                optimizeKeyFramePose();

                createNewMapPoints();
            }
            else
            {
                createInitMapPoints();
            }

            lastKeyFrame = currKeyFrame;
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
            mp->observations[currKeyFrame] = Vec2d(x, y);

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
        const double x = fx * pos(0) / pos(2) + cx;
        const double y = fy * pos(1) / pos(2) + cy;
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

            currKeyFrame->mapPoints[bestIdx] = mp;
            auto &kp = currKeyFrame->keyPoints[bestIdx];
            mp->observations[currKeyFrame] = Vec2d(kp.pt.x, kp.pt.y);
            totalMatches++;
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

    cv::Ptr<cv::ORB> detector = cv::ORB::create();
    detector->detect(currKeyFrame->imRGB, keyPoints);
    detector->compute(currKeyFrame->imRGB, keyPoints, descriptors);

    currKeyFrame->keyPoints = keyPoints;
    currKeyFrame->descriptors = descriptors;
    currKeyFrame->mapPoints.resize(keyPoints.size());

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
            mp->observations[currKeyFrame] = Vec2d(x, y);

            currKeyFrame->mapPoints[i] = mp;
            map->addMapPoint(mp);

            numPointCreated++;
        }
    }
}

void LocalMapper::optimizeKeyFramePose()
{
    ceres::Problem problem;
    problem.AddParameterBlock(currKeyFrame->RT.data(), SE3::num_parameters, new LocalParameterizationSE3);
    double KBlock[4] = {K(0, 0), K(1, 1), K(0, 2), K(1, 2)};
    problem.AddParameterBlock(&KBlock[0], 4);
    problem.SetParameterBlockConstant(&KBlock[0]);

    for (auto KF : localKeyFrameSet)
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
            problem.AddResidualBlock(
                ReprojectionErrorFunctor::create(obs.second(0), obs.second(1)),
                NULL,
                &KBlock[0],
                obs.first->RT.data(),
                mp->pos.data());

            numResidualBlocks++;
        }

        problem.SetParameterBlockConstant(mp->pos.data());
    }

    if (numResidualBlocks == 0)
        return;

    ceres::Solver::Options options;
    ceres::Solver::Summary summary;
    options.linear_solver_type = ceres::DENSE_SCHUR;
    Solve(options, &problem, &summary);
    std::cout << summary.BriefReport() << std::endl;

    currKeyFrame->setPose(currKeyFrame->RT);

    if (viewer)
        viewer->addOptimizedKFPose(currKeyFrame->RT);
}
