#include "localMapper/localMapper.h"
#include "utils/costFunctors.h"
#include "utils/mapCUDA.h"
#include <numeric>
#include <iostream>
#include <algorithm>

template <class T>
std::vector<size_t> sortIndex(const std::vector<T> &v)
{
    std::vector<size_t> idx(v.size());
    std::iota(idx.begin(), idx.end(), 0);
    std::sort(idx.begin(), idx.end(), [&v](size_t i1, size_t i2) { return v[i1] < v[i2]; });

    return idx;
}

FeatureMapper::FeatureMapper(
    Mat33d &K,
    std::shared_ptr<Map> map)
    : K(K),
      shouldQuit(false),
      map(map)
{
    matcher = std::make_shared<FeatureMatcher>(PointType::ORB, DescType::ORB);
}

void FeatureMapper::loop()
{
    while (!shouldQuit)
    {
        if (auto frameUnProc = getNewKeyframe())
        {
            matchFeatures(frameUnProc);
            createNewPoints(frameUnProc);
            map->addKeyFrame(frameUnProc);
            map->setCurrentKeyframe(frameUnProc);
        }
    }
}

std::shared_ptr<Frame> FeatureMapper::getNewKeyframe()
{
    auto frameUnProc = map->getUnprocessedKeyframe();

    if (!frameUnProc)
        return NULL;

    // update kf pose based on reference kf in case its pose changed
    auto refKF = frameUnProc->getReferenceKF();
    if (refKF)
    {
        auto dT = frameUnProc->getTrackingResult();
        auto refT = refKF->getPoseInGlobalMap();
        auto T = refT * dT;
        frameUnProc->setOptimizationResult(T);
    }

    return frameUnProc;
}

void FeatureMapper::matchFeatures(std::shared_ptr<Frame> kf)
{
    if (kf->getNumPointsDetected() == 0)
        kf->detectKeyPoints(matcher);

    auto numPointsDetected = kf->getNumPointsDetected();
    if (numPointsDetected == 0)
    {
        printf("Error: no features detected! THe new keyframe not accepted.\n");
        return;
    }

    auto lastKF = map->getCurrentKeyframe();
    if (lastKF)
    {
        std::vector<cv::DMatch> matches;
        matcher->matchByProjection2NN(lastKF, kf, K, matches, NULL);

        for (auto m : matches)
        {
            auto &pt = lastKF->mapPoints[m.queryIdx];
            auto &framePt = kf->mapPoints[m.trainIdx];
            auto &kp = kf->cvKeyPoints[m.trainIdx];
            auto &z = kf->keyPointDepth[m.trainIdx];

            if (!framePt)
            {
                framePt = pt;
                pt->addObservation(kf, Vec3d(kp.pt.x, kp.pt.y, z));
            }
        }

        // Mat outImg;
        // cv::drawMatches(
        //     lastKF->getImage(),
        //     lastKF->cvKeyPoints,
        //     kf->getImage(),
        //     kf->cvKeyPoints,
        //     matches,
        //     outImg);
        // cv::imshow("img", outImg);
        // cv::waitKey(0);
    }
}

void FeatureMapper::createNewPoints(std::shared_ptr<Frame> kf)
{
    auto numKeyPoints = kf->getNumPointsDetected();
    for (int i = 0; i < numKeyPoints; i += 1)
    {
        auto &framePt = kf->mapPoints[i];
        if (framePt)
            continue;

        const auto &z = kf->keyPointDepth[i];
        const auto &n = kf->keyPointNorm[i];
        if (z > FLT_EPSILON && n(2) > FLT_EPSILON)
        {
            const auto &kp = kf->cvKeyPoints[i];
            Vec3d pos = kf->getPoseInGlobalMap() * (K.inverse() * Vec3d(kp.pt.x, kp.pt.y, 1.0) * z);
            Vec3f normal = (kf->getPoseInGlobalMap().rotationMatrix() * n.cast<double>()).cast<float>();

            auto pt = std::make_shared<MapPoint>();
            pt->setHost(kf);
            pt->setPosWorld(pos);
            pt->setNormal(normal);
            pt->setDescriptor(kf->descriptors.row(i));
            pt->addObservation(kf, Vec3d(kp.pt.x, kp.pt.y, z));
            framePt = pt;
            map->addMapPoint(pt);
            std::cout << i << std::endl;
        }
    }
}

void FeatureMapper::detectLoop(std::shared_ptr<Frame> kf)
{
    Mat freePtDesc;
    std::vector<std::shared_ptr<MapPoint>> freePts;

    auto numKeyPoints = kf->getNumPointsDetected();
    for (int i = 0; i < numKeyPoints; i += 1)
    {
        const auto &framePt = kf->mapPoints[i];
        if (framePt)
            continue;

        const auto &z = kf->keyPointDepth[i];
        if (z > FLT_EPSILON)
        {
            const auto &kp = kf->cvKeyPoints[i];
            Vec3d pos = K.inverse() * Vec3d(kp.pt.x, kp.pt.y, 1.0) * z;

            auto pt = std::make_shared<MapPoint>();
            pt->setHost(kf);
            pt->setPosWorld(pos);
            pt->setDescriptor(kf->descriptors.row(i));
            pt->addObservation(kf, Vec3d(kp.pt.x, kp.pt.y, z));
            kf->mapPoints[i] = pt;

            freePtDesc.push_back(kf->descriptors.row(i));
            freePts.push_back(pt);
        }
    }

    std::cout << "searching correspondence in the map." << std::endl;
    Mat mapdescriptors = map->getdescriptorsriptorsAll();

    if (mapdescriptors.rows != 0)
    {
        cv::Ptr<cv::DescriptorMatcher> matcher2 = cv::DescriptorMatcher::create(cv::DescriptorMatcher::BRUTEFORCE_HAMMING);
        std::vector<std::vector<cv::DMatch>> rawMatches;
        std::vector<cv::DMatch> matches;
        matcher2->knnMatch(freePtDesc, mapdescriptors, rawMatches, 2);

        for (auto mlist : rawMatches)
        {
            if (mlist[0].distance / mlist[1].distance < 0.8)
                matches.push_back(mlist[0]);
        }

        std::vector<Vec3f> pts;
        auto &mapPointsAll = map->getMapPointsAll();
        for (auto &m : matches)
        {
            auto &pt = mapPointsAll[m.trainIdx];
            auto &framePt = kf->mapPoints[m.queryIdx];
            auto &kp = kf->cvKeyPoints[m.queryIdx];
            auto &z = kf->keyPointDepth[m.queryIdx];
            framePt = pt;
            pt->addObservation(kf, Vec3d(kp.pt.x, kp.pt.y, z));
            pts.push_back(pt->getPosWorld().cast<float>());
        }
    }
}

void FeatureMapper::setViewer(MapViewer *viewer)
{
    this->viewer = viewer;
}

void FeatureMapper::setMap(std::shared_ptr<Map> map)
{
    this->map = map;
}

void FeatureMapper::setShouldQuit()
{
    shouldQuit = true;
}

void FeatureMapper::optimize(std::shared_ptr<Frame> kf)
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
        if (pt && !pt->isBad())
        {
            std::unique_lock<std::mutex> lock(pt->lock);
            for (auto obs : pt->getObservations())
            {
                if (obs.first != kf && fixedKFs.find(obs.first) == fixedKFs.end())
                    fixedKFs.insert(obs.first);
            }
        }
    }

    if (fixedKFs.size() == 0)
        return;

    // std::cout << fixedKFs.size() << std::endl;

    for (auto frame : fixedKFs)
    {
        problem.AddParameterBlock(frame->getParameterBlock(), SE3::num_parameters, new LocalParameterizationSE3);
        problem.SetParameterBlockConstant(frame->getParameterBlock());
    }

    size_t numResidualBlocks = 0;
    for (auto pt : kf->mapPoints)
    {
        if (pt && !pt->isBad())
        {
            std::unique_lock<std::mutex> lock(pt->lock);
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