#include "featureMatcher.h"

#define MatchWindowDist 3
// #define MatchMinScore 4
#define MatchMinScore 32

inline float interpolateBiLinear(Mat map, const float &x, const float &y)
{
    int u = static_cast<int>(std::floor(x));
    int v = static_cast<int>(std::floor(y));
    float cox = x - u;
    float coy = y - v;
    return (map.ptr<float>(v)[u] * (1 - cox) + map.ptr<float>(v)[u + 1] * cox) * (1 - coy) +
           (map.ptr<float>(v + 1)[u] * (1 - cox) + map.ptr<float>(v + 1)[u + 1] * cox) * coy;
}

FeatureMatcher::FeatureMatcher(PointType pType, DescType dType)
    : pointType(pType), descType(dType)
{
    switch (pType)
    {
    case PointType::ORB:
        orbDetector = cv::ORB::create(1000);
        break;
    case PointType::FAST:
        fastDetector = cv::FastFeatureDetector::create(25);
        break;
    case PointType::BRISK:
        briskDetector = cv::BRISK::create();
        break;
    case PointType::SURF:
        surfDetector = cv::xfeatures2d::SURF::create(150);
        break;
    }

    switch (dType)
    {
    case DescType::ORB:
        orbDetector = cv::ORB::create(1000);
        minMatchingDistance = 32;
        matcher = cv::DescriptorMatcher::create(cv::DescriptorMatcher::BRUTEFORCE_HAMMING);
        break;
    case DescType::BRISK:
        briskDetector = cv::BRISK::create();
        minMatchingDistance = 32;
        matcher = cv::DescriptorMatcher::create(cv::DescriptorMatcher::BRUTEFORCE_HAMMING);
        break;
    case DescType::SURF:
        surfDetector = cv::xfeatures2d::SURF::create(150);
        minMatchingDistance = 10;
        matcher = cv::DescriptorMatcher::create(cv::DescriptorMatcher::BRUTEFORCE);
        break;
    }
}

void FeatureMatcher::detectAndCompute(
    const Mat image,
    std::vector<cv::KeyPoint> &keyPoints,
    Mat &descriptors)
{
    switch (pointType)
    {
    case PointType::ORB:
        orbDetector->detect(image, keyPoints);
        break;
    case PointType::FAST:
        fastDetector->detect(image, keyPoints);
        break;
    case PointType::BRISK:
        briskDetector->detect(image, keyPoints);
        break;
    case PointType::SURF:
        surfDetector->detect(image, keyPoints);
        break;
    }

    switch (descType)
    {
    case DescType::ORB:
        orbDetector->compute(image, keyPoints, descriptors);
        break;
    case DescType::BRISK:
        briskDetector->compute(image, keyPoints, descriptors);
        break;
    case DescType::SURF:
        surfDetector->compute(image, keyPoints, descriptors);
        break;
    }
}

float FeatureMatcher::computeMatchingScore(Mat desc, Mat refDesc)
{
    std::vector<cv::DMatch> match;
    matcher->match(desc, refDesc, match);
    return match[0].distance;
}

void FeatureMatcher::matchByProjection(
    const std::shared_ptr<Frame> kf,
    const std::shared_ptr<Frame> frame,
    const Mat33d &K,
    std::vector<cv::DMatch> &matches,
    std::vector<bool> *matchesFound)
{
    matches.clear();
    size_t numSuccessMatch = 0;

    const float fx = K(0, 0);
    const float fy = K(1, 1);
    const float cx = K(0, 2);
    const float cy = K(1, 2);

    auto &mapPoints = kf->mapPoints;
    auto &keyPoints = frame->cvKeyPoints;
    auto &descriptors = frame->descriptors;
    auto framePoseInv = frame->getPoseInGlobalMap().inverse();

    for (auto iter = mapPoints.begin(), iend = mapPoints.end(); iter != iend; ++iter)
    {
        const auto pt = *iter;
        cv::DMatch m;
        m.queryIdx = iter - mapPoints.begin();
        if (pt && (pt->getHost() == kf))
        {
            Vec2d obs(-1, -1);
            int bestPointIdx = -1;
            float bestPairScore = std::numeric_limits<float>::max();

            Vec3d ptWarped = framePoseInv * pt->getPosWorld();
            const float u = fx * ptWarped(0) / ptWarped(2) + cx;
            const float v = fy * ptWarped(1) / ptWarped(2) + cy;

            if (u > 0 && v > 0 && u < 639 && v < 479)
            {
                for (int i = 0; i < keyPoints.size(); ++i)
                {
                    if (matchesFound && (*matchesFound)[i])
                        continue;

                    const auto &x = keyPoints[i].pt.x;
                    const auto &y = keyPoints[i].pt.y;
                    float dist = (Vec2f(x, y) - Vec2f(u, v)).norm();

                    if (dist <= MatchWindowDist)
                    {
                        // const auto score = computePatchScoreL2Norm(ptgetDescriptor(), descriptors[i]);
                        const auto score = computeMatchingScore(pt->getDescriptor(), descriptors.row(i));

                        if (score > 32)
                            continue;

                        if (score < bestPairScore)
                        {
                            bestPointIdx = i;
                            obs = Vec2d(x, y);
                            bestPairScore = score;
                        }
                    }
                }

                if (bestPointIdx >= 0 && bestPairScore < minMatchingDistance)
                {
                    m.trainIdx = bestPointIdx;
                    matches.push_back(m);
                    numSuccessMatch++;
                    if (matchesFound)
                        (*matchesFound)[bestPointIdx] = true;
                }
            }
        }
    }
}

void FeatureMatcher::matchByProjection2NN(
    const std::vector<std::shared_ptr<MapPoint>> &mapPoints,
    const std::shared_ptr<Frame> frame,
    const Mat33d &K,
    std::vector<cv::DMatch> &matches,
    std::vector<bool> *matchesFound)
{
    matches.clear();
    size_t numSuccessMatch = 0;

    const float fx = K(0, 0);
    const float fy = K(1, 1);
    const float cx = K(0, 2);
    const float cy = K(1, 2);

    auto &keyPoints = frame->cvKeyPoints;
    auto &descriptors = frame->descriptors;
    auto framePoseInv = frame->getPoseInGlobalMap().inverse();

    for (auto iter = mapPoints.begin(), iend = mapPoints.end(); iter != iend; ++iter)
    {
        const auto pt = *iter;
        cv::DMatch m;
        m.queryIdx = iter - mapPoints.begin();
        if (pt)
        {
            Vec2d obs(-1, -1);
            int bestPointIdx = -1;
            float bestPairScore = std::numeric_limits<float>::max();

            Vec3d ptWarped = framePoseInv * pt->getPosWorld();
            const float u = fx * ptWarped(0) / ptWarped(2) + cx;
            const float v = fy * ptWarped(1) / ptWarped(2) + cy;

            if (u > 0 && v > 0 && u < 639 && v < 479)
            {
                for (int i = 0; i < keyPoints.size(); ++i)
                {
                    const auto &x = keyPoints[i].pt.x;
                    const auto &y = keyPoints[i].pt.y;
                    float dist = (Vec2f(x, y) - Vec2f(u, v)).norm();

                    if (dist <= MatchWindowDist)
                    {
                        // const auto score = computePatchScoreL2Norm(ptgetDescriptor(), descriptors[i]);
                        const auto score = computeMatchingScore(pt->getDescriptor(), descriptors.row(i));

                        if (score < bestPairScore)
                        {
                            bestPointIdx = i;
                            obs = Vec2d(x, y);
                            bestPairScore = score;
                        }
                    }
                }

                if (bestPointIdx >= 0 && bestPairScore < minMatchingDistance)
                {
                    m.trainIdx = bestPointIdx;
                    matches.push_back(m);
                    numSuccessMatch++;
                }
            }
        }
    }
}

void FeatureMatcher::matchByProjection2NN(
    const std::shared_ptr<Frame> kf,
    const std::shared_ptr<Frame> frame,
    const Mat33d &K,
    std::vector<cv::DMatch> &matches,
    std::vector<bool> *matchesFound)
{
    matches.clear();
    size_t numSuccessMatch = 0;

    const float fx = K(0, 0);
    const float fy = K(1, 1);
    const float cx = K(0, 2);
    const float cy = K(1, 2);

    auto &mapPoints = kf->mapPoints;
    auto &keyPoints = frame->cvKeyPoints;
    auto &descriptors = frame->descriptors;
    auto currentDepth = frame->getDepth();
    auto framePoseInv = frame->getPoseInGlobalMap().inverse();

    for (auto iter = mapPoints.begin(), iend = mapPoints.end(); iter != iend; ++iter)
    {
        const auto pt = *iter;
        cv::DMatch m;
        m.queryIdx = iter - mapPoints.begin();
        if (pt && (pt->getHost() == kf))
        {
            int bestPointIdx = -1;
            int secondBestIdx = -1;
            float bestPairScore = std::numeric_limits<float>::max();
            float secondBestPairScore = std::numeric_limits<float>::max();

            Vec3d ptWarped = framePoseInv * pt->getPosWorld();
            const float u = fx * ptWarped(0) / ptWarped(2) + cx;
            const float v = fy * ptWarped(1) / ptWarped(2) + cy;

            if (u > 0 && v > 0 && u < 639 && v < 479)
            {
                for (int i = 0; i < keyPoints.size(); ++i)
                {
                    if (matchesFound && (*matchesFound)[i])
                        continue;

                    const auto &x = keyPoints[i].pt.x;
                    const auto &y = keyPoints[i].pt.y;
                    float dist = (Vec2f(x, y) - Vec2f(u, v)).norm();
                    // auto currentZ = currentDepth.ptr<float>((int)round(y))[(int)round(x)];
                    auto currentZ = frame->keyPointDepth[i];

                    if (dist <= MatchWindowDist && abs(ptWarped(2) - currentZ) < 0.1)
                    {
                        const auto score = computeMatchingScore(pt->getDescriptor(), descriptors.row(i));

                        // if (score > minMatchingDistance)
                        //     continue;
                        // std::cout << score << std::endl;

                        if (score < bestPairScore)
                        {
                            bestPointIdx = i;
                            bestPairScore = score;
                        }
                        else if (score < secondBestPairScore)
                        {
                            secondBestIdx = i;
                            secondBestPairScore = score;
                        }
                    }
                }

                if (bestPointIdx >= 0)
                {
                    bool chooseBest = false;
                    if (secondBestIdx < 0)
                        chooseBest = true;
                    else if (bestPairScore / secondBestPairScore < 0.8)
                        chooseBest = true;

                    if (chooseBest)
                    {
                        m.trainIdx = bestPointIdx;
                        matches.push_back(m);
                        numSuccessMatch++;
                        if (matchesFound)
                            (*matchesFound)[bestPointIdx] = true;
                    }
                }
            }
        }
    }
}

void FeatureMatcher::computePointDepth(
    const Mat depth,
    const std::vector<cv::KeyPoint> &cvKeyPoint,
    std::vector<float> &pointDepth)
{
    pointDepth.resize(cvKeyPoint.size());
    for (int i = 0; i < cvKeyPoint.size(); ++i)
    {
        const auto &kp = cvKeyPoint[i];
        const float &x = kp.pt.x;
        const float &y = kp.pt.y;
        float z = 0.f;

        if (x > 0 && y > 0 && x < depth.cols && y < depth.rows)
        {
            // TODO: depth interpolation seems to cause trouble, solution: edge aware interpolation?
            z = depth.ptr<float>(static_cast<int>(round(y)))[static_cast<int>(round(x))];
            if (std::isfinite(z) && z > FLT_EPSILON)
                z = z;
            else
                z = 0.f;
        }

        pointDepth[i] = z;
    }
}

void FeatureMatcher::computePointNormal(
    const Mat normal,
    const std::vector<cv::KeyPoint> &cvKeyPoint,
    std::vector<Vec3f> &pointNormal)
{
    pointNormal.resize(cvKeyPoint.size());
    for (int i = 0; i < cvKeyPoint.size(); ++i)
    {
        const auto &kp = cvKeyPoint[i];
        const float &x = kp.pt.x;
        const float &y = kp.pt.y;
        Vec4f z = Vec4f(0, 0, 0, 0);

        if (x > 0 && y > 0 && x < normal.cols && y < normal.rows)
        {
            z = normal.ptr<Vec4f>(static_cast<int>(round(y)))[static_cast<int>(round(x))];
            if (!(std::isfinite(z(0)) && z == z))
                z = Vec4f(0, 0, 0, 0);
        }

        pointNormal[i] = z.head<3>();
    }
}