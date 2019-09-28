#include "featureMatcher.h"

#define MatchWindowDist 5
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

void FeatureMatcher::detect(
    Mat image, Mat depth,
    std::vector<cv::KeyPoint> &keyPoints,
    Mat &descriptor,
    std::vector<float> &depthVec)
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
        orbDetector->compute(image, keyPoints, descriptor);
        break;
    case DescType::BRISK:
        briskDetector->compute(image, keyPoints, descriptor);
        break;
    case DescType::SURF:
        surfDetector->compute(image, keyPoints, descriptor);
        break;
    }

    extractDepth(depth, keyPoints, depthVec);
}

template <typename Derived>
EIGEN_STRONG_INLINE float computePatchScoreL2Norm(const Eigen::MatrixBase<Derived> &a, const Eigen::MatrixBase<Derived> &b)
{
    return (a - b).norm() / 9;
}

void FeatureMatcher::compute(
    Mat image,
    std::vector<cv::KeyPoint> pt,
    Mat &desc)
{
    orbDetector->compute(image, pt, desc);
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
    auto &descriptors = frame->pointDesc;
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
    const std::vector<std::shared_ptr<Frame>> mapPoints,
    const std::shared_ptr<Frame> frame,
    const Mat33d &K,
    std::vector<cv::DMatch> &matches,
    std::vector<bool> *matchesFound)
{
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
    auto &descriptors = frame->pointDesc;
    auto currentDepth = frame->getDepth();
    auto framePoseInv = frame->getPoseInGlobalMap().inverse();

    for (auto iter = mapPoints.begin(), iend = mapPoints.end(); iter != iend; ++iter)
    {
        const auto pt = *iter;
        cv::DMatch m;
        m.queryIdx = iter - mapPoints.begin();
        if (pt && (pt->getHost() == kf))
        {
            Vec2d bestObs(-1, -1);
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
                    auto currentZ = frame->depthVec[i];

                    if (dist <= MatchWindowDist && abs(ptWarped(2) - currentZ) < 0.05)
                    {
                        // const auto score = computePatchScoreL2Norm(ptgetDescriptor(), descriptors[i]);
                        const auto score = computeMatchingScore(pt->getDescriptor(), descriptors.row(i));

                        if (score > minMatchingDistance)
                            continue;
                        // std::cout << score << std::endl;

                        if (score < bestPairScore)
                        {
                            bestPointIdx = i;
                            bestObs = Vec2d(x, y);
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

void FeatureMatcher::matchByDescriptor(
    const std::shared_ptr<Frame> kf,
    const std::shared_ptr<Frame> frame,
    const Mat33d &K,
    std::vector<cv::DMatch> &matches)
{
    std::vector<std::vector<cv::DMatch>> rawMatches;
    matcher->knnMatch(frame->pointDesc, kf->pointDesc, rawMatches, 2);

    for (auto match2NN : rawMatches)
    {
        if (match2NN[0].distance / match2NN[1].distance < 0.6)
            matches.push_back(match2NN[0]);
    }
}

void FeatureMatcher::computePatch3x3(
    Mat image,
    std::vector<cv::KeyPoint> &points,
    std::vector<Vec9f> &patches)
{
    patches.resize(points.size());
    auto ibegin = points.begin();

    for (auto iter = ibegin, iend = points.end(); iter != iend; ++iter)
    {
        float &x = iter->pt.x;
        float &y = iter->pt.y;
        Vec9f hostPatch = Vec9f::Zero();

        if (x > 1 && y > 1 && x < image.cols - 2 && y < image.rows - 2)
        {
            for (int i = 0; i < 9; ++i)
            {
                int v = i / 3 - 1;
                int u = i - v * 3 - 1;
                auto val = interpolateBiLinear(image, x + u, y + v);
                hostPatch(i) = (val == val ? val : 0);
            }
        }

        patches[iter - ibegin] = hostPatch;
    }
}

void FeatureMatcher::extractDepth(
    Mat depth,
    std::vector<cv::KeyPoint> &points,
    std::vector<float> &depthVec)
{
    depthVec.resize(points.size());
    for (int i = 0; i < points.size(); ++i)
    {
        const auto &kp = points[i];
        const float &x = kp.pt.x;
        const float &y = kp.pt.y;
        float z = 0.f;

        if (x > 1 && y > 1 && x < depth.cols - 2 && y < depth.rows - 2)
        {
            // TODO: depth interpolation seems to cause trouble, solution: edge aware interpolation?
            z = depth.ptr<float>(static_cast<int>(y + 0.5f))[static_cast<int>(x + 0.5f)];
            if (std::isfinite(z) && z > FLT_EPSILON)
                z = z;
            else
                z = 0.f;
        }

        depthVec[i] = z;
    }
}