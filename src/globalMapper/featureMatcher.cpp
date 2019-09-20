#include "featureMatcher.h"

#define MatchWindowDist 5
#define MatchMinScore 5

inline float interpolateBiLinear(Mat map, const float &x, const float &y)
{
    int u = static_cast<int>(std::floor(x));
    int v = static_cast<int>(std::floor(y));
    float cox = x - u;
    float coy = y - v;
    return (map.ptr<float>(v)[u] * (1 - cox) + map.ptr<float>(v)[u + 1] * cox) * (1 - coy) +
           (map.ptr<float>(v + 1)[u] * (1 - cox) + map.ptr<float>(v + 1)[u + 1] * cox) * coy;
}

FeatureMatcher::FeatureMatcher(PointType pType)
    : pointType(pType)
{
    switch (pType)
    {
    case PointType::ORB:
        orbDetector = cv::ORB::create(1000);
        break;
    case PointType::FAST:
        fastDetector = cv::FastFeatureDetector::create();
        break;
    }
}

void FeatureMatcher::detect(
    Mat image, Mat depth, Mat intensity,
    std::vector<cv::KeyPoint> &keyPoints,
    std::vector<Vec9f> &patch3x3,
    std::vector<float> &depthVec)
{
    switch (pointType)
    {
    case ORB:
        orbDetector->detect(image, keyPoints);
        break;

    case FAST:
        fastDetector->detect(image, keyPoints);
        break;
    }

    computePatch3x3(intensity, keyPoints, patch3x3);
    extractDepth(depth, keyPoints, depthVec);
}

template <typename Derived>
EIGEN_STRONG_INLINE float computePatchScoreL2Norm(const Eigen::MatrixBase<Derived> &a, const Eigen::MatrixBase<Derived> &b)
{
    return (a - b).norm() / 9;
}

void FeatureMatcher::matchByProjection(
    const std::shared_ptr<Frame> frame,
    std::vector<bool> &matchesFound,
    const std::shared_ptr<Frame> kf,
    const Mat33d &K,
    std::vector<cv::DMatch> &matches)
{
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
    auto framePoseInv = frame->getPose().inverse();

    for (auto iter = mapPoints.begin(), iend = mapPoints.end(); iter != iend; ++iter)
    {
        const auto pt = *iter;
        cv::DMatch m;
        m.queryIdx = iter - mapPoints.begin();
        if (pt && (pt->hostKF == kf))
        {
            Vec2d obs(-1, -1);
            int bestPointIdx = -1;
            float bestPairScore = std::numeric_limits<float>::max();

            Vec3d ptWarped = framePoseInv * pt->position;
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
                        const auto score = computePatchScoreL2Norm(pt->descriptor, descriptors[i]);

                        if (score < bestPairScore)
                        {
                            bestPointIdx = i;
                            obs = Vec2d(x, y);
                            bestPairScore = score;
                        }
                    }
                }

                if (bestPointIdx >= 0 && bestPairScore < MatchMinScore)
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
            z = depth.ptr<float>(static_cast<int>(y + 0.5f))[static_cast<int>(x + 0.5f)];
            if (std::isfinite(z) && z > FLT_EPSILON)
                z = z;
            else
                z = 0.f;
        }

        depthVec[i] = z;
    }
}