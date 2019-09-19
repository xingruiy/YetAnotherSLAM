#include "featureMatcher.h"

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

void FeatureMatcher::matchByProjection(
    std::shared_ptr<Frame> reference,
    std::shared_ptr<Frame> current,
    SE3 &Transform,
    std::vector<cv::DMatch> &matches,
    std::vector<bool> *mask)
{
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