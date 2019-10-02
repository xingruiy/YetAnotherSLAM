#include "localizer/localizer.h"

void Localizer::AbsoluteOrientation(
    std::vector<Vec3d> &ref,
    std::vector<Vec3d> &src,
    SE3 &finalEstimate,
    size_t &numInliers,
    const int maxIterations)
{
    using std::chrono::duration_cast;
    using std::chrono::microseconds;
    using std::chrono::system_clock;

    size_t nPointPairs = ref.size();
    if (nPointPairs < 3)
        return;

    std::vector<bool> outliers(nPointPairs);

    Mat33d bestRot = Mat33d::Identity();
    Vec3d bestTrans = Vec3d::Zero();
    size_t bestNumInliers = 0;

    auto now = system_clock::now();
    int seed = duration_cast<microseconds>(now.time_since_epoch()).count();
    srand(seed);

    size_t iter = 0;
    const float inlierTh = 0.05;
    size_t nBadSamples = 0;
    const size_t ransacMaxIterations = 100;

    while (iter < ransacMaxIterations)
    {
        iter++;

        bool badSample = false;
        int samples[3] = {0, 0, 0};
        for (int i = 0; i < 3; ++i)
            samples[i] = rand() % nPointPairs;

        if (samples[0] == samples[1] ||
            samples[1] == samples[2] ||
            samples[2] == samples[0])
            badSample = true;

        Vec3d srcA(src[samples[0]]);
        Vec3d srcB(src[samples[1]]);
        Vec3d srcC(src[samples[2]]);

        Vec3d refA(ref[samples[0]]);
        Vec3d refB(ref[samples[1]]);
        Vec3d refC(ref[samples[2]]);

        float srcD = (srcB - srcA).cross(srcA - srcC).norm();
        float refD = (refB - refA).cross(refA - refC).norm();

        if (badSample || srcD < FLT_EPSILON || refD < FLT_EPSILON)
        {
            nBadSamples++;
            continue;
        }

        Vec3d srcMean = (srcA + srcB + srcC) / 3;
        Vec3d refMean = (refA + refB + refC) / 3;

        srcA -= srcMean;
        srcB -= srcMean;
        srcC -= srcMean;

        refA -= refMean;
        refB -= refMean;
        refC -= refMean;

        Mat33d Ab = Mat33d::Zero();
        Ab += srcA * refA.transpose();
        Ab += srcB * refB.transpose();
        Ab += srcC * refC.transpose();

        Eigen::JacobiSVD<Mat33d> svd;
        svd.compute(Ab, Eigen::ComputeFullU | Eigen::ComputeFullV);
        Mat33d V = svd.matrixV();
        Mat33d U = svd.matrixU();
        Mat33d R = (V * U.transpose()).transpose();
        if (R.determinant() < 0)
            continue;

        Vec3d t = srcMean - R * refMean;

        int nInliers = 0;
        std::fill(outliers.begin(), outliers.end(), true);
        for (int i = 0; i < src.size(); ++i)
        {
            double dist = (src[i] - (R * ref[i] + t)).norm();
            if (dist <= inlierTh)
            {
                nInliers++;
                outliers[i] = false;
            }
        }

        if (nInliers > bestNumInliers)
        {

            Ab = Mat33d::Zero();
            srcMean = Vec3d::Zero();
            refMean = Vec3d::Zero();
            for (int i = 0; i < outliers.size(); ++i)
            {
                if (!outliers[i])
                {
                    srcMean += src[i];
                    refMean += ref[i];
                }
            }

            srcMean /= nInliers;
            refMean /= nInliers;

            for (int i = 0; i < outliers.size(); ++i)
                if (!outliers[i])
                    Ab += src[i] * ref[i].transpose();

            Ab -= nInliers * srcMean * refMean.transpose();

            svd.compute(Ab, Eigen::ComputeFullU | Eigen::ComputeFullV);
            V = svd.matrixV();
            U = svd.matrixU();
            bestRot = (V * U.transpose()).transpose();
            bestTrans = srcMean - bestRot * refMean;
            bestNumInliers = nInliers;
        }
    }

    numInliers = bestNumInliers;
}

SE3 Localizer::getWorldTransform(
    std::shared_ptr<Frame> frame,
    std::vector<cv::DMatch> &matches,
    std::vector<std::shared_ptr<MapPoint>> &pts)
{
    std::vector<Vec3d> referencePts;
    std::vector<Vec3d> currentPts;
    for (auto match : matches)
    {
        auto &refPt = pts[match.trainIdx];
        auto &currPt = frame->mapPoints[match.queryIdx];
        if (refPt && !refPt->isBad() && currPt && !currPt->isBad())
        {
            referencePts.push_back(refPt->getPosWorld());
            currentPts.push_back(currPt->getPosWorld());
        }
    }
}