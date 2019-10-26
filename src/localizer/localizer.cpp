#include "utils/mapCUDA.h"
#include "localizer/localizer.h"
#include "optimizer/featureMatcher.h"

#define MinimumNumMatches 3

void Localizer::absoluteOrientation(
    const std::vector<Vec3d> &srcPts,
    const std::vector<Vec3d> &dstPts,
    SE3 &estimate)
{
    Vec3d srcMean = (srcPts[0] + srcPts[1] + srcPts[2]) / 3.0;
    Vec3d dstMean = (dstPts[0] + dstPts[1] + dstPts[2]) / 3.0;
    Mat33d SVDMat = Mat33d::Zero();
    SVDMat += (srcPts[0] - srcMean) * (dstPts[0] - dstMean).transpose();
    SVDMat += (srcPts[1] - srcMean) * (dstPts[1] - dstMean).transpose();
    SVDMat += (srcPts[2] - srcMean) * (dstPts[2] - dstMean).transpose();
    auto UVMat = SVDMat.jacobiSvd(Eigen::ComputeFullU | Eigen::ComputeFullV);
    Mat33d MatV = UVMat.matrixV();
    Mat33d MatU = UVMat.matrixU();
    Mat33d Rot = Mat33d::Identity();
    if (MatU.determinant() * MatV.determinant() < 0)
    {
        Mat33d I = Mat33d::Identity();
        I(2, 2) = -1;
        Rot = MatU * I * MatV.transpose();
    }
    else
        Rot = MatU * MatV.transpose();
    Vec3d t = dstMean - Rot * srcMean;
    Mat44d Transform = Mat44d::Identity();
    Transform.topLeftCorner(3, 3) = Rot;
    Transform.topRightCorner(3, 1) = t;
    estimate = SE3(Transform);
}

void Localizer::absoluteOrientation(
    const std::vector<Vec3d> &srcPts,
    const std::vector<Vec3d> &dstPts,
    const std::vector<bool> &outliers,
    SE3 &estimate)
{
    auto numInliers = 0;
    Vec3d srcMean = Vec3d::Zero();
    Vec3d dstMean = Vec3d::Zero();
    for (int n = 0; n < outliers.size(); ++n)
    {
        if (outliers[n])
            continue;

        numInliers++;
        srcMean += srcPts[n];
        dstMean += dstPts[n];
    }

    srcMean /= numInliers;
    dstMean /= numInliers;

    Mat33d SVDMat = Mat33d::Zero();

    for (int n = 0; n < outliers.size(); ++n)
    {
        if (outliers[n])
            continue;

        SVDMat += (srcPts[n] - srcMean) * (dstPts[n] - dstMean).transpose();
    }

    auto UVMat = SVDMat.jacobiSvd(Eigen::ComputeFullU | Eigen::ComputeFullV);
    Mat33d MatV = UVMat.matrixV();
    Mat33d MatU = UVMat.matrixU();
    Mat33d Rot = Mat33d::Identity();
    if (MatU.determinant() * MatV.determinant() < 0)
    {
        Mat33d I = Mat33d::Identity();
        I(2, 2) = -1;
        Rot = MatU * I * MatV.transpose();
    }
    else
        Rot = MatU * MatV.transpose();
    Vec3d t = dstMean - Rot * srcMean;
    Mat44d Transform = Mat44d::Identity();
    Transform.topLeftCorner(3, 3) = Rot;
    Transform.topRightCorner(3, 1) = t;
    estimate = SE3(Transform);
}

int Localizer::evaluateOutlier(
    const std::vector<Vec3d> &srcPts,
    const std::vector<Vec3d> &dstPts,
    SE3 &estimate,
    std::vector<bool> &outliers)
{
    Mat33d R = estimate.rotationMatrix();
    Vec3d t = estimate.translation();

    int nInliers = 0;
    std::fill(outliers.begin(), outliers.end(), true);
    const double inlierDistTh = 0.05;

    for (int i = 0; i < srcPts.size(); ++i)
    {
        double dist = (dstPts[i] - (R * srcPts[i] + t)).norm();
        if (dist <= inlierDistTh)
        {
            nInliers++;
            outliers[i] = false;
        }
    }

    return nInliers;
}

void Localizer::runRansacAO(
    const std::vector<Vec3d> &src,
    const std::vector<Vec3d> &dst,
    SE3 &bestEsimate,
    size_t &numInliers,
    const int maxIterations)
{
    size_t nPointPairs = src.size();
    if (nPointPairs < 3)
        return;

    std::vector<bool> outliers(nPointPairs);
    size_t bestNumInliers = 0;
    size_t numIter = 0;

    while (numIter < maxIterations)
    {
        numIter++;

        int samples[3] = {0, 0, 0};
        for (int i = 0; i < 3; ++i)
        {
            samples[i] = rand() % nPointPairs;
        }

        if (samples[0] == samples[1] ||
            samples[1] == samples[2] ||
            samples[2] == samples[0])
            continue;

        Vec3d srcA(src[samples[0]]);
        Vec3d srcB(src[samples[1]]);
        Vec3d srcC(src[samples[2]]);

        Vec3d refA(dst[samples[0]]);
        Vec3d refB(dst[samples[1]]);
        Vec3d refC(dst[samples[2]]);

        float srcD = (srcB - srcA).cross(srcC - srcA).norm();
        float refD = (refB - refA).cross(refC - refA).norm();

        if (srcD < FLT_EPSILON || refD < FLT_EPSILON)
            continue;

        std::vector<Vec3d> srcPts = {src[samples[0]],
                                     src[samples[1]],
                                     src[samples[2]]};

        std::vector<Vec3d> dstPts = {dst[samples[0]],
                                     dst[samples[1]],
                                     dst[samples[2]]};

        SE3 estimate;
        absoluteOrientation(srcPts, dstPts, estimate);
        int nInliers = evaluateOutlier(src, dst, estimate, outliers);

        if (nInliers > bestNumInliers)
        {
            absoluteOrientation(src, dst, outliers, bestEsimate);
            bestNumInliers = evaluateOutlier(src, dst, estimate, outliers);
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

template <class Derived>
cv::Vec3f eigenVecToCV(const Eigen::MatrixBase<Derived> &in)
{
    cv::Vec3f out;
    out(0) = in(0);
    out(1) = in(1);
    out(2) = in(2);
    return out;
}

std::vector<cv::DMatch> Localizer::getMatches2NN(Mat src, Mat dst, bool allowAmbiguity)
{
    std::vector<std::vector<cv::DMatch>> rawMatches;
    std::vector<cv::DMatch> matches;

    auto matcher2 = cv::BFMatcher(cv::NORM_HAMMING);
    matcher2.knnMatch(src, dst, rawMatches, 2);

    for (auto knn : rawMatches)
        if (knn[0].distance / knn[1].distance < 0.8)
            matches.push_back(knn[0]);
        else if (allowAmbiguity)
        {
            matches.push_back(knn[0]);
            matches.push_back(knn[1]);
        }

    return matches;
}

std::vector<SE3> Localizer::getWorldTransform(
    const std::vector<std::vector<Vec3d>> &src,
    const std::vector<std::vector<Vec3d>> &dst)
{
    std::vector<SE3> result;
    const auto &numLists = src.size();
    printf("total of %lu hypotheses need to be computed...\n", numLists);

    for (auto i = 0; i < numLists; ++i)
    {
        SE3 estimate;
        size_t numInliers;

        runRansacAO(
            src[i],
            dst[i],
            estimate,
            numInliers,
            200);

        printf("hypothesis %i has an inlier ratio of %lu / %lu...\n", i, numInliers, src[i].size());
        result.push_back(estimate);
    }

    return result;
}

bool Localizer::getRelocHypotheses(
    const std::shared_ptr<Map> map,
    const std::vector<Vec3d> &framePts,
    const Mat framePtDesc,
    const std::vector<bool> &framePtValid,
    std::vector<SE3> &estimateList,
    const bool &useGraphMatching)
{
    estimateList.clear();
    auto &mapPts = map->getMapPointsAll();
    auto mapPtDesc = map->getPointDescriptorsAll();

    // get a rough match of the map points and frame points
    auto matches = getMatches2NN(framePtDesc, mapPtDesc, useGraphMatching);
    auto numPointPairs = matches.size();

    if (numPointPairs < MinimumNumMatches)
    {
        printf("Too few points matched(%lu)! abort...\n", numPointPairs);
        return false;
    }

    std::vector<std::vector<Vec3d>> src;
    std::vector<std::vector<Vec3d>> dst;

    if (useGraphMatching)
    {
        // refine the match result
        // Mat srcPointPos(1, numPointPairs, CV_32FC3);
        // Mat dstPointPos(1, numPointPairs, CV_32FC3);
        // Mat descriptorDist(1, numPointPairs, CV_32FC3);

        // for (auto i = 0; i < numPointPairs; ++i)
        // {
        //     const auto &match = matches[i];
        //     auto &mapPt = mapPts[match.trainIdx];
        //     auto &framePt = framePts[match.queryIdx];

        //     descriptorDist.ptr<float>(0)[i] = match.distance;
        //     srcPointPos.ptr<Vec3f>(0)[i] = mapPt->getPosWorld().cast<float>();
        //     dstPointPos.ptr<Vec3f>(0)[i] = framePt->getPosWorld().cast<float>();
        // }

        // auto adjacencyMat = createAdjacencyMat(
        //     numPointPairs,
        //     descriptorDist,
        //     srcPointPos,
        //     dstPointPos);
    }
    else
    {
        std::vector<Vec3d> srcTemp, dstTemp;
        for (const auto &match : matches)
        {
            auto &mapPt = mapPts[match.trainIdx];
            auto &framePt = framePts[match.queryIdx];

            if (!mapPt ||
                (mapPt && mapPt->isBad()) ||
                !framePtValid[match.queryIdx])
                continue;

            srcTemp.push_back(framePt);
            dstTemp.push_back(mapPt->getPosWorld());
        }

        src.push_back(srcTemp);
        dst.push_back(dstTemp);
    }

    estimateList = getWorldTransform(src, dst);
    return true;
}