#pragma once
#include <memory>
#include "dataStruct/frame.h"
#include "utils/numType.h"

class DenseTracker
{
    GMat bufferFloat96x29;
    GMat bufferFloat96x3;
    GMat bufferFloat96x2;
    GMat bufferFloat96x1;
    GMat bufferFloat1x29;
    GMat bufferFloat1x3;
    GMat bufferFloat1x2;
    GMat bufferFloat1x1;
    GMat bufferVec4hxw;
    GMat bufferVec7hxw;

    GMat rawDepthBuffer;
    GMat rawImageBuffer;

    const int numTrackingLvl;
    std::vector<int> iterationPerLvl;

    std::vector<int> frameWidth;
    std::vector<int> frameHeight;
    std::vector<Mat33d> intrinsics;

    std::vector<GMat> currentDepth;
    std::vector<GMat> referenceDepth;
    std::vector<GMat> currentIntensity;
    std::vector<GMat> referenceIntensity;
    std::vector<GMat> intensityGradientX;
    std::vector<GMat> intensityGradientY;
    std::vector<GMat> referencePointTransformed;

    std::vector<GMat> currentInvDepth;
    std::vector<GMat> referenceInvDepth;
    std::vector<GMat> invDepthGradientX;
    std::vector<GMat> invDepthGradientY;

    void computeSE3StepRGB(
        const int lvl,
        const SE3 &T,
        float *hessian,
        float *residual);

    void computeSE3StepRGBDLinear(
        const int lvl,
        const SE3 &T,
        float *hessian,
        float *residual);

    void computeSE3StepRGBD(
        const int lvl,
        const SE3 &T,
        float *hessian,
        float *residual);

    void computeSE3StepD(
        const int lvl,
        const SE3 &T,
        float *hessian,
        float *residual);

    void transformReferencePoint(const int lvl, const SE3 &estimate);

    float residualSum;
    float iResidualSum;
    float dResidualSum;
    float numResidual;

public:
    DenseTracker(int w, int h, Mat33d &K, int numLvl);

    void setReferenceInvDepth(GMat refInvDepth);
    void setReferenceFrame(std::shared_ptr<Frame> frame);
    void setTrackingFrame(std::shared_ptr<Frame> frame);

    SE3 getIncrementalTransform(SE3 initAlign = SE3(), bool switchBuffer = true);

    GMat getReferenceDepth(const int lvl = 0) const;
};