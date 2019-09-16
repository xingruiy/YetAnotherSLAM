#pragma once
#include <memory>
#include "utils/frame.h"
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

    const int numTrackingLvl;
    std::vector<int> iterationPerLvl;

    std::vector<int> frameWidth;
    std::vector<int> frameHeight;
    std::vector<Mat33d> intrinsics;

    std::vector<GMat> currentIntensity;
    std::vector<GMat> referenceIntensity;
    std::vector<GMat> currentInvDepth;
    std::vector<GMat> referenceInvDepth;
    std::vector<GMat> referencePointWarped;
    std::vector<GMat> invDepthGradientX;
    std::vector<GMat> invDepthGradientY;
    std::vector<GMat> IntensityGradientX;
    std::vector<GMat> IntensityGradientY;

    void computeSE3StepRGB(
        const int lvl,
        SE3 &estimate,
        float *hessian,
        float *residual);

    float residualSum;
    float iResidualSum;
    float dResidualSum;
    float numResidual;

public:
    DenseTracker(int w, int h, Mat33d &K, int numLvl);
    void setReferenceInvDepth(GMat refInvDepth);
    void setReferenceFrame(std::shared_ptr<Frame> ref);
    SE3 getIncrementalTransform(std::shared_ptr<Frame> frame, SE3 initAlign = SE3(), bool switchBuffer = true);
};