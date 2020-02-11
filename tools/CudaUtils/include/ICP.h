#pragma once

class ICP
{
public:
    static void computeSingleStepRGB(cv::cuda::GpuMat &vmapCurr, cv::cuda::GpuMat &vmapLast);
};