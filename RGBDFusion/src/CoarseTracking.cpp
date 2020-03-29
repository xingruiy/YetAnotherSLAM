#include "CoarseTracking.h"

CoarseTracking::CoarseTracking(int w, int h, int minLvl, int maxLvl) : minLvl(minLvl), maxLvl(maxLvl)
{
    nLvl = maxLvl - minLvl + 1;
    for (int lvl = minLvl; lvl < maxLvl; ++lvl)
    {
        int wlvl = w / (1 << lvl);
        int hlvl = w / (1 << lvl);

        this->w.push_back(wlvl);
        this->h.push_back(hlvl);

        float *imgRefLvl = (float *)aligned_alloc(16, sizeof(float) * wlvl * hlvl * 3);
        float *depthRefLvl = (float *)aligned_alloc(16, sizeof(float) * wlvl * hlvl * 3);
        float *imgCurLvl = (float *)aligned_alloc(16, sizeof(float) * wlvl * hlvl * 3);
        float *depthCurLvl = (float *)aligned_alloc(16, sizeof(float) * wlvl * hlvl * 3);

        mvReferenceImage.push_back(imgRefLvl);
        mvTrackingImage.push_back(imgCurLvl);
        mvReferenceDepth.push_back(depthRefLvl);
        mvTrackingDepth.push_back(depthCurLvl);
    }
}

void CoarseTracking::setTrackingReference(cv::Mat img, cv::Mat depth)
{
}

void CoarseTracking::setTrackingTarget(cv::Mat img, cv::Mat depth)
{
    for (int lvl = 0; lvl < nLvl; ++lvl)
    {
    }
}

void CoarseTracking::setCameraCalibration(float fx, float fy, float cx, float cy)
{
    for (int lvl = minLvl; lvl < maxLvl; ++lvl)
    {
        float fxlvl = fx / (1 << lvl);
        float fylvl = fy / (1 << lvl);
        float cxlvl = cx / (1 << lvl);
        float cylvl = cy / (1 << lvl);

        this->fx.push_back(fxlvl);
        this->fy.push_back(fylvl);
        this->cx.push_back(cxlvl);
        this->cy.push_back(cylvl);
        this->fxi.push_back(1.0 / fxlvl);
        this->fyi.push_back(1.0 / fylvl);
    }
}

Sophus::SE3d CoarseTracking::getCoarseAlignment(const Sophus::SE3d &Tini)
{
    Sophus::SE3d estimate = Tini;
    Sophus::SE3d lastSuccessEstimate = Tini;
}