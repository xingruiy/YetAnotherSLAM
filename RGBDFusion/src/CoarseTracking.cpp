#include "CoarseTracking.h"

CoarseTracking::CoarseTracking(int w, int h, float fx, float fy, float cx, float cy)
{
    for (int lvl = 0; lvl < NUM_PYR; ++lvl)
    {
        int wlvl = w / (1 << lvl);
        int hlvl = h / (1 << lvl);
        this->w[lvl] = wlvl;
        this->h[lvl] = hlvl;
        this->fx[lvl] = fx / (1 << lvl);
        this->fy[lvl] = fy / (1 << lvl);
        this->cx[lvl] = cx / (1 << lvl);
        this->cy[lvl] = cy / (1 << lvl);
        this->fxi[lvl] = 1.0 / this->fx[lvl];
        this->fyi[lvl] = 1.0 / this->fy[lvl];

        referenceImage[lvl] = new float[wlvl * hlvl * 3];
        trackingImage[lvl] = new float[wlvl * hlvl * 3];
    }
}

void CoarseTracking::makeReferenceImage(float *img)
{
    int wlvl = w[0];
    int hlvl = h[0];
    int nlvl = wlvl * hlvl;
    float *I = referenceImage[0];

    for (int i = 0; i < nlvl; ++i)
        I[i * 3] = img[i];

    for (int lvl = 0; lvl < NUM_PYR; ++lvl)
    {
        wlvl = w[lvl];
        hlvl = h[lvl];
        nlvl = wlvl * hlvl;
        float *Ilvl = referenceImage[lvl];

        if (lvl != 0)
        {
            int wlast = w[lvl - 1];
            int hlast = h[lvl - 1];
            float *Ilast = referenceImage[lvl - 1];
            for (int i = 0; i < nlvl; ++i)
            {
                I[i * 3] = 0.25 * (Ilast[i * 3] + Ilast[(i + 1) * 3] + Ilast[(i + wlast) * 3] + Ilast[(i + wlast + 1) * 3]);
            }
        }

        for (int i = 0; i < nlvl; ++i)
        {
        }
    }
}

void CoarseTracking::makeTrackingImage(float *img)
{
}

Sophus::SE3d CoarseTracking::getCoarseAlignment(const Sophus::SE3d &Tini)
{
}