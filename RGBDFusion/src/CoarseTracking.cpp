#include "CoarseTracking.h"

CoarseTracking::CoarseTracking(int w, int h, float fx, float fy, float cx, float cy) : lastlastF(nullptr), lastF(nullptr)
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
    }
}

void CoarseTracking::AddFrame(float *img, float *depth, double timestamp)
{
    FrameShell *F = new FrameShell();
    MakeImages(F, img, depth);

    F->timestamp = timestamp;

    auto t1 = std::chrono::high_resolution_clock::now();
    if (FsHist.size() != 0)
        trackCoarseLevel(F);
    auto t2 = std::chrono::high_resolution_clock::now();
    std::cout << "track images cost: " << std::chrono::duration_cast<std::chrono::microseconds>(t2 - t1).count() << " us" << std::endl;

    FsHist.push_back(F);
    lastlastF = lastF;
    lastF = F;

    if (lastlastF)
        for (int i = 0; i < NUM_PYR; ++i)
        {
            delete lastlastF->gradSqrPyr[i];
            delete lastlastF->imgPyr[i];
            delete lastlastF->depthPyr[i];
        }

    // int lvl = 1;
    // int wlvl = w[lvl];
    // int hlvl = h[lvl];
    // int nlvl = wlvl * wlvl;
    // float *imgtemp = new float[nlvl];
    // for (int idx = 0; idx < nlvl; ++idx)
    // {
    //     imgtemp[idx] = F->imgPyr[lvl][idx][0];
    // }
    // cv::Mat out(hlvl, wlvl, CV_32FC1, imgtemp);
    // cv::Mat out2;
    // cv::Mat out2(hlvl, wlvl, CV_32FC3, F->depthPyr[lvl]);
    // out.convertTo(out2, CV_8UC1);
    // cv::imshow("out", out2);
    // cv::waitKey(0);
}

inline Eigen::Vector3f GetInterpolateElementVec3(Eigen::Vector3f *I, float u, float v, int w)
{
    int x = static_cast<int>(std::floor(u));
    int y = static_cast<int>(std::floor(v));
    float dx = u - x, dy = v - y;

    return (1 - dx) * (1 - dy) * I[y * w + x] +
           dx * (1 - dy) * I[y * w + x + 1] +
           (1 - dx) * dy * I[y * w + w + x] +
           dx * dy * I[y * w + w + x + 1];
}

void CoarseTracking::discardLastFrame()
{
}

void CoarseTracking::trackCoarseLevel(FrameShell *F)
{
    FrameShell *lastF = FsHist.back();

    estimate = Sophus::SE3d();
    Sophus::SE3f estimateF;
    float gradSqrTh[NUM_PYR] = {64, 49, 36, 25, 16};
    for (int lvl = NUM_PYR - 1; lvl >= 0; --lvl)
    {
        int wlvl = w[lvl];
        int hlvl = h[lvl];
        int nlvl = wlvl * hlvl;
        float fxlvl = fx[lvl];
        float fylvl = fy[lvl];
        float cxlvl = cx[lvl];
        float cylvl = cy[lvl];

        float *gradSqrLvl = lastF->gradSqrPyr[lvl];
        Eigen::Vector3f *lastF_img = lastF->imgPyr[lvl];
        Eigen::Vector<float, 3> *lastF_dp = lastF->depthPyr[lvl];
        Eigen::Vector3f *F_img = F->imgPyr[lvl];
        Eigen::Vector<float, 3> *F_dp = F->depthPyr[lvl];

        Eigen::Matrix<float, 6, 6> Hessian;
        Eigen::Vector<float, 6> Residual;

        for (int iter = 0; iter < 5; ++iter)
        {
            estimateF = estimate.cast<float>();

            Hessian.setZero();
            Residual.setZero();
            float costSum = 0;

            cv::Mat out(hlvl, wlvl, CV_8UC3);
            out.setTo(0);

            for (int y = 1; y < hlvl - 1; ++y)
            {
                for (int x = 1; x < wlvl - 1; ++x)
                {
                    float gradSqr = gradSqrLvl[y * wlvl + x];
                    if (gradSqr < gradSqrTh[lvl])
                        continue;

                    Eigen::Vector3f pt = lastF_dp[y * wlvl + x];

                    if (pt(2) == 0)
                        continue;

                    Eigen::Vector3f ptWarped = estimateF * pt;
                    float idepth_new = 1.0 / ptWarped(2);
                    float u = fxlvl * ptWarped(0) * idepth_new + cxlvl;
                    float v = fylvl * ptWarped(1) * idepth_new + cylvl;

                    if (u < 3 || v < 3 || u >= wlvl - 3 || v >= hlvl - 3)
                        continue;

                    Eigen::Vector3f hit = GetInterpolateElementVec3(F_img, u, v, wlvl);
                    if (!std::isfinite(hit(0)) || !std::isfinite(hit(1)) || !std::isfinite(hit(2)))
                        continue;

                    out.ptr<cv::Vec3b>(y)[x](0) = lastF_img[y * wlvl + x][0];
                    out.ptr<cv::Vec3b>(y)[x](1) = 255;
                    out.ptr<cv::Vec3b>(y)[x](2) = lastF_img[y * wlvl + x][0];

                    float res = hit[0] - lastF_img[y * wlvl + x][0];
                    const float huberTh = 12;
                    float hw = abs(res) < huberTh ? 1 : huberTh / abs(res);
                    Eigen::Vector<float, 6> J;
                    J[0] = hw * hit[1] * fxlvl * idepth_new;
                    J[1] = hw * hit[2] * fylvl * idepth_new;
                    J[2] = -(J[0] * ptWarped(0) + J[1] * ptWarped(1)) * idepth_new;
                    J[3] = J[2] * ptWarped(1) - hw * hit[2] * fylvl;
                    J[4] = -J[2] * ptWarped(0) + hw * hit[1] * fxlvl;
                    J[5] = -J[0] * ptWarped(1) + J[1] * ptWarped(0);

                    Hessian += J * J.transpose();
                    Residual -= J * res;
                    costSum += hw * res * res;
                }
            }

            Eigen::Vector<double, 6> dxi = Hessian.cast<double>().ldlt().solve(Residual.cast<double>());

            if (lvl == 0)
            {
                cv::imshow("out", out);
                cv::waitKey(1);
            }

            if (dxi.norm() < 1e-3)
                break;

            estimate = Sophus::SE3d::exp(dxi) * estimate;
        }
    }
}

void CoarseTracking::MakeImages(FrameShell *F, float *img, float *depth)
{
    // Make images
    for (int lvl = 0; lvl < NUM_PYR; ++lvl)
    {
        int wlvl = w[lvl];
        int hlvl = h[lvl];
        int nlvl = wlvl * hlvl;

        F->imgPyr[lvl] = new Eigen::Vector3f[nlvl];
        F->gradSqrPyr[lvl] = new float[nlvl];

        float *gradSqrLvl = F->gradSqrPyr[lvl];
        Eigen::Vector3f *Ilvl = F->imgPyr[lvl];

        if (lvl == 0)
        {
            for (int i = 0; i < nlvl; ++i)
                Ilvl[i][0] = img[i];
        }
        else
        {
            int wm1 = w[lvl - 1];
            Eigen::Vector3f *Ilm1 = F->imgPyr[lvl - 1];

            for (int y = 0; y < hlvl; ++y)
                for (int x = 0; x < wlvl; ++x)
                {
                    Ilvl[x + y * wlvl][0] = 0.25 * (Ilm1[y * 2 * wm1 + x * 2][0] +
                                                    Ilm1[y * 2 * wm1 + x * 2 + 1][0] +
                                                    Ilm1[y * 2 * wm1 + wm1 + x * 2][0] +
                                                    Ilm1[y * 2 * wm1 + wm1 + x * 2 + 1][0]);
                }
        }

        for (int i = wlvl; i < wlvl * (hlvl - 1); ++i)
        {
            float dx = 0.5 * (Ilvl[i + 1][0] - Ilvl[i - 1][0]);
            float dy = 0.5 * (Ilvl[i + wlvl][0] - Ilvl[i - lvl][0]);

            if (!std::isfinite(dx))
                dx = 0;
            if (!std::isfinite(dy))
                dy = 0;

            Ilvl[i][1] = dx;
            Ilvl[i][2] = dy;
            gradSqrLvl[i] = dx * dx + dy * dy;
        }
    }

    F->img = F->imgPyr[0];

    // Make depth maps
    for (int lvl = 0; lvl < NUM_PYR; ++lvl)
    {
        int wlvl = w[lvl];
        int hlvl = h[lvl];
        int nlvl = wlvl * hlvl;
        float fxilvl = fxi[lvl];
        float fyilvl = fyi[lvl];
        float cxlvl = cx[lvl];
        float cylvl = cy[lvl];

        F->depthPyr[lvl] = new Eigen::Vector<float, 3>[nlvl];
        Eigen::Vector<float, 3> *Dlvl = F->depthPyr[lvl];

        if (lvl == 0)
        {
            for (int i = 0; i < nlvl; ++i)
                Dlvl[i][2] = depth[i];
        }
        else
        {
            int wm1 = w[lvl - 1];
            Eigen::Vector<float, 3> *Dlm1 = F->depthPyr[lvl - 1];

            for (int y = 0; y < hlvl; ++y)
                for (int x = 0; x < wlvl; ++x)
                {
                    float z = 0.25 * (Dlm1[y * 2 * wm1 + x * 2][2] +
                                      Dlm1[y * 2 * wm1 + x * 2 + 1][2] +
                                      Dlm1[y * 2 * wm1 + wm1 + x * 2][2] +
                                      Dlm1[y * 2 * wm1 + wm1 + x * 2 + 1][2]);
                    if (std::isnan(z))
                        z = 0;
                    Dlvl[x + y * wlvl][2] = z;
                }
        }

        for (int idx = 0; idx < nlvl; ++idx)
        {
            int y = idx / wlvl;
            int x = idx - y * wlvl;
            const float &z = Dlvl[idx][2];
            Dlvl[idx][0] = (x - cxlvl) * fxilvl * z;
            Dlvl[idx][1] = (y - cylvl) * fyilvl * z;
        }
    }
}
