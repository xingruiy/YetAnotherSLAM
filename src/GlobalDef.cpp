#include "GlobalDef.h"

namespace SLAM
{

bool g_bReverseRGB = false;
bool g_bSystemRunning = false;
bool g_bSystemKilled = false;
bool g_bEnableViewer = false;

float g_DepthScaleInv = 0.01;
bool g_bUseColour = true;
bool g_bUseDepth = false;
float g_thDepth = 40;
float g_bf = 40;

// Feature extraction
float g_ORBScaleFactor = 1.2;
int g_ORBNFeatures = 1000;
int g_ORBNLevels = 8;
int g_ORBIniThFAST = 20;
int g_ORBMinThFAST = 7;

// Global calibration
int g_width[NUM_PYR];
int g_height[NUM_PYR];
float g_fx[NUM_PYR];
float g_fy[NUM_PYR];
float g_cx[NUM_PYR];
float g_cy[NUM_PYR];
float g_invfx[NUM_PYR];
float g_invfy[NUM_PYR];
float g_invcx[NUM_PYR];
float g_invcy[NUM_PYR];
Eigen::Matrix3f g_calib[NUM_PYR];
Eigen::Matrix3f g_calibInv[NUM_PYR];
cv::Mat g_distCoeff;
std::mutex g_calibMutex;

void setGlobalCalibration(const int width, const int height, const Eigen::Matrix3d &K)
{
    std::unique_lock<std::mutex>(g_calibMutex);
    for (int level = 0; level < NUM_PYR; ++level)
    {
        g_width[level] = width >> level;
        g_height[level] = height >> level;

        if (level == 0)
        {
            g_fx[0] = K(0, 0);
            g_fy[0] = K(1, 1);
            g_cx[0] = K(0, 2);
            g_cy[0] = K(1, 2);
        }
        else
        {
            g_fx[level] = g_fx[level - 1] * 0.5;
            g_fy[level] = g_fy[level - 1] * 0.5;
            g_cx[level] = (g_cx[0] + 0.5) / ((int)1 << level) - 0.5;
            g_cy[level] = (g_cy[0] + 0.5) / ((int)1 << level) - 0.5;
        }

        g_calib[level] << g_fx[level], 0.0, g_cx[level],
            0.0, g_fy[level], g_cy[level],
            0.0, 0.0, 1.0;
        g_calibInv[level] = g_calib[level].inverse();

        g_invfx[level] = g_calibInv[level](0, 0);
        g_invfy[level] = g_calibInv[level](1, 1);
        g_invcx[level] = g_calibInv[level](0, 2);
        g_invcy[level] = g_calibInv[level](1, 2);
    }
}

} // namespace SLAM
