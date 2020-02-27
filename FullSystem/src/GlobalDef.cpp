#include "GlobalDef.h"
#include <cmath>

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
cv::Mat g_cvCalib;
cv::Mat g_distCoeff;
std::mutex g_calibMutex;
float g_minX, g_minY, g_maxX, g_maxY;
float g_gridElementWidthInv;
float g_gridElementHeightInv;

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

    g_cvCalib = cv::Mat::eye(3, 3, CV_32F);
    g_cvCalib.at<float>(0, 0) = g_fx[0];
    g_cvCalib.at<float>(1, 1) = g_fy[0];
    g_cvCalib.at<float>(0, 2) = g_cx[0];
    g_cvCalib.at<float>(1, 2) = g_cy[0];
}

void computeImageBounds()
{
    int cols = g_width[0];
    int rows = g_height[0];

    if (g_distCoeff.at<float>(0) != 0.0)
    {
        cv::Mat mat(4, 2, CV_32F);
        mat.at<float>(0, 0) = 0.0;
        mat.at<float>(0, 1) = 0.0;
        mat.at<float>(1, 0) = cols;
        mat.at<float>(1, 1) = 0.0;
        mat.at<float>(2, 0) = 0.0;
        mat.at<float>(2, 1) = rows;
        mat.at<float>(3, 0) = cols;
        mat.at<float>(3, 1) = rows;

        // Undistort corners
        mat = mat.reshape(2);
        cv::undistortPoints(mat, mat, g_cvCalib, g_distCoeff, cv::Mat(), g_cvCalib);
        mat = mat.reshape(1);

        g_minX = std::min(mat.at<float>(0, 0), mat.at<float>(2, 0));
        g_maxX = std::max(mat.at<float>(1, 0), mat.at<float>(3, 0));
        g_minY = std::min(mat.at<float>(0, 1), mat.at<float>(1, 1));
        g_maxY = std::max(mat.at<float>(2, 1), mat.at<float>(3, 1));
    }
    else
    {
        g_minX = 0.0f;
        g_maxX = cols;
        g_minY = 0.0f;
        g_maxY = rows;
    }

    g_gridElementWidthInv = static_cast<float>(FRAME_GRID_COLS) / (g_maxX - g_minX);
    g_gridElementHeightInv = static_cast<float>(FRAME_GRID_ROWS) / (g_maxY - g_minY);
}

int g_pointSize = 1;

size_t g_nFailedFrame = 0;
size_t g_nTrackedFrame = 0;
size_t g_nTrackedKeyframe = 0;
ORBextractor *g_pORBExtractor = NULL;

} // namespace SLAM
