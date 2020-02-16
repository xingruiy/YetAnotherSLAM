#pragma once
#include <opencv2/opencv.hpp>
#include <Eigen/Dense>
#include <mutex>
#include <ORBextractor.h>

namespace SLAM
{

#define FRAME_GRID_ROWS 48
#define FRAME_GRID_COLS 64

extern bool g_bReverseRGB;
extern bool g_bSystemRunning;
extern bool g_bSystemKilled;
extern bool g_bEnableViewer;

extern float g_DepthScaleInv;
extern bool g_bUseColour;
extern bool g_bUseDepth;
extern float g_thDepth;
extern float g_bf;

extern float g_ORBScaleFactor;
extern int g_ORBNFeatures;
extern int g_ORBNLevels;
extern int g_ORBIniThFAST;
extern int g_ORBMinThFAST;

#define NUM_PYR 5

extern int g_width[NUM_PYR];
extern int g_height[NUM_PYR];
extern float g_fx[NUM_PYR];
extern float g_fy[NUM_PYR];
extern float g_cx[NUM_PYR];
extern float g_cy[NUM_PYR];
extern float g_invfx[NUM_PYR];
extern float g_invfy[NUM_PYR];
extern float g_invcx[NUM_PYR];
extern float g_invcy[NUM_PYR];
extern Eigen::Matrix3f g_calib[NUM_PYR];
extern Eigen::Matrix3f g_calibInv[NUM_PYR];
extern cv::Mat g_cvCalib;
extern cv::Mat g_distCoeff;
extern std::mutex g_calibMutex;
extern float g_minX, g_minY, g_maxX, g_maxY;
extern float g_gridElementWidthInv;
extern float g_gridElementHeightInv;

void setGlobalCalibration(const int width, const int height, const Eigen::Matrix3d &K);
void computeImageBounds();

extern float g_pointSize;

} // namespace SLAM
