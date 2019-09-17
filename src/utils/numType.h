#pragma once
#include <Eigen/Core>
#include <sophus/se3.hpp>
#include <opencv2/opencv.hpp>

#undef EIGEN_DEFAULT_DENSE_INDEX_TYPE
#define EIGEN_DEFAULT_DENSE_INDEX_TYPE int

using uchar = unsigned char;
using ushort = unsigned short;

// Eigen Shortcuts
using Mat33f = Eigen::Matrix<float, 3, 3>;
using Mat33d = Eigen::Matrix<double, 3, 3>;
using Mat34f = Eigen::Matrix<float, 3, 4>;
using Mat44f = Eigen::Matrix<float, 4, 4>;
using Mat44d = Eigen::Matrix<double, 4, 4>;
using Mat22f = Eigen::Matrix<float, 2, 2>;
using Mat22d = Eigen::Matrix<double, 2, 2>;
using Mat26f = Eigen::Matrix<float, 2, 6>;
using Mat66f = Eigen::Matrix<float, 6, 6>;

using Vec2b = Eigen::Matrix<uchar, 2, 1>;
using Vec2s = Eigen::Matrix<short, 2, 1>;
using Vec2i = Eigen::Matrix<int, 2, 1>;
using Vec2f = Eigen::Matrix<float, 2, 1>;
using Vec2d = Eigen::Matrix<double, 2, 1>;
using Vec3b = Eigen::Matrix<uchar, 3, 1>;
using Vec3i = Eigen::Matrix<int, 3, 1>;
using Vec3f = Eigen::Matrix<float, 3, 1>;
using Vec3d = Eigen::Matrix<double, 3, 1>;
using Vec4b = Eigen::Matrix<uchar, 4, 1>;
using Vec4f = Eigen::Matrix<float, 4, 1>;
using Vec4d = Eigen::Matrix<double, 4, 1>;
using Vec6f = Eigen::Matrix<float, 6, 1>;
using Vec6d = Eigen::Matrix<double, 6, 1>;
using Vec7f = Eigen::Matrix<float, 7, 1>;
using Vec7d = Eigen::Matrix<double, 7, 1>;
using Vec9f = Eigen::Matrix<float, 9, 1>;
using Vec9d = Eigen::Matrix<double, 9, 1>;
using Vec25f = Eigen::Matrix<float, 25, 1>;
using Vec25d = Eigen::Matrix<double, 25, 1>;
using Vec29f = Eigen::Matrix<float, 29, 1>;
using Vec29d = Eigen::Matrix<double, 29, 1>;

// Sophus Shortcuts
using SE3 = Sophus::SE3d;
using SO3 = Sophus::SO3d;
using SE3f = Sophus::SE3f;

// OpenCV Shortcuts
using Mat = cv::Mat;
using GMat = cv::cuda::GpuMat;

#define CV_32FC6 CV_32FC(6)
#define CV_32FC7 CV_32FC(7)