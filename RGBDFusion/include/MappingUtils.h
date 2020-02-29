#pragma once

#include "VoxelMap.h"
#include <opencv2/opencv.hpp>
#include <Eigen/Core>
#include <sophus/se3.hpp>

void create_rendering_blocks(
    MapStruct map_struct,
    uint count_visible_block,
    uint &count_rendering_block,
    cv::cuda::GpuMat &zrange_x,
    cv::cuda::GpuMat &zrange_y,
    RenderingBlock *listRenderingBlock,
    const Sophus::SE3d &T,
    const Eigen::Matrix3d &K);

void raycast(
    MapStruct map,
    cv::cuda::GpuMat vmap,
    cv::cuda::GpuMat zRangeX,
    cv::cuda::GpuMat zRangeY,
    const Sophus::SE3d &T,
    const Eigen::Matrix3d &K);
