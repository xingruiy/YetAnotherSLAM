#pragma once

#include "VoxelMap.h"
#include <opencv2/opencv.hpp>
#include <Eigen/Core>

void fuseDepth(
    MapStruct map_struct,
    const cv::cuda::GpuMat depth,
    const Sophus::SE3d &T,
    const Eigen::Matrix3d &K,
    unsigned int frameId,
    uint &visible_block_count);

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

void create_mesh_with_normal(
    MapStruct map_struct,
    uint &block_count,
    uint &triangle_count,
    void *vertexBuffer,
    void *normalBuffer,
    size_t bufferSize = 0);
