#ifndef FUSION_VOXEL_HASHING_MAP_PROC
#define FUSION_VOXEL_HASHING_MAP_PROC

#include <sophus/se3.hpp>
#include <opencv2/cudaarithm.hpp>
#include "localMapper2/denseMap.h"

void update(
    MapStruct map_struct,
    // MapState state,
    const cv::cuda::GpuMat depth,
    const cv::cuda::GpuMat image,
    const Sophus::SE3d &frame_pose,
    const Eigen::Matrix3d &K,
    cv::cuda::GpuMat &cv_flag,
    cv::cuda::GpuMat &cv_pos_array,
    HashEntry *visible_blocks,
    uint &visible_block_count);

void update_weighted(
    MapStruct map_struct,
    // MapState state,
    const cv::cuda::GpuMat depth,
    const cv::cuda::GpuMat normal,
    const cv::cuda::GpuMat image,
    const Sophus::SE3d &frame_pose,
    const Eigen::Matrix3d &K,
    cv::cuda::GpuMat &cv_flag,
    cv::cuda::GpuMat &cv_pos_array,
    HashEntry *visible_blocks,
    uint &visible_block_count);

void create_rendering_blocks(
    MapStruct map_struct,
    uint count_visible_block,
    uint &count_rendering_block,
    HashEntry *visible_blocks,
    GMat &zrange_x,
    GMat &zrange_y,
    RenderingBlock *listRenderingBlock,
    const SE3 &frame_pose,
    const Mat33d &cam_params);

void raycast(
    MapStruct map_struct,
    // MapStruct map_struct,
    // MapState state,
    cv::cuda::GpuMat vmap,
    cv::cuda::GpuMat nmap,
    cv::cuda::GpuMat zrange_x,
    cv::cuda::GpuMat zrange_y,
    const Sophus::SE3d &pose,
    const Eigen::Matrix3d &K);

void raycast_with_colour(
    MapStruct map_struct,
    // MapState state,
    cv::cuda::GpuMat vmap,
    cv::cuda::GpuMat nmap,
    cv::cuda::GpuMat image,
    cv::cuda::GpuMat zrange_x,
    cv::cuda::GpuMat zrange_y,
    const Sophus::SE3d &pose,
    const Eigen::Matrix3d &K);

// void create_mesh_vertex_only(
//     MapStruct map_struct,
//     MapState state,
//     uint &block_count,
//     HashEntry *block_list,
//     uint &triangle_count,
//     void *vertex_data);

void create_mesh_with_normal(
    MapStruct map_struct,
    // MapState state,
    uint &block_count,
    HashEntry *block_list,
    uint &triangle_count,
    void *vertex_data,
    void *vertex_normal);

// void create_mesh_with_colour(
//     MapStruct map_struct,
//     MapState state,
//     uint &block_count,
//     HashEntry *block_list,
//     uint &triangle_count,
//     void *vertex_data,
//     void *vertex_colour);

void count_visible_entry(
    const MapStruct map_struct,
    // const MapSize map_size,
    const Eigen::Matrix3d &K,
    const Sophus::SE3d frame_pose,
    HashEntry *const visible_entry,
    uint &visible_block_count);

#endif