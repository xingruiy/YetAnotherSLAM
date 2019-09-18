#pragma once
#include "utils/numType.h"
#include "localMapper/denseMap.h"

void update(
    MapStruct map,
    const GMat depth,
    const SE3 &T,
    const Mat33d &K,
    GMat &cv_flag,
    GMat &cv_pos_array,
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
    GMat vmap,
    GMat nmap,
    GMat zrange_x,
    GMat zrange_y,
    const SE3 &pose,
    const Mat33d &K);

// void raycast_with_colour(
//     MapStruct map_struct,
//     GMat vmap,
//     GMat nmap,
//     GMat image,
//     GMat zrange_x,
//     GMat zrange_y,
//     const SE3 &pose,
//     const Mat33d &K);

void create_mesh_with_normal(
    MapStruct map_struct,
    uint &block_count,
    HashEntry *block_list,
    uint &triangle_count,
    void *vertex_data,
    void *vertex_normal);

// void count_visible_entry(
//     const MapStruct map_struct,
//     // const MapSize map_size,
//     const Mat33d &K,
//     const SE3 frame_pose,
//     HashEntry *const visible_entry,
//     uint &visible_block_count);
