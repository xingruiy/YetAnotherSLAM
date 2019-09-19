#pragma once
#include "utils/numType.h"
#include "localMapper/denseMap.h"

void fuseDepth(
    MapStruct map,
    const GMat depth,
    const SE3 &T,
    const Mat33d &K,
    uint &visible_block_count);

void create_rendering_blocks(
    MapStruct map_struct,
    uint count_visible_block,
    uint &count_rendering_block,
    GMat &zrange_x,
    GMat &zrange_y,
    RenderingBlock *listRenderingBlock,
    const SE3 &T,
    const Mat33d &K);

void raycast(MapStruct map, GMat vmap, GMat zRangeX, GMat zRangeY, const SE3 &T, const Mat33d &K);

void create_mesh_with_normal(
    MapStruct map,
    uint &block_count,
    uint &triangle_count,
    void *vertex_data,
    void *vertex_normal);
