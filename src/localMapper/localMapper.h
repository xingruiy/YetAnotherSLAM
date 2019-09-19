#pragma once
#include <memory>
#include "utils/numType.h"
#include "localMapper/denseMap.h"

class DenseMapping
{
public:
  ~DenseMapping();
  DenseMapping(int w, int h, Mat33d &K);

  size_t fetch_mesh_with_normal(void *vertex, void *normal);
  void fuseFrame(GMat depth, const SE3 &T);
  void raytrace(GMat &vertex, const SE3 &T);
  void reset();

private:
  Mat33d intrinsics;
  MapStruct deviceMap;

  // for map udate
  GMat flag;
  GMat pos_array;
  uint count_visible_block;
  HashEntry *visible_blocks;

  // for raycast
  GMat zrange_x;
  GMat zrange_y;
  uint count_rendering_block;
  RenderingBlock *rendering_blocks;
};
