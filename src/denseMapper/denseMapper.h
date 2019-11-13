#pragma once
#include <memory>
#include "utils/numType.h"
#include "denseMapper/denseMap.h"

class DenseMapping
{
  Mat33d intrinsics;
  MapStruct deviceMap;

  // for map udate
  uint count_visible_block;

  // for raycast
  GMat zRangeX;
  GMat zRangeY;
  uint count_rendering_block;
  RenderingBlock *listRenderingBlock;

public:
  ~DenseMapping();
  DenseMapping(int w, int h, Mat33d &K);

  void reset();
  void fuseFrame(GMat depth, const SE3 &T);
  void raytrace(GMat &vertex, const SE3 &T);
  size_t fetchMeshWithNormal(void *vertex, void *normal);
};
