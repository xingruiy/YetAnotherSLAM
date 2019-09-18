#pragma once
#include <memory>
#include "utils/numType.h"
#include "utils/frame.h"
#include "localMapper/denseMap.h"

class LocalMapper
{
  int frameWidth;
  int frameHeight;
  Mat33d intrinsics;

  GMat zrangeX;
  GMat zrangeY;
  MapStruct deviceMap;
  RenderingBlock *listRenderingBlock;
  uint *numTriangle;
  uint *numRenderingBlock;
  uint numVisibleBlock;

  void preAllocateBlock(GMat depth, const SE3 &T);
  void checkBlockInFrustum(const SE3 &T);
  void projectVisibleBlock(const SE3 &T);
  void predictDepthMap(uint renderingBlockNum);

public:
  LocalMapper(int w, int h, Mat33d &K);
  ~LocalMapper();

  void fuseFrame(GMat depth, const SE3 &T);
  void raytrace(GMat &vertex, const SE3 &T);
  size_t getSceneMesh(float *vertexBuffer, float *normalBuffer, size_t bufferSize);
  void reset();
};