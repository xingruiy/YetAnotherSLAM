#pragma once
#include "utils/numType.h"
#include <cuda_runtime_api.h>

void renderScene(const GMat vmap, const GMat nmap, GMat &image);
void computeNormal(const GMat vmap, GMat &nmap);

void computeImageGradientCentralDiff(GMat image, GMat &gx, GMat &gy);
void transformReferencePoint(GMat depth, GMat &vmap, const Mat33d &K, const SE3 &T);
void convertDepthToInvDepth(const GMat depth, GMat &invDepth);
void convertVMapToInvDepth(const GMat vmap, GMat &invDepth);
void pyrdownInvDepth(const GMat src, GMat &dst);