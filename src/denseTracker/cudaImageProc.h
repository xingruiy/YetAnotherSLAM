#pragma once
#include "utils/numType.h"
#include <cuda_runtime_api.h>

void convertRGBtoIntensity(GMat image, GMat intensity);
void computeVMap(GMat depth, GMat vamp, float fx, float fy, float cx, float cy);
void computeNMap(GMat vmap, GMat nmap);
