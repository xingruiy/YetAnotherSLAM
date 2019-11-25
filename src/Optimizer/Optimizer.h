#pragma once
#include "DataStruct/keyFrame.h"
#include "DataStruct/map.h"
#include "DataStruct/mapPoint.h"
#include <g2o/types/sba/types_six_dof_expmap.h>

void LocalBundleAdjustment(std::shared_ptr<KeyFrame> KF, Map *map);