#pragma once
#include "utils/frame.h"
#include "utils/mapPoint.h"

class Localizer
{
public:
    Localizer();
    SE3 getRelativeTransform(
        std::shared_ptr<Frame> reference,
        std::shared_ptr<Frame> current);
    SE3 getWorldTransform(
        std::shared_ptr<Frame> frame,
        std::vector<std::shared_ptr<MapPoint>> pts);
};