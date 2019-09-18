#pragma once
#include <memory>
#include "utils/numType.h"
#include "utils/frame.h"

class GlobalMap
{
    std::vector<std::shared_ptr<Frame>> keyframeOptWin;
    std::vector<std::shared_ptr<Frame>> keyframeHistory;
    std::vector<std::pair<SE3, std::shared_ptr<Frame>>> frameHistory;

public:
    GlobalMap();

    void reset();
    void addFrameHistory(const SE3 &T);
    void addReferenceFrame(std::shared_ptr<Frame> frame);
};