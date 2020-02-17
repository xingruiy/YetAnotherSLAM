#pragma once
#include "utils/numType.h"
#include "DataStruct/frame.h"
#include "MapViewer/mapViewer.h"
#include "DataStruct/keyFrame.h"

class Relocalizer
{
public:
    inline void setViewer(MapViewer *viewer)
    {
        display = viewer;
    }

    inline void setTargetFrame(const Frame &F)
    {
        currFrame = F;
    }

    inline void setDstPoints(std::vector<std::shared_ptr<MapPoint>> *pts)
    {
        mapPoints = pts;
    }

    void run();

public:
    Frame currFrame;
    MapViewer *display;
    std::vector<std::shared_ptr<MapPoint>> *mapPoints;

    std::vector<SE3> RTProposals;
    std::vector<std::vector<bool>> filter;
    std::vector<std::vector<cv::DMatch>> matches;
};