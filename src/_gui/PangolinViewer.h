#ifndef _PANGOLIN_WRAPPER_H
#define _PANGOLIN_WRAPPER_H

#include <pangolin/pangolin.h>
#include <pangolin/gl/glcuda.h>
#include <mutex>
#include "BaseIOWrapper.h"

namespace slam
{

class PangolinViewer : public BaseIOWrapper
{
public:
    PangolinViewer(int w, int h);

    void run();
    void setGlobalMap(Map *map);
    void setSystemIO(FullSystem *fs);

protected:
    void drawMapPoints(int dSize);
    void drawKeyFrames(bool drawGraph, bool drawKF, int N);

    int w, h;
    Map *map;
    FullSystem *fsIO;
};

} // namespace slam

#endif