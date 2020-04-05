#ifndef PANGOLIN_VIEWER_H
#define PANGOLIN_VIEWER_H

#include "BaseOutput.h"
#include <Eigen/Core>
#include <mutex>

namespace slam
{

class PangolinViewer : public BaseOutput
{
public:
    PangolinViewer();

    void publishFrame(Frame *F);
    void publishKeyFrame();
    void publishGraph();

protected:
    void Run();
    void displayLiveCam();
    void displayKeyFrame();

    const int width = 640;
    const int height = 480;
    float fx, fy, cx, cy;

    Eigen::Matrix3f Kinv;
    std::mutex camPoseMutex;
    Eigen::Matrix4f camToWorld;
};

} // namespace slam

#endif