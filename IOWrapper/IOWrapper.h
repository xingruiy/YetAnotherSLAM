#ifndef IO_WRAPPER_H
#define IO_WRAPPER_H

#include <Eigen/Core>
#include <opencv2/opencv.hpp>

class FrameShell
{
public:
    FrameShell();
};

class IOWrapper
{
public:
    virtual void AddKeyFrame(const unsigned long id, const Eigen::Matrix4d &Tcw) = 0;
    virtual void UpdateKeyFrame(const unsigned long id, const Eigen::Matrix4d &Tcw) = 0;

private:
};

#endif