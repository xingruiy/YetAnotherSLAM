#ifndef BASE_IO_SYSTEM_H
#define BASE_IO_SYSTEM_H

namespace slam
{

class Frame;

class BaseOutput
{
public:
    BaseOutput() : isFinished(false) {}
    ~BaseOutput() { isFinished = true; }

    virtual void publishFrame(Frame *F) = 0;
    virtual void publishKeyFrame() = 0;
    virtual void publishGraph() = 0;

    bool isFinished;
};

} // namespace slam

#endif