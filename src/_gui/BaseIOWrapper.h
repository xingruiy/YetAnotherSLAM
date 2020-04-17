#ifndef _BASE_IO_WRAPPER_H
#define _BASE_IO_WRAPPER_H

namespace slam
{

class Map;
class Frame;
class FullSystem;

class BaseIOWrapper
{
public:
    inline BaseIOWrapper() {}
    inline ~BaseIOWrapper() {}

    virtual void publishLiveFrame(Frame *newF) = 0;
    virtual void setGlobalMap(Map *map) = 0;
    virtual void setSystemIO(FullSystem *fs) = 0;
};

} // namespace slam

#endif