#include "dataStruct/frame.h"
#include "dataStruct/mapPoint.h"
#include "utils/numType.h"

class KeyFrame
{
public:
    KeyFrame(const Frame &F);

public:
    size_t KFId;
    static size_t nextKFId;

    cv::Mat imRGB, imDepth;
    cv::Mat vmap, nmap;

    cv::Mat descriptors;
    shared_vector<MapPoint> mapPoints;
    std::vector<cv::KeyPoint> keyPoints;
};