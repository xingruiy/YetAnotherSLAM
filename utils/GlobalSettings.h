#ifndef GLOBAL_SETTINGS_H
#define GLOBAL_SETTINGS_H

#include <string>
#include <vector>

namespace slam
{

class GlobalSettings
{
public:
    GlobalSettings(const std::string &file);
    void printDebugInfo();

    // ==== general ====
    bool colourArrangeRGB = true;
    float idepthScale = 0.001;

    // ==== calib params ====
    int wlv0 = -1, hlv0 = -1;
    float fxlv0, fylv0, cxlv0, cylv0;
    float p1, p2, k1, k2, k3;

    // ==== dense direct tracking ====
    bool useRGB = true;
    bool useDepth = true;
    int coarsestlvlForTracking = 6;
    std::vector<float> imgGrad2ThForTracking = {64, 49, 36, 25, 16, 9};
    std::vector<int> maxIterationsForTracking = {10, 10, 20, 20, 20, 30};

    // ==== debug ====
    bool verbose = false;
    int verboselvl = -1;
    bool displayDebugImages = false;

protected:
    void readFromFile(const std::string &file);
};

} // namespace slam

#endif