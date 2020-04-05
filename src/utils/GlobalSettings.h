#ifndef GLOBAL_SETTINGS_H
#define GLOBAL_SETTINGS_H

#include <string>

namespace slam
{

class GlobalSettings
{
public:
    GlobalSettings(const std::string &strSettings);

protected:
    void readSettingsFromFile(const std::string &strSettings);
    void writeSettingsToFile(const std::string &strSettings);

public:
    int widthLvl0 = -1;
    int heightLvl0 = -1;
    float fxLvl0, fyLvl0;
    float cxLvl0, cyLvl0;
    float ifxLvl0, ifyLvl0;

    int coarsestLvlForTracking = 6;
    float minIncForCoarseTracking = 1e-3;

    bool logRawOutputs = false;
};

} // namespace slam

#endif