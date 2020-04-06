#include "GlobalSettings.h"
#include <map>
#include <fstream>
#include <sstream>
#include <iostream>

namespace slam
{

GlobalSettings::GlobalSettings(const std::string &strSettings)
{
    readFromFile(strSettings);
}

void GlobalSettings::printDebugInfo()
{
    std::cout << "==== here's some debug info about the system:\n"
              << "w: " << wlv0 << " h: " << hlv0 << "\n"
              << "fx: " << fxlv0 << " fy: " << fylv0 << " cx: " << cxlv0 << " cy: " << cylv0 << "\n"
              << "verbose? - " << (verbose ? "yes" : "no") << " verbose lvl: " << verboselvl << "\n"
              << "inverse depth scale: " << idepthScale
              << std::endl;
}

void GlobalSettings::readFromFile(const std::string &strSettings)
{
    std::ifstream file(strSettings);
    if (file.is_open())
    {
        std::string line;
        std::map<std::string, std::string> params;

        while (getline(file, line))
        {
            std::istringstream lineStream(line);
            std::string key;
            if (std::getline(lineStream, key, '='))
            {
                std::string value;
                if (key[0] == '#' || key[0] == '[')
                    continue;

                if (std::getline(lineStream, value))
                    params[key] = value;
            }
        }

        for (auto mit = params.begin(), mend = params.end(); mit != mend; ++mit)
        {
            if (mit->first == "fx ")
                fxlv0 = std::atof(mit->second.c_str());
            else if (mit->first == "fy ")
                fylv0 = std::atof(mit->second.c_str());
            else if (mit->first == "cx ")
                cxlv0 = std::atof(mit->second.c_str());
            else if (mit->first == "cy ")
                cylv0 = std::atof(mit->second.c_str());
            else if (mit->first == "width ")
                wlv0 = std::atoi(mit->second.c_str());
            else if (mit->first == "height ")
                hlv0 = std::atoi(mit->second.c_str());
            else if (mit->first == "depthScale ")
                idepthScale = 1.0 / std::atof(mit->second.c_str());
            else if (mit->first == "maxIterations ")
            {
                std::stringstream ss(mit->second);
                std::string token;
                while (std::getline(ss, token, ','))
                    maxIterationsForTracking.push_back(std::atoi(token.c_str()));
            }
            else if (mit->first == "imgGrad2Th ")
            {
                std::stringstream ss(mit->second);
                std::string token;
                while (std::getline(ss, token, ','))
                    imgGrad2ThForTracking.push_back(std::atof(token.c_str()));
            }
            else if (mit->first == "displayDebugImages ")
                displayDebugImages = mit->second == " true" ? true : false;
            else if (mit->first == "verbose ")
                verbose = mit->second == " true" ? true : false;
            else if (mit->first == "verboselvl ")
                verboselvl = std::atoi(mit->second.c_str());
        }
    }
}

} // namespace slam
