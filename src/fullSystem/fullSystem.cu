#include "fullSystem/fullSystem.h"

FullSystem::FullSystem(const char *configFile)
{
}

FullSystem::FullSystem(int w, int h, Mat33d K, int numLvl, bool view)
    : enableMapViewer(view), currentState(-1)
{
    if (enableMapViewer)
    {
        mapViewer = std::make_shared<MapViewer>(640, 480);
    }

    coarseTracker = std::make_shared<DenseTracker>(w, h, K, numLvl);
}

void FullSystem::processFrame(Mat rawImage, Mat rawDepth)
{
}

bool FullSystem::trackCurrentFrame()
{
    return true;
}

bool FullSystem::needNewKF()
{
    return false;
}

void FullSystem::createNewKF()
{
}

void FullSystem::resetSystem()
{
}