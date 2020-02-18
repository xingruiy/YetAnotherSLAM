#include "LoopFinder.h"

namespace SLAM
{

LoopFinder::LoopFinder()
{
}

void LoopFinder::Run()
{
    std::cout << "LoopFinder Thread Started." << std::endl;

    while (!g_bSystemKilled)
    {
        if (CheckNewKeyFrames())
        {
            if (DetectLoop())
            {
            }
        }
    }

    std::cout << "LoopFinder Thread Killed." << std::endl;
}

bool LoopFinder::CheckNewKeyFrames()
{
}

bool LoopFinder::DetectLoop()
{
}

} // namespace SLAM