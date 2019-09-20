#include "globalMapper/sophusEigenHack.h"

class CeresSolver
{
public:
    CeresSolver();
    bool solveBundle(
        double *pointBlock,
        size_t numPoints,
        double *cameraBlock,
        size_t numCameras,
        double *KBlock);
};