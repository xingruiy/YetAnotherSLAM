#include "LocalOptimizer.h"

LocalOptimizer::LocalOptimizer(int w, int h)
{
}

struct LinearisationFunctor
{
    float *x;
    float *x0;
    int *target;
    int *host;
    float *framePosePrecalc;
    int N;

    __device__ void operator()() const
    {
        int x = threadIdx.x + blockDim.x * blockIdx.x;
        if (x >= N)
            return;
    }
};

void LocalOptimizer::LineariseAll()
{
    LinearisationFunctor functor;
    functor.x = state;
}