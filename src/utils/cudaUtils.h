#include <cuda_runtime_api.h>

static dim3 getGridConfiguration2D(dim3 &block, int w, int h)
{
    dim3 grid;
    grid.x = (w + block.x - 1) / block.x;
    grid.y = (h + block.y - 1) / block.y;
    return grid;
}

template <class T>
__global__ void callDeviceFunctor(const T functor)
{
    functor();
}

template <typename T>
__device__ __forceinline__ T interpolateBiLinear(
    const cv::cuda::PtrStep<T> &map,
    const float &x, const float &y)
{
    int u = static_cast<int>(std::floor(x));
    int v = static_cast<int>(std::floor(y));
    float cox = x - u;
    float coy = y - v;
    return (map.ptr(v)[u] * (1 - cox) + map.ptr(v)[u + 1] * cox) * (1 - coy) +
           (map.ptr(v + 1)[u] * (1 - cox) + map.ptr(v + 1)[u + 1] * cox) * coy;
}