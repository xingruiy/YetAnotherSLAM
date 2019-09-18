#ifndef FUSION_MAPPING_DATA_TYPES_H
#define FUSION_MAPPING_DATA_TYPES_H

#include "macros.h"
#include "matrix_type.h"
#include "vector_type.h"

struct FUSION_EXPORT Voxel
{
    FUSION_HOST_AND_DEVICE inline Voxel();
    FUSION_HOST_AND_DEVICE inline Voxel(float sdf, float weight, Vector3c rgb);
    FUSION_HOST_AND_DEVICE inline float getSDF() const;
    FUSION_HOST_AND_DEVICE inline void setSDF(float val);
    FUSION_HOST_AND_DEVICE inline float getWeight() const;
    FUSION_HOST_AND_DEVICE inline void setWeight(float val);

    short sdf;
    float weight;
    Vector3c rgb;
};

FUSION_HOST_AND_DEVICE inline Voxel::Voxel()
    : sdf(0), weight(0), rgb(0)
{
}

FUSION_HOST_AND_DEVICE inline Voxel::Voxel(float sdf, float weight, Vector3c rgb)
    : weight(weight), rgb(rgb)
{
    setSDF(sdf);
}

FUSION_HOST_AND_DEVICE inline float unpackFloat(short val)
{
    return val / (float)32767;
}

FUSION_HOST_AND_DEVICE inline short packFloat(float val)
{
    return (short)(val * 32767);
}

FUSION_HOST_AND_DEVICE inline float Voxel::getSDF() const
{
    return unpackFloat(sdf);
}

FUSION_HOST_AND_DEVICE inline void Voxel::setSDF(float val)
{
    sdf = packFloat(val);
}

FUSION_HOST_AND_DEVICE inline float Voxel::getWeight() const
{
    return weight;
}

FUSION_HOST_AND_DEVICE inline void Voxel::setWeight(float val)
{
    weight = val;
    if (weight > 255)
        weight = 255;
}

#endif