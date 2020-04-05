#ifndef CUDA_DEVICE_FUNCS_H
#define CUDA_DEVICE_FUNCS_H

#include <Eigen/Core>

void MakeImageVec3(float *img_src, Eigen::Vector3f *img, int w, int h);
void MakeDepthVec3(float *depth_src, Eigen::Vector3f *depth, int w, int h);
void PyraDownImage(Eigen::Vector3f *img_src, Eigen::Vector3f *img_dst, int w, int h);
void MakeImageGradients(Eigen::Vector3f *img, float *grad, int w, int h);
void PointSelection(Eigen::Vector3f *img, float *grad2, int w, int h);

#endif