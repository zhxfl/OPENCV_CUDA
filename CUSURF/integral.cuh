#ifndef INTEGRAL_H
#define INTEGRAL_H
#include "opencv2/core/gpumat.hpp"
#include "opencv/cv.h"
#include "cuda_runtime.h"
#include "device_launch_parameters.h"
void Integral(const cv::gpu::GpuMat& source, cv::gpu::GpuMat& out);

#endif