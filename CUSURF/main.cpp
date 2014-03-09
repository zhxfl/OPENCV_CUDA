#include "surflib.cuh"
#include "opencv2/contrib/contrib.hpp"
#include <ctime>
#include <iostream>
#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include "cuda_texture_types.h"
#include "opencv2/highgui/highgui.hpp"
#include "opencv/cv.h"
#include "opencv2/contrib/contrib.hpp"
#include "opencv2/core/types_c.h"
#include "opencv2/core/gpumat.hpp"
#include "opencv2/core/operations.hpp"
int mainImage(void)
{
	//read the image 
	cv::Mat matReadin = cv::imread("imgs/sf.jpg", cv::IMREAD_GRAYSCALE);
	cv::TickMeter tm;
	tm.start();
	cv::gpu::GpuMat gMat(matReadin);
	cv::gpu::GpuMat ipts;
	ipts.create(gMat.size(), gMat.channels());
	surfDetDes(gMat, ipts, false, 5, 4, 2, 0.0004f); 
	tm.stop();
	printf("curtime %fms",tm.getTimeMilli());
	return 0;
}


void checkCudaAndWarm()
{
	int deviceCount;
	cudaError_t error_id = cudaGetDeviceCount(&deviceCount);

	if (error_id != cudaSuccess)
	{
		printf("cudaGetDeviceCount returned %d\n-> %s\n", (int)error_id, cudaGetErrorString(error_id));
		printf("Result = FAIL\n");
		exit(EXIT_FAILURE);
	}

	char * pWorm;
	cudaMalloc((void**) &pWorm , 1);
	cudaFree(pWorm);
}


int main(void) 
{
	checkCudaAndWarm();
	mainImage();
	return true;
}
