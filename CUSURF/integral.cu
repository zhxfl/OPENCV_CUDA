#include "integral.cuh"
#include "opencv2/contrib/contrib.hpp"
#include "opencv2/core/gpumat.hpp"
#include "cuda_runtime.h"
#include "device_launch_parameters.h"

__global__ void scanCols(uchar* in, float* out, int width, int height, int in_step, int out_step)
{
	int id = blockIdx.x * blockDim.x + threadIdx.x;

	if(id >= width)
		return;

	const float scale = 1.0f / 255.0f;

	uchar* ptr_in  = in  + id;
	float* ptr_out = out + id;

	float pre_out = *ptr_in;
	pre_out = scale * pre_out;
	*ptr_out = pre_out;
	for(int i = 1; i < height; i++)
	{
		ptr_out += out_step;
		ptr_in  += in_step;
		pre_out = pre_out + scale * (*ptr_in);
		*ptr_out = pre_out;
	}
}

__global__ void scanRows(float* out, int width, int height, int step)
{
	int id = blockIdx.x * blockDim.x + threadIdx.x;
	if(id >= height)
		return;
	float* ptr_out = out + id * step;
	float pre_out = *ptr_out;
	for(int i = 1; i < width; i++)
	{
		ptr_out += 1;
		pre_out = pre_out + (*ptr_out);
		*ptr_out = pre_out;
	}
}


void Integral(const cv::gpu::GpuMat& source, cv::gpu::GpuMat& out)
{
	cv::TickMeter tm;
	tm.start();
	out.create(source.size(), CV_32F);
	dim3 blocksize(256);
	dim3 gridsize((source.cols + blocksize.x - 1) / blocksize.x);

	scanCols <<< gridsize, blocksize >>> ((uchar*) source.data, (float*)out.data,
		source.cols, source.rows, source.step, out.step / 4);
	scanRows <<< gridsize, blocksize >>> ((float*)out.data, out.cols, out.rows, out.step / 4);
	cudaDeviceSynchronize();

	tm.stop();
	printf("CUDA integral %fms\n",tm.getTimeMilli());
	/*//////////////////////////////////check integral////////////////////////////////
	cv::Mat test;
	out.download(test);
	printf("check integral\n");
	float* ptr = (float*)test.data;
	for(int i = 0; i < 10; i++)
	{
		printf("%f ",ptr[i * 2]);
	}puts("");
	/////////////////////////////////////////////////////////////////////////////////*/
}


