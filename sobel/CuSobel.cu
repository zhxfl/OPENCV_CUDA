#include "cuSobel.cuh"
#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include "cuda_texture_types.h"

texture <float,2> texIn;

__global__ void kernel_x_sobel_1(float* d_pOutput, int nRows, int nCols);
__global__ void kernel_x_sobel_2(float* d_pOutput, int nRows, int nCols);

void CuSobel(const cv::Mat &in, cv::Mat &out,
	int xorder, int yorder)
{
	float *pInput  = (float*) in.data;
	float *pOutput = (float*) out.data;
	float *d_pInput, *d_pOutput;

	size_t nRows = in.rows;
	size_t nCols = in.cols;
	size_t size = nRows * nCols;

	cudaMalloc((void**) &d_pInput , sizeof(float) * size);
	cudaMalloc((void**) &d_pOutput, sizeof(float) * size);
	
	dim3 blocksize(32,32);
	dim3 gridsize((nCols + blocksize.x - 1) / blocksize.x,
		(nRows + blocksize.y - 1) / blocksize.y);

	cudaChannelFormatDesc channelDesc = cudaCreateChannelDesc<float>();

	
	cudaBindTexture2D(NULL,texIn, 
		d_pInput, channelDesc, nCols, nRows, 
		sizeof(float) * nCols);

	cudaMemcpy(d_pInput, pInput, sizeof(float) * size, cudaMemcpyHostToDevice);

	kernel_x_sobel_1<<<gridsize, blocksize>>>(d_pOutput, nRows, nCols);
	cudaDeviceSynchronize();

	cudaMemcpy(pOutput, d_pOutput, sizeof(float) * size, cudaMemcpyDeviceToHost);
	cudaUnbindTexture(texIn);
	cudaFree(d_pOutput);
	cudaFree(d_pInput);
}

__global__ void kernel_x_sobel_1(float* pOutput, int nRows, int nCols)
{
	int x = blockIdx.x * blockDim.x + threadIdx.x;
	int y = blockIdx.y * blockDim.y + threadIdx.y;
	int block_id = blockDim.x * threadIdx.y + threadIdx.x;
	/*****************************
	if xorder == 1
	-1 0 1
	-2 0 2
	-1 0 1
	******************************/
	__shared__ float h[1024];
	if(x>= 1 && x<nCols - 1 && y >= 1 && y< nRows - 1)
	{
		float output_value =
			(-  tex2D(texIn,x-1,y-1)) + ( tex2D(texIn,x+1,y-1)) + 
			(-2*tex2D(texIn,x-1,y  )) + (2*tex2D(texIn,x+1,y )) + 
			(-  tex2D(texIn,x-1,y+1)) + ( tex2D(texIn,x+1,y+1));

		pOutput[y * nCols + x]= output_value;
	}
}

__global__ void kernel_x_sobel_2(float* pOutput, int nRows, int nCols)
{
	int x = blockIdx.x * blockDim.x + threadIdx.x;
	int y = blockIdx.y * blockDim.y + threadIdx.y;
	int block_id = blockDim.x * threadIdx.y + threadIdx.x;
	/*****************************
	if xorder == 1
	-1 0 1
	-2 0 2
	-1 0 1
	******************************/
	__shared__ float h[1024];
	if(x>= 1 && x<nCols - 1 && y >= 1 && y< nRows - 1)
	{
		float output_value =
			(-  tex2D(texIn,x-1,y-1)) + ( tex2D(texIn,x+1,y-1)) + 
			(-2*tex2D(texIn,x-1,y  )) + (2*tex2D(texIn,x+1,y )) + 
			(-  tex2D(texIn,x-1,y+1)) + ( tex2D(texIn,x+1,y+1));

		pOutput[y * nCols + x]= output_value;
	}
}