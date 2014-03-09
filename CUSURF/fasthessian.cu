#include "integral.cuh"
#include <vector>
#include "fasthessian.cuh"
#include "opencv2/contrib/contrib.hpp"
using namespace std;

__global__ void testText(float * r, int width, int height);


__global__ void d_getResponseLayer(
	float* responses,
	uchar* laplacian, 
	int img_width,
	int img_height,
	int width,
	int height,
	int step,
	int b,
	int l,
	int w,
	float inverse_area
	);


__device__ float BoxIntegral(int row, int col, int rows, int cols, int width, int height);

texture<float, 2> m_texImg;

FastHessian::FastHessian(
	cv::gpu::GpuMat& img,
	cv::gpu::GpuMat &ipts, 
	const int octaves,
	const int intervals, 
	const int init_sample, 
	const float thresh)
	:m_ipts(ipts),m_img(img)

{
	this->saveParameters(octaves, intervals, init_sample, thresh);
	this->bindTexture();
}


FastHessian::~FastHessian()
{
	cudaUnbindTexture(m_texImg);
}


void FastHessian::saveParameters(const int octaves, const int intervals, 
	const int init_sample, const float thresh)
{
	// Initialise variables with bounds-checked values
	this->octaves = 
		(octaves > 0 && octaves <= 4 ? octaves : OCTAVES);
	this->intervals = 
		(intervals > 0 && intervals <= 4 ? intervals : INTERVALS);
	this->init_sample = 
		(init_sample > 0 && init_sample <= 6 ? init_sample : INIT_SAMPLE);
	this->thresh = (thresh >= 0 ? thresh : THRES);
}


void FastHessian::getIpoints()
{
	for(size_t i = 0; i < responseMap.size(); i++)
	{
		delete responseMap[i];
	}
	responseMap.clear();

	int w = (m_img.cols / init_sample);
	int h = (m_img.rows / init_sample);
	int s = (init_sample);

	// Calculate approximated determinant of hessian values
	if (octaves >= 1)
	{
		responseMap.push_back(new ResponseLayer(w,   h,   s,   9));
		responseMap.push_back(new ResponseLayer(w,   h,   s,   15));
		responseMap.push_back(new ResponseLayer(w,   h,   s,   21));
		responseMap.push_back(new ResponseLayer(w,   h,   s,   27));
	}

	if (octaves >= 2)
	{
		responseMap.push_back(new ResponseLayer(w/2, h/2, s*2, 39));
		responseMap.push_back(new ResponseLayer(w/2, h/2, s*2, 51));
	}

	if (octaves >= 3)
	{
		responseMap.push_back(new ResponseLayer(w/4, h/4, s*4, 75));
		responseMap.push_back(new ResponseLayer(w/4, h/4, s*4, 99));
	}

	if (octaves >= 4)
	{
		responseMap.push_back(new ResponseLayer(w/8, h/8, s*8, 147));
		responseMap.push_back(new ResponseLayer(w/8, h/8, s*8, 195));
	}

	if (octaves >= 5)
	{
		responseMap.push_back(new ResponseLayer(w/16, h/16, s*16, 291));
		responseMap.push_back(new ResponseLayer(w/16, h/16, s*16, 387));
	}

	for(size_t i = 0; i < responseMap.size(); i++)
	{
		buildResponseLayer(responseMap[i]);
	}

}

void FastHessian::buildResponseLayer(ResponseLayer *rl)
{
	cv::TickMeter tm;
	tm.start();

	float*  responses = rl->responses;         // response storage
	float*d_responses = rl->get_device_response();
	uchar*  laplacian = rl->laplacian;         // laplacian sign storage
	uchar*d_laplacian = rl->get_device_laplacian();

	int step = rl->step;                       // step size for this filter
	int b    = (rl->filter - 1) / 2;              // border for this filter
	int l    = rl->filter / 3;                    // lobe for this filter (filter size / 3)
	int w    = rl->filter;                        // filter size
	float inverse_area = 1.f/(w*w);            // normalisation factor

	int width  = rl->width;
	int height = rl->height;


	dim3 blockSize(width * height > 1024? 1024: width * height);
	dim3 gridSize((width * height + blockSize.x - 1) / blockSize.x );

	d_getResponseLayer<<<gridSize, blockSize>>>(
		d_responses,
		d_laplacian,
		m_img.cols,
		m_img.rows,
		width,
		height,
		step,b,l,w,
		inverse_area);
	cudaDeviceSynchronize();
	rl->download();
	tm.stop();
	printf("CUDA buildResponseLayer time %f ms\n", tm.getTimeMilli());
	/*//////////////////////////////check the ans//////////////////////////
	static int cur = 0;
	printf("%d %d \n", rl->width * rl->height, cur++);
	rl->downloadResponse();
	rl->downloadLaplacian();
	/////////////////////////////////////////////////////////////////////*/
}


__global__ void d_getResponseLayer(
	float* responses,
	uchar* laplacian, 
	int img_width,
	int img_height,
	int width,
	int height,
	int step,
	int b,
	int l,
	int w,
	float inverse_area
	)
{
	int id = blockDim.x * blockIdx.x + threadIdx.x;
	if(id >= width * height)
		return;
	int ar = id / width;
	int ac = id % width;
	int r = ar * step;
	int c = ac * step;
	float Dxx, Dyy, Dxy;
	
	Dxx = BoxIntegral(r - l + 1, c - b,     2*l - 1, w, img_width, img_height)
		- BoxIntegral(r - l + 1, c - l / 2, 2*l - 1, l, img_width, img_height) * 3;

	Dyy = BoxIntegral(r - b,     c - l + 1, w, 2*l - 1, img_width, img_height)
		- BoxIntegral(r - l / 2, c - l + 1, l, 2*l - 1, img_width, img_height) * 3;
	Dxy = 
		+ BoxIntegral(r - l, c + 1, l, l, img_width, img_height)
		+ BoxIntegral(r + 1, c - l, l, l, img_width, img_height)
		- BoxIntegral(r - l, c - l, l, l, img_width, img_height)
		- BoxIntegral(r + 1, c + 1, l, l, img_width, img_height);

	Dxx *= inverse_area;
	Dyy *= inverse_area;
	Dxy *= inverse_area;
	
	responses[id] = (Dxx * Dyy - 0.81f * Dxy * Dxy);
	laplacian[id] = (Dxx + Dyy >= 0 ? 1 : 0);
}

__device__ float BoxIntegral(int row, int col, int rows, int cols, int width, int height)
{
	int r1 = min(row,        height) - 1;
	int c1 = min(col,        width)  - 1;
	int r2 = min(row + rows, height) - 1;
	int c2 = min(col + cols, width)  - 1;
	
	float A(0.0f), B(0.0f), C(0.0f), D(0.0f);

	if (r1 >= 0 && c1 >= 0) A = tex2D(m_texImg, c1, r1);
	if (r1 >= 0 && c2 >= 0) B = tex2D(m_texImg, c2, r1);
	if (r2 >= 0 && c1 >= 0) C = tex2D(m_texImg, c1, r2);
	if (r2 >= 0 && c2 >= 0) D = tex2D(m_texImg, c2, r2);

	return max(0.f, A - B - C + D);
}


void FastHessian::bindTexture()
{
	cudaChannelFormatDesc channelDesc = cudaCreateChannelDesc<float>();

	cudaBindTexture2D(NULL,m_texImg, 
		m_img.data, 
		channelDesc,
		m_img.cols, 
		m_img.rows, 
		m_img.step);


	///////////////////////////test Texture////////////////////////////////*
	
	/*float* pTest;
	cudaMalloc((void**) &pTest, sizeof(float) * m_img.cols * m_img.rows);
	testText<<<m_img.rows, m_img.cols>>>(pTest, m_img.cols, m_img.rows);

	cv::Mat test_1;
	m_img.download(test_1);
	float* ptest_1 = (float*)cv::fastMalloc(sizeof(float) * m_img.cols * m_img.rows);
	cudaMemcpy(ptest_1, pTest, sizeof(float) * m_img.cols * m_img.rows , cudaMemcpyDeviceToHost);

	printf("test Texture\n");
	float* ptr = (float*)test_1.data;
	for(int i = 0; i < 10; i++)
	{
		printf("%.2f ",(float)ptr[i * m_img.step / 4]);
	}puts("");

	for(int i = 0; i < 10; i++)
	{
		printf("%.2f ",(float)ptest_1[i * m_img.cols]);
	}puts("");
	///////////////////////////////////////////////////////////////////////*/
}

/*
__global__ void testText(float* r, int width, int height)
{
	int y = blockIdx.x;
	int x = threadIdx.x;

	if(x >= width)
		return;
	if(y >= height)
		return;

	r[y * width + x] = tex2D(m_texImg, x, y);

}*/