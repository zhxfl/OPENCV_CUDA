#include "opencv2/highgui/highgui.hpp"
#include "opencv/cv.h"
#include "opencv2/contrib/contrib.hpp"
#include "CppSobel.h"
#include "CuSobel.cuh"
#include <iostream>
#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include "cuda_texture_types.h"

static cv::Mat loadImage(const std::string& name)
{
	cv::Mat image = cv::imread("D://code//OPENCV_CUDA//bin//sobel//release//1.jpg", cv::IMREAD_GRAYSCALE);
	if (image.empty())
	{
		std::cerr << "Can't load image - " << name << std::endl;
		exit(-1);
	}
	return image;
}

void runCvSobel()
{
	cv::Mat _in  = loadImage("1.jpg");
	cv::Mat _out;
	_out.create(_in.size(), CV_MAKETYPE(CV_32F, _in.channels()));
	cv::TickMeter tm;
	tm.start();
	Sobel(_in, _out, CV_32F, 1, 0);
	tm.stop();
	printf("cvSobel  time: %4.4f ms\n", tm.getTimeMilli());
	/*for(int i = 0; i < 10; i++)
	{
		printf("%f ",_out.at<float>(1,i));
	}printf("\n");*/
	//cvNamedWindow("cvSobel",1);
	//imshow("cvSobel",_out);
}

void runCppSobel_1()
{
	cv::Mat inImage = loadImage("1.jpg");
	cv::Mat _out;
	_out.create(inImage.size(), CV_MAKETYPE(CV_32F, inImage.channels()));
	cv::TickMeter tm;
	tm.start();
	CppSobel_1(inImage, _out, 1, 0);
	tm.stop();

	printf("CppSobel_1 time: %4.4f ms\n", tm.getTimeMilli());
	for(int i = 0; i < 10; i++)
	{
		printf("%f ",_out.at<float>(1,i));
	}printf("\n");
	//cvNamedWindow("CppSobel_1",1);
	//imshow("CppSobel_1",_out);
}

void runCppSobel()
{
	cv::Mat inImage = loadImage("1.jpg");
	cv::Mat _out;
	_out.create(inImage.size(), CV_MAKETYPE(CV_32F, inImage.channels()));
	cv::TickMeter tm;
	tm.start();
	CppSobel(inImage, _out, 1, 0);
	tm.stop();

	printf("CppSobel time: %4.4f ms\n", tm.getTimeMilli());
	/*for(int i = 0; i < 10; i++)
	{
		printf("%f ",_out.at<float>(1,i));
	}printf("\n");*/
	//cvNamedWindow("CppSobel",1);
	//imshow("CppSobel",_out);
}

void runCuSobel()
{
	cv::Mat inImage = loadImage("1.jpg");
	cv::Mat _in,_out;
	inImage.convertTo(_in,CV_32F);
	_out.create(inImage.size(), CV_MAKETYPE(CV_32F, inImage.channels()));

	cv::TickMeter tm;
	tm.start();
	CuSobel(_in, _out, 1, 0);
	tm.stop();

	printf("CuSobel time: %4.4f ms\n", tm.getTimeMilli());
	for(int i = 0; i < 10; i++)
	{
		printf("%f ",_out.at<float>(1,i));
	}printf("\n");
	//cvNamedWindow("CuSobel",1);
	//imshow("CuSobel",_out);
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

int main()
{
	float* tmp;
	checkCudaAndWarm();
	runCppSobel();
	runCvSobel();
	runCppSobel_1();
	runCuSobel();
	cvWaitKey(0);
	
	return 0;
}