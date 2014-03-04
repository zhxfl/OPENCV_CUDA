#include "opencv2/highgui/highgui.hpp"
#include "opencv/cv.h"
#include "opencv2/contrib/contrib.hpp"
#include "CppSobel.h"
#include <iostream>

static cv::Mat loadImage(const std::string& name)
{
	cv::Mat image = cv::imread(name, cv::IMREAD_GRAYSCALE);
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
	Sobel(_in, _out, CV_32F, 0, 1);
	tm.stop();
	printf("cvSobel  time: %4.4f ms\n", tm.getTimeMilli());
	/*for(int i = 0; i < 10; i++)
	{
		printf("%f ",_out.at<float>(1,i));
	}printf("\n");*/
	cvNamedWindow("cvSobel",1);
	imshow("cvSobel",_out);
}

void runCppSobel_1()
{
	cv::Mat inImage = loadImage("1.jpg");
	cv::Mat _out;
	_out.create(inImage.size(), CV_MAKETYPE(CV_32F, inImage.channels()));
	cv::TickMeter tm;
	tm.start();
	CppSobel_1(inImage, _out, 0, 1);
	tm.stop();

	printf("CppSobel_1 time: %4.4f ms\n", tm.getTimeMilli());
	/*for(int i = 0; i < 10; i++)
	{
		printf("%f ",_out.at<float>(1,i));
	}printf("\n");*/
	cvNamedWindow("CppSobel_1",1);
	imshow("CppSobel_1",_out);
}

void runCppSobel()
{
	cv::Mat inImage = loadImage("1.jpg");
	cv::Mat _out;
	_out.create(inImage.size(), CV_MAKETYPE(CV_32F, inImage.channels()));
	cv::TickMeter tm;
	tm.start();
	CppSobel(inImage, _out, 0, 1);
	tm.stop();

	printf("CppSobel time: %4.4f ms\n", tm.getTimeMilli());
	/*for(int i = 0; i < 10; i++)
	{
		printf("%f ",_out.at<float>(1,i));
	}printf("\n");*/
	cvNamedWindow("CppSobel",1);
	imshow("CppSobel",_out);
}


int main()
{
	runCppSobel();

	runCvSobel();
	
	runCppSobel_1();

	cvWaitKey(0);
	return 0;
}