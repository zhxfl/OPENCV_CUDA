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
	Sobel(_in, _out, CV_32F, 1,0);
	tm.stop();
	printf("cvSobel  time: %4.4f ms\n", tm.getTimeMilli());
	for(int i = 0; i < 10; i++)
	{
		printf("%f ",_out.at<float>(1,i));
	}printf("\n");
	cvNamedWindow("cvSobel",1);
	imshow("cvSobel",_out);
}

void runCppSobel()
{
	cv::Mat inImage = loadImage("1.jpg");
	cv::Mat _out;
	_out.create(inImage.size(), CV_MAKETYPE(CV_32F, inImage.channels()));
	cv::TickMeter tm;
	tm.start();
	CppSobel(inImage, _out, 1,0);
	tm.stop();

	printf("cvSobel  time: %4.4f ms\n", tm.getTimeMilli());
	for(int i = 0; i < 10; i++)
	{
		printf("%f ",_out.at<float>(1,i));
	}printf("\n");
	cvNamedWindow("cvSobel_",1);
	imshow("cvSobel_",_out);
}


int main()
{
	runCvSobel();
	runCppSobel();
	cvWaitKey(0);
	return 0;
}