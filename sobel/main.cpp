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


void runCppSobel()
{
	IplImage *pSrc  = cvLoadImage("1.png",1);
	IplImage *pIn   = cvCreateImage(cvSize(pSrc->width, pSrc->height), IPL_DEPTH_8U, 1);
	IplImage *pOut  = cvCreateImage(cvSize(pSrc->width, pSrc->height), IPL_DEPTH_8U, 1);
	cvCvtColor(pSrc, pIn, CV_BGR2GRAY);  // color to gray
	cv::TickMeter tm;
	tm.start();
	CppSobel(pIn,pOut,1,1);
	tm.stop();
	printf("CppSobel time: %4.4f ms\n", tm.getTimeMilli());
	cvNamedWindow("CppSobel",1);
	cvShowImage("CppSobel",pOut);
}


void runCvSobel()
{
	cv::Mat _in  = loadImage("1.png");
	cv::Mat _out;
	_out.create(_in.size(), CV_MAKETYPE(CV_32F, _in.channels()));
	cv::TickMeter tm;
	tm.start();
	Sobel(_in, _out, CV_32F, 1, 1);
	tm.stop();
	printf("cvSobel  time: %4.4f ms\n", tm.getTimeMilli());
	cvNamedWindow("cvSobel",1);
	imshow("cvSobel",_out);
}

int main()
{
	runCppSobel();
	runCvSobel();
	cvWaitKey(0);
	return 0;

}