#ifndef __CPP_SOBEL_H__
#define __CPP_SOBEL_H__

#include "opencv2/highgui/highgui.hpp"
#include "opencv/cv.h"
#include "opencv2/contrib/contrib.hpp"

/*****************************
if xorder == 1
-1 0 1
-2 0 2
-1 0 1

if yorder == 1
-1 -2 -1
 0  0  0
 1  2  1

if xorder == 1 && yorder == 1
*****************************/

void CppSobel(const cv::Mat &in, cv::Mat &out,
	int xorder, int yorder);
void CppSobel_1(const cv::Mat &in, cv::Mat &out,
	int xorder, int yorder);


#endif