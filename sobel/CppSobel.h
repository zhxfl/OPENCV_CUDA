#ifndef __CPP_SOBEL_H__
#define __CPP_SOBEL_H__

#include "opencv2/highgui/highgui.hpp"
#include "opencv/cv.h"
#include "opencv2/contrib/contrib.hpp"

void CppSobel(const cv::Mat &in, cv::Mat &out,
	int xorder, int yorder);


#endif