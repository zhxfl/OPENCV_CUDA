/*
referent: 
http://stackoverflow.com/questions/14358916/applying-sobel-edge-detection-with-cuda-and-opencv-on-a-grayscale-jpg-image

*/
#ifndef __CU_SOBEL_H__
#define __CU_SOBEL_H__
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

void CuSobel(const cv::Mat &in, cv::Mat &out,
	int xorder, int yorder);





#endif