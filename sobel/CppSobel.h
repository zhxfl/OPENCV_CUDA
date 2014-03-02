#ifndef __CPP_SOBEL_H__
#define __CPP_SOBEL_H__

#include "opencv2/highgui/highgui.hpp"
#include "opencv/cv.h"
#include "opencv2/contrib/contrib.hpp"

void CppSobel(IplImage* in, IplImage *out,
	int xorder,int yorder,
	int aperture_size CV_DEFAULT(3));

#endif
