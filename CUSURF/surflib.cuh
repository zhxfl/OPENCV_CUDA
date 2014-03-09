#ifndef SURFLIB_H
#define SURFLIB_H

#include "opencv/cv.h"
#include "opencv2/highgui/highgui.hpp"
#include "opencv2/contrib/contrib.hpp"
#include "integral.cuh"

static const int OCTAVES = 5;
static const int INTERVALS = 4;
static const float THRES = 0.0004f;
static const int INIT_SAMPLE = 2;

void surfDetDes(cv::gpu::GpuMat& img,  /* image to find Ipoints in */
	cv::gpu::GpuMat& ipts, /* reference to vector of Ipoints */
	bool upright = false, /* run in rotation invariant mode? */
	int octaves = OCTAVES, /* number of octaves to calculate */
	int intervals = INTERVALS, /* number of intervals per octave */
	int init_sample = INIT_SAMPLE, /* initial sampling step */
	float thres = THRES /* blob response threshold */);


#endif