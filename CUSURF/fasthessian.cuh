#ifndef FASTHESSIAN_H
#define FASTHESSIAN_H

#include "opencv/cv.h"
#include "surflib.cuh"
#include "responselayer.h"
#include "cuda_texture_types.h"

class FastHessian {
  
public:
	FastHessian(cv::gpu::GpuMat& img, 
		cv::gpu::GpuMat &ipts, 
		const int octaves = OCTAVES, 
		const int intervals = INTERVALS, 
		const int init_sample = INIT_SAMPLE, 
		const float thres = THRES);
	~FastHessian();

	void getIpoints();
private:
	cv::gpu::GpuMat& m_ipts;
	cv::gpu::GpuMat& m_img;

	int octaves;	//! Number of Octaves
	int intervals;//! Number of Intervals per octave
	int init_sample;	//! Initial sampling step for Ipoint detection
	float thresh;//! Threshold value for blob resonses
	std::vector<ResponseLayer *> responseMap;

private:
	void saveParameters(const int octaves, const int intervals, 
		const int init_sample, const float thresh);
	void buildResponseLayer(ResponseLayer *rl);
	void bindTexture();
};


#endif