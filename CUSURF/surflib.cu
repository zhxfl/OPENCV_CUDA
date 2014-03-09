#include "surflib.cuh"
#include "fasthessian.cuh"
void surfDetDes(cv::gpu::GpuMat& img,  /* image to find Ipoints in */
	cv::gpu::GpuMat& ipts, /* reference to vector of Ipoints */
	bool upright, /* run in rotation invariant mode? */
	int octaves, /* number of octaves to calculate */
	int intervals, /* number of intervals per octave */
	int init_sample, /* initial sampling step */
	float thres /* blob response threshold */)

{
	cv::gpu::GpuMat out;
	Integral(img, out);

	FastHessian fh(out, ipts, octaves, intervals, init_sample, thres);
	fh.getIpoints();
}