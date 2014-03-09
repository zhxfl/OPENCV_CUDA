/*********************************************************** 
*  --- OpenSURF ---                                       *
*  This library is distributed under the GNU GPL. Please   *
*  use the contact form at http://www.chrisevansdev.com    *
*  for more information.                                   *
*                                                          *
*  C. Evans, Research Into Robust Visual Features,         *
*  MSc University of Bristol, 2008.                        *
*                                                          *
************************************************************/

#include <memory.h>
#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include "cuda_texture_types.h"
//#define RL_DEBUG  // un-comment to test response layer

class ResponseLayer
{
public:

	int width, height, step, filter;
	float* responses;
	float* d_response;

	uchar* laplacian;
	uchar* d_laplacian;

	ResponseLayer(int width, int height, int step, int filter):
	d_response(NULL),
		d_laplacian(NULL)
	{
		assert(width > 0 && height > 0);

		this->width = width;
		this->height = height;
		this->step = step;
		this->filter = filter;

		responses = new float[width*height];
		laplacian = new uchar[width*height];

		memset(responses,0,sizeof(float)*width*height);
		memset(laplacian,0,sizeof(unsigned char)*width*height);
	}


	float* get_device_response()
	{
		if(!d_response)
		{
			cudaMalloc((void**)&d_response, width * height * sizeof(float));
			cudaMemcpy(d_response, responses,  width * height * sizeof(float), cudaMemcpyHostToDevice);
		}
		return d_response;
	}
	uchar* get_device_laplacian()
	{
		if(!d_laplacian)
		{
			cudaMalloc((void**)&d_laplacian, width * height * sizeof(uchar));
			cudaMemcpy(d_laplacian, laplacian,  width * height * sizeof(uchar), cudaMemcpyHostToDevice);
		}
		return d_laplacian;
	}

	void downloadResponse()
	{
		cudaMemcpy(responses, d_response,  width * height * sizeof(float), cudaMemcpyDeviceToHost);
		printf("download responses\n");
		for(int i = 0; i < 10; i++)
		{
			printf("%.4f ",(float)responses[i * width + i]);
		}
		puts("");
	}
	void downloadLaplacian()
	{
		cudaMemcpy(laplacian, d_laplacian,  width * height * sizeof(uchar), cudaMemcpyDeviceToHost);
		printf("download laplacian\n");
		for(int i = 0; i < 10; i++)
		{
			printf("%d ",(uchar)laplacian[i * width + i]);
		}puts("");
	}

	void download()
	{
		cudaMemcpy(responses, d_response ,  width * height * sizeof(float), cudaMemcpyDeviceToHost);
		cudaMemcpy(laplacian, d_laplacian,  width * height * sizeof(uchar), cudaMemcpyDeviceToHost);
	}

	~ResponseLayer()
	{
		if(responses)
			delete [] responses;
		if(laplacian) 
			delete [] laplacian;
		if(d_response) 
			cudaFree(d_response);
		if(d_laplacian)
			cudaFree(d_laplacian);
	}

	inline unsigned char getLaplacian(unsigned int row, unsigned int column)
	{
		return laplacian[row * width + column];
	}

	inline unsigned char getLaplacian(unsigned int row, unsigned int column, ResponseLayer *src)
	{
		int scale = this->width / src->width;

#ifdef RL_DEBUG
		assert(src->getCoords(row, column) == this->getCoords(scale * row, scale * column));
#endif

		return laplacian[(scale * row) * width + (scale * column)];
	}

	inline float getResponse(unsigned int row, unsigned int column)
	{
		return responses[row * width + column];
	}

	inline float getResponse(unsigned int row, unsigned int column, ResponseLayer *src)
	{
		int scale = this->width / src->width;

#ifdef RL_DEBUG
		assert(src->getCoords(row, column) == this->getCoords(scale * row, scale * column));
#endif

		return responses[(scale * row) * width + (scale * column)];
	}

#ifdef RL_DEBUG
	std::vector<std::pair<int, int>> coords;

	inline std::pair<int,int> getCoords(unsigned int row, unsigned int column)
	{
		return coords[row * width + column];
	}

	inline std::pair<int,int> getCoords(unsigned int row, unsigned int column, ResponseLayer *src)
	{
		int scale = this->width / src->width;
		return coords[(scale * row) * width + (scale * column)];
	}
#endif
};
