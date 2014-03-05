#include "CppSobel.h"

/******************************
使用Mat来优化内存
******************************/

void CppSobel(const cv::Mat& in, cv::Mat& out,
	int xorder, int yorder)
{
	int nRows = in.rows;
	int nCols = in.cols;
	if(yorder)
	{
		for(int i = 1; i < nRows - 1; i++)
		{
			for(int j = 1; j < nCols - 1; j++)
			{
				float t = 0.0; 
				t -= in.at<char>(i - 1, j - 1);
				t -= 2 * in.at<char>(i - 1, j);
				t -= in.at<char>(i - 1, j + 1);

				t += in.at<char>(i + 1, j - 1);
				t += 2 * in.at<char>(i + 1, j);
				t += in.at<char>(i + 1, j + 1);
				out.at<float>(i,j) = t;
			}
		}
	}

	if(xorder)
	{
		for(int i = 1; i < nRows - 1; i++)
		{
			for(int j = 1; j < nCols - 1; j++)
			{
				float t = 0.0; 
				t -= in.at<char>(i - 1, j - 1);
				t -= 2 * in.at<char>(i    , j - 1);
				t -= in.at<char>(i + 1, j - 1);

				t += in.at<char>(i - 1, j + 1);
				t += 2 * in.at<char>(i, j + 1);
				t += in.at<char>(i + 1, j + 1);
				out.at<float>(i,j) = t;
			}
		}
	}
	return ;
}


void CppSobel_1(const cv::Mat& in, cv::Mat& out,
	int xorder, int yorder)
{
	int nRows = in.rows;
	int nCols = in.cols;

	float ptrVec[3] = {1.0f,2.0f,1.0f};
	if(yorder)
	{
		const char* ptrIn;
		float* ptrOut;
		memset(out.data, 0, sizeof(float) * nRows * nCols);

		ptrIn = in.ptr<char>(0);
		ptrOut = out.ptr<float>(1);
		for(int j = 1; j < nCols - 1; j++)
		{
			ptrOut[j] -= ptrIn[j - 1] + 2.0f * ptrIn[j] + ptrIn[j + 1];
		}

		for(int i = 1; i < nRows - 1; i++)
		{
			ptrIn = in.ptr<char>(i);
			float* ptrOut1 = out.ptr<float>(i - 1);
			float* ptrOut2 = out.ptr<float>(i + 1);
			for(int j = 1; j < nCols - 1; j++)
			{
				float t = ptrIn[j - 1] + 2.0f * ptrIn[j] + ptrIn[j + 1];
				ptrOut1[j] += t;
				ptrOut2[j] -= t;
			}
		}

		ptrIn = in.ptr<char>(nRows - 1);
		ptrOut = out.ptr<float>(nRows - 2);
		for(int j = 1; j < nCols - 1; j++)
		{
			ptrOut[j] += ptrIn[j - 1] + 2.0f * ptrIn[j] + ptrIn[j + 1];
		}
	}

	if(xorder)
	{
		size_t rowSize = sizeof(float) * nCols;
		float* ptrRow = (float*) cv::fastMalloc(rowSize);
		memset(out.data, 0, sizeof(float) * nRows * nCols);
		for(int i = 1; i < nRows - 1; i++)
		{
			const char* ptrIn1 = in.ptr<char>(i - 1);
			const char* ptrIn2 = in.ptr<char>(i);
			const char* ptrIn3 = in.ptr<char>(i + 1);
			for(int f = 0; f < nCols; f++)
			{
				ptrRow[f] = (ptrIn1[f] + 2 * ptrIn2[f] + ptrIn3[f]);
			}
			
			float* ptrOut = out.ptr<float>(i);
			ptrOut[1] -= ptrRow[0];
			for(int j = 1; j < nCols - 1; j++)
			{
				ptrOut[j - 1] += ptrRow[j];
				ptrOut[j + 1] -= ptrRow[j];
			}
			ptrOut[nCols -2 ] += ptrRow[nCols - 1];
		}

		cv::fastFree(ptrRow);
	}
	return ;
}