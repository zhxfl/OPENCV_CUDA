#include "CppSobel.h"

/******************************
使用Mat来优化内存

******************************/

void CppSobel(const cv::Mat& in, cv::Mat& out,
	int xorder, int yorder)
{
	int nRows = in.rows;
	int nCols = in.cols;

	int i,j;

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
	
		/*float *oneRow = (float*) cv::fastMalloc(sizeof(float) * nCols);
		memset(oneRow,0,sizeof(sizeof(float) * nCols));
		for(i = 0; i < 2; i++)
		{
			inPtr = in.ptr<uchar>(i);
			for(j = 0; j < nCols; j++,inPtr++)
			{
				oneRow[j] += *inPtr;
			}
		}
		for(i = 2; i < nRows; i++)
		{
			outPtr = (float*)(out.data) + i * (out.step.p[0] >> 2);
			inPtr = in.ptr<uchar>(i - 1);
			oneRowPtr = oneRow;
			*(outPtr + 1) = *inPtr + *oneRowPtr;
			printf("%f ",*(outPtr+1));
			for(j = 1; j < nCols - 1; j++)
			{
				outPtr++;
				inPtr++;
				oneRowPtr++;
				*(outPtr - 1) -= *oneRowPtr + *inPtr;
				*(outPtr + 1) += *oneRowPtr + *inPtr;
			}
			outPtr++;
			inPtr++;
			oneRowPtr++;
			*(outPtr - 1) -= *oneRowPtr + *inPtr;

			inPtr = in.ptr<uchar>(i - 2);
			for(j = 0; j < nCols; j++,inPtr++)
			{
				oneRow[j] -= (float)*inPtr;
			}

			inPtr = in.ptr<uchar>(i);
			for(j = 0; j < nCols; j++,inPtr++)
			{
				oneRow[j] += (float)*inPtr;
			}

		}
		*/

	return ;
}