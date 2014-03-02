#include "opencv2/highgui/highgui.hpp"
#include "opencv/cv.h"



int FastSobel(unsigned char *in, int width, int height, int widthStep,

	unsigned char *edg, unsigned char *ang)

{

	int i,j;



	unsigned char *inPtr = NULL;

	unsigned char *inPtr1 = NULL;

	unsigned char *inPtr2 = NULL;

	unsigned char *inPtr3 = NULL;

	unsigned char *inPtr4 = NULL;

	unsigned char *inPtr5 = NULL;

	unsigned char *inPtr6 = NULL;



	int *pEdgeX = (int *)calloc(width*height, sizeof(int));

	int *pEdgeY = (int *)calloc(width*height, sizeof(int));

	int *pEdgeXPtr = NULL;

	int *pEdgeYPtr = NULL;



	unsigned char *angPtr = NULL;

	unsigned char *edgPtr = NULL;



	// this is heuristic, and it should add receptive area

	int widH = 1;

	int widV = 1;

	widV *= widthStep;



	for (i=1;i<height-1;i++)

	{

		pEdgeXPtr = pEdgeX + i*width + 1;

		inPtr = in + i * widthStep + 1;

		inPtr1 = inPtr + widH - widV;

		inPtr2 = inPtr + widH;

		inPtr3 = inPtr + widH + widV;

		inPtr4 = inPtr - widH - widV;

		inPtr5 = inPtr - widH;

		inPtr6 = inPtr - widH + widV;

		for (j=1;j<width-1;j++, pEdgeXPtr++)

		{

			*pEdgeXPtr

				= (*inPtr1++ * 1 + (int)*inPtr2++ * 2 + *inPtr3++ * 1)

				- (*inPtr4++ * 1 + (int)*inPtr5++ * 2 + *inPtr6++ * 1);

		}

	}



	for (i=1;i<height-1;i++)

	{

		pEdgeYPtr = pEdgeY + i*width + 1;

		inPtr = in + i * widthStep + 1;

		inPtr1 = inPtr + widV - widH;

		inPtr2 = inPtr + widV;

		inPtr3 = inPtr + widV + widH;

		inPtr4 = inPtr - widV - widH;

		inPtr5 = inPtr - widV;

		inPtr6 = inPtr - widV + widH;

		for (j=1;j<width-1;j++, pEdgeYPtr++)

		{

			*pEdgeYPtr

				= (*inPtr1++ * 1 + (int)*inPtr2++ * 2 + *inPtr3++ * 1)

				- (*inPtr4++ * 1 + (int)*inPtr5++ * 2 + *inPtr6++ * 1);

		}

	}



	for (i=1;i<height-1;i++)

	{

		pEdgeXPtr = pEdgeX + i*width + 1;

		pEdgeYPtr = pEdgeY + i*width + 1;

		angPtr = ang + i * widthStep + 1;

		edgPtr = edg + i * widthStep + 1;

		for (j=1; j<width-1; j++, pEdgeYPtr++, pEdgeXPtr++, angPtr++, edgPtr++)

		{

			*angPtr = atan2((double)*pEdgeYPtr,(double)*pEdgeXPtr)*180/3.141592654;

			*edgPtr = std::min(255.0f,sqrt((float)*pEdgeXPtr**pEdgeXPtr + (float)*pEdgeYPtr**pEdgeYPtr)/2.0f);

		}

	}



	free(pEdgeY); pEdgeY = NULL;

	free(pEdgeX); pEdgeX = NULL;



	return 0;

}

int main(int argc, char* argv[])

{

	IplImage * img = cvLoadImage("1.jpg",1);



	IplImage *grayImage = cvCreateImage(cvSize(img->width, img->height), 8, 1);

	IplImage *gradientImage = cvCreateImage(cvSize(img->width, img->height), 8, 1);

	IplImage *anglImage = cvCreateImage(cvSize(img->width, img->height), 8, 1);



	cvCvtColor(img, grayImage, CV_BGR2GRAY);  // color to gray

	double t = cvGetTickCount();



	FastSobel((unsigned char*)grayImage->imageData, grayImage->width, grayImage->height, grayImage->widthStep,

		(unsigned char*)gradientImage->imageData, (unsigned char*)anglImage->imageData);



	t = (cvGetTickCount()-t)/1000000;

	printf("time: %4.4f\n", t);



	cvNamedWindow("grayImage",1);

	cvShowImage("grayImage",grayImage);



	cvNamedWindow("gradientImage",1);

	cvShowImage("gradientImage",gradientImage);



	cvWaitKey(0);

	cvReleaseImage(&img);

	cvReleaseImage(&grayImage);

	cvReleaseImage(&gradientImage);

	cvReleaseImage(&anglImage);

	return 0;

}