
#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2\imgproc\imgproc.hpp>
#include <iostream>
#include <stdio.h>
#include "HPT.h"


using namespace cv;
using namespace std;

Mat image;
int trackbarSize;
unsigned char* ImageModified;
typedef unsigned char uByte;

//kernal takes in two arrays and size
__global__ void thresholdKernel(unsigned char* data, unsigned char* data2, int size, int thresholdSlider) {


	int j = (blockIdx.x *blockDim.x) + threadIdx.x;

	if (j < size) {
		if (data[j] > thresholdSlider) {
			data2[j] = 255;
		}
		else {
			data2[j] = 0;
		}
	}
}
//threshold change in cpu
void threshold(int threshold, int width, int height, unsigned char* data);
//threshold change in gpu
bool initializeImageGPU(int width, int height, Mat image);
//creates trackbar for image
void on_trackbar(int thresholdSlider, void*);
void BoxFilter(uByte* s, uByte* d, int w, int h, uByte* k, int kW, int kH, uByte* Temp);


int main(int argc, char** argv) {
	if (argc != 2) {
		cout << "Usage: display_image ImageToLoadAndDisplay" << endl;
		return -1;
	}


	image = imread(argv[1], CV_LOAD_IMAGE_COLOR);

	if (!image.data) {
		cout << "Could not open or find image" << endl;
		return -1;
	}

	cvtColor(image, image, COLOR_RGB2GRAY);


	threshold(128, image.rows, image.cols, image.data);


	if (initializeImageGPU(image.rows, image.cols, image)) {
		cout << "We worked with the GPU" << endl;
	}
	else {
		cout << "It failed." << endl;
	}

	namedWindow("Display Window", WINDOW_NORMAL);
	//createTrackbar("Threshold", "Display Window", &threshold_slider, THRESHOLD_SLIDER_MAX, on_tracker(int, void *, Image, unsigned char* data2, size,threshold_slider));
	imshow("Display Window", image);

	waitKey(0);
	return 0;
}

void threshold(int threshold, int width, int height, unsigned char * data)
{
	HighPrecisionTime timeTheModification;
	double currentTime;
	timeTheModification.TimeSinceLastCall();
	for (int i = 0; i < height *width; i++) {
		if (data[i] > threshold) {
			data[i] = 255;
		}
		else {
			data[i] = 0;
		}
	}
	currentTime = timeTheModification.TimeSinceLastCall();
	cout << "CPU: " << currentTime << endl;
}

bool initializeImageGPU(int width, int height, Mat image)
{
	HighPrecisionTime timeTheModification;
	double currentTime;

	bool temp = true;
	unsigned char* ImageOriginal = nullptr;
	ImageModified = nullptr;
	int size = width*height * sizeof(char);
	trackbarSize = size;

	cudaError_t cudaTest;

	cudaTest = cudaSetDevice(0);
	if (cudaTest != cudaSuccess) {
		cout << "Error with device" << endl;
	}
	else {
		//cout << "suscsess" << endl;
	}

	cudaTest = cudaMalloc(&ImageOriginal, size);
	if (cudaTest != cudaSuccess) {
		cout << "cudaMalloc failed!" << endl;
		temp = false;
	}
	else {
		//cout << "suscsess" << endl;
	}

	cudaTest = cudaMalloc(&ImageModified, size);
	if (cudaTest != cudaSuccess) {
		cout << "cudaMalloc2 failed!" << endl;
		temp = false;
	}
	else {
		//cout << "suscsess" << endl;
	}

	cudaTest = cudaDeviceSynchronize();
	if (cudaTest != cudaSuccess) {
		cout << "cudaSync failed!" << endl;
		temp = false;
	}
	else {
		//cout << "suscsess" << endl;
	}

	cudaTest = cudaMemcpy(ImageOriginal, image.data, size, cudaMemcpyHostToDevice);
	if (cudaTest != cudaSuccess) {
		cout << "cudacpy failed!" << endl;
		temp = false;
	}
	else {
		//cout << "suscsess" << endl;
	}

	int blocksNeeded = (size + 1023) / 1024;

	timeTheModification.TimeSinceLastCall();
	thresholdKernel << <blocksNeeded, 1024 >> > (ImageOriginal, ImageModified, size, 128);
	currentTime = timeTheModification.TimeSinceLastCall();
	cout << "GPU: " << currentTime << endl;

	cudaTest = cudaMemcpy(image.data, ImageModified, size, cudaMemcpyDeviceToHost);
	if (cudaTest != cudaSuccess) {
		cout << "cudacpy2 failed!" << endl;
		temp = false;
	}

	int thresholdSlider = 50;
	namedWindow("Display Window", WINDOW_NORMAL);
	createTrackbar("Threshold", "Display Window", &thresholdSlider, 255, on_trackbar, &image);
	on_trackbar(thresholdSlider, 0);
	waitKey(0);

	return temp;
}

void on_trackbar(int thresholdSlider, void*)
{
	HighPrecisionTime T;
	double currentTime;
	int blocksNeeded = (trackbarSize + 1023) / 1024;
	cudaDeviceSynchronize();

	T.TimeSinceLastCall();
	thresholdKernel << < blocksNeeded, 1024 >> > (image.data, ImageModified, (image.rows*image.cols), thresholdSlider);
	currentTime = T.TimeSinceLastCall();
	cout << "CurrentTime: " << currentTime << endl;

	if (cudaMemcpy(image.data, ImageModified, trackbarSize, cudaMemcpyDeviceToHost) != cudaSuccess) {
		cout << "Error copying." << endl;
	}
	imshow("Display Window", image);
}

void BoxFilter(uByte * s, uByte * d, int w, int h, uByte * k, int kW, int kH, uByte * Temp)
{
	//(s.data, d.data,s.cols, s.rows, k,3,3,temp.data)
	//int widthMinus1 = w - 1;
	int size = w*h;
	size_t xIndex = blockIdx.x * blockDim.x + threadIdx.x;
	size_t yIndex = blockIdx.y * blockDim.y + threadIdx.y;

	int xOffset = w / 2;
	int yOffset = h / 2;

	float outputValue;

	// Checking to see if the indexs are within the bounds of the image and not on edges
	if (xIndex < (w - 1) && xIndex >0 && yIndex < h - 1 && yIndex >0)
	{
		int xPixel = (xIndex - w / 2 + xIndex + w) % w;
		int yPixel = (yIndex - h / 2 + yIndex + h) % h;

		for (int i = -xOffset; i <= xOffset; i++) {
			for (int j = -yOffset; j <= yOffset; j++) {
				outputValue+=
			}

		}

	}
}
