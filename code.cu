#include "Image.h"
#include "PPM.h"
#include <iostream>
#include <cstdlib>
#include <time.h>
#include <cuda_runtime.h>
#include <math.h>
using namespace std;



#define TILE_WIDTH 16
#define filcols 3
#define filrows 3
#define w (TILE_WIDTH + filcols -1)


__global__ void Convolution(float * InputImage, const float *__restrict__ filtro,
		float* new_img, int channels, int width, int height){

	__shared__ float BlockS[w][w];  						//block of image in shared memory


	// allocation in shared memory of image blocks
	int radio = filrows/2;
 	for (int k = 0; k < channels; k++) {
 		int dest = threadIdx.y * TILE_WIDTH + threadIdx.x;
 		int RY = dest/w;    						//row of shared memory
 		int CX = dest%w;						//col of shared memory
 		int srcY = blockIdx.y *TILE_WIDTH + RY - radio; 		//fetch the data from input image
 		int srcX = blockIdx.x *TILE_WIDTH + RX- radio;	
 		int src = (srcY *width +srcX) * channels + k;   		// input image
 		if(srcY>= 0 && srcY < height && srcX>=0 && srcX < width)
 			BlockS[RY][RX] = InputImage[src];  			// copy element of image in shared memory
 		else
 			BlockS[RY][RX] = 0;



 		dest = threadIdx.y * TILE_WIDTH+ threadIdx.x + TILE_WIDTH * TILE_WIDTH;
 		RY = dest/w;
		RX = dest%w;
		srcY = blockIdx.y *TILE_WIDTH + RY - radio;
		srcX = blockIdx.x *TILE_WIDTH + RX - radio;
		src = (srcY *width +srcX) * channels + k;
		if(RY < w){
			if(srcY>= 0 && srcY < height && srcX>=0 && srcX < width)
				BlockS[RY][RX] = InputImage[src];
			else
				BlockS[RY][RX] = 0;
		}

 		__syncthreads();


 		//compute filter and image convolution
		
 		float accum = 0;
 		int y, x;

 		for (y= 0; y < filcols; y++)
 			for(x = 0; x<filrows; x++)
 				accum += BlockS[threadIdx.y + y][threadIdx.x + x] *filtro[y * filcols + x];

 		y = blockIdx.y * TILE_WIDTH + threadIdx.y;
 		x = blockIdx.x * TILE_WIDTH + threadIdx.x;
 		if(y < height && x < width)
 			new_img[(y * width + x) * channels + k] = accum;
 		__syncthreads();
 	}

}



int main(int argc, char **argv){

	char *iter = argv[1];
  	char *imgpath = argv[2];
	char *filterpath = argv[3];
	char *imgrespath = argv[4];

	//1. Set memory for variables in host and device
	int imgCh;
	int imgH;
	int imgW;
	Image_t* inputImage;
	Image_t* outputImage;
	float* hostInputImage;
	float* hostOutputImage;
	float* deviceInputImage;
	float* deviceOutputImage;
	float* devicefilter;
	float filter[filrows * filcols];
	
	//time variables
	cudaEvent_t start, stop;
	float t;

	if(imgpath == "lena"){
		inputImage = PPM_import("lena.ppm");
	} else if (imgpath == "buildings"){
		inputImage = PPM_import("edificios1.ppm");
	} else if (imgpath == "landscape"){
		inputImage = PPM_import("paisaje1.ppm");
	}

	imgW = Image_getWidth(inputImage);
	imgH = Image_getHeight(inputImage);
	imgCh = Image_getChannels(inputImage);

	outputImage = Image_new(imgW, imgH, imgCh);

	hostInputImage = Image_getData(inputImage);
	hostOutputImage = Image_getData(outputImage);

	
	//2. Copy data from Host to Device
	cudaMalloc((void **) &deviceInputImage, imgW * imgH *
			imgCh * sizeof(float));
	cudaMalloc((void **) &deviceOutputImage, imgW * imgH *
			imgCh * sizeof(float));
	cudaMalloc((void **) &devicefilter, filrows * filcols
			* sizeof(float));
	cudaMemcpy(deviceInputImage, hostInputImage,
			imgW * imgH * imgCh * sizeof(float),
			cudaMemcpyHostToDevice);
	cudaMemcpy(devicefilter, filter,
			filrows * filcols * sizeof(float),
			cudaMemcpyHostToDevice);


	//Grid dimensions block and grid
	dim3 dimGrid(ceil((float) imgW/TILE_WIDTH),
			ceil((float) imgH/TILE_WIDTH));
	dim3 dimBlock(TILE_WIDTH,TILE_WIDTH,1);

	// 3. execute convolution
	cudaEventCreate(&start); cudaEventCreate(&stop); 
	cudaEventRecord(start, 0);

	Convolution<<<dimGrid,dimBlock>>>(deviceInputImage, devicefilter, deviceOutputImage,
	imgCh, imgW, imgH);
	cudaEventRecord(stop, 0);

	cudaEventElapsedTime(&t, start, stop);                                                                                                                                                                 
	cudaEventDestroy(start); cudaEventDestroy(stop);
	//Tiempo
	printf("convolution time:%f ms",t/1000.0);


	//4. Copy data from Device to Host
	cudaMemcpy(hostOutputImage, deviceOutputImage, imgW * imgH *
			imgCh * sizeof(float), cudaMemcpyDeviceToHost);

	PPM_export("result.ppm", outputImage);

	
	//5. Free memories
	cudaFree(deviceInputImage);
	cudaFree(deviceOutputImage);
	cudaFree(devicefilter);

	Image_delete(outputImage);
	Image_delete(inputImage);


}