// Histogram Equalization

#include <wb.h>

#define HISTOGRAM_LENGTH 256
#define BLOCK_SIZE 128

//@@ insert code here
#define wbCheck(stmt)                                                     \
  do {                                                                    \
    cudaError_t err = stmt;                                               \
    if (err != cudaSuccess) {                                             \
      wbLog(ERROR, "Failed to run stmt ", #stmt);                         \
      wbLog(ERROR, "Got CUDA error ...  ", cudaGetErrorString(err));      \
      return -1;                                                          \
    }                                                                     \
  } while (0)


/*Cast the image from float to unsigned char
Implement a kernel that casts the image from float * to unsigned char *.

for ii from 0 to (width * height * channels) do
	ucharImage[ii] = (unsigned char) (255 * inputImage[ii])
end*/
__global__ void float_to_uchar(float *input, unsigned char *output, int len) {
  int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i < len) {
    output[i] = (unsigned char)(255 * input[i]);
  }
}

/*Convert the image from RGB to GrayScale
Implement a kernel that converts the RGB image to GrayScale. A sample sequential pseudo code is shown below.
You will find one the lectures and one of the textbook chapters helpful.

for ii from 0 to height do
	for jj from 0 to width do
		idx = ii * width + jj
		# here channels is 3
		r = ucharImage[3*idx]
		g = ucharImage[3*idx + 1]
		b = ucharImage[3*idx + 2]
		grayImage[idx] = (unsigned char) (0.21*r + 0.71*g + 0.07*b)
	end
end*/
__global__ void rgb_to_gray(unsigned char *input, unsigned char *output, int width, int height) {
  int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i < width * height) {
    int r = input[3 * i];
    int g = input[3 * i + 1];
    int b = input[3 * i + 2];
    output[i] = (unsigned char)(0.21 * r + 0.71 * g + 0.07 * b);
  }
} 

/*Compute the histogram of grayImage
Implement a kernel that computes the histogram (like in the lectures) of the image. A sample pseudo code is
shown below. You will find one of the lectures and one of the textbook chapters helpful.

histogram = [0, ...., 0] # here len(histogram) = 256
for ii from 0 to width * height do
	histogram[grayImage[ii]]++
end*/
__global__ void compute_histogram(unsigned char *input, unsigned int *output, int len) {
   //Privatization: use shared memory to hold the histogram temporarily to avoid frequent global memory atomic operations.
  __shared__ unsigned int histo_private[HISTOGRAM_LENGTH];  //bins
  if (threadIdx.x < HISTOGRAM_LENGTH) {
    histo_private[threadIdx.x] = 0;
  } //Thread i initializes the element i in the histogram, so make sure the number of threads >= HISTOGRAM_LENGTH
  __syncthreads();

  //Build private histogram
  int i = blockIdx.x * blockDim.x + threadIdx.x;
  int stride = blockDim.x * gridDim.x;  //stride is the total number of threads
  while (i < len) {
    atomicAdd(&(histo_private[input[i]]), 1); //count on each bin
    i += stride;
  }
  __syncthreads();  //Wait until all threads in the block completed

  //Build final histogram
  if (threadIdx.x < HISTOGRAM_LENGTH) {
    atomicAdd(&(output[threadIdx.x]), histo_private[threadIdx.x]);
  }   //Thread i takes care of element i in the histogram, so make sure the number of threads >= HISTOGRAM_LENGTH
} 

/*Compute the Cumulative Distribution Function of histogram
This is a scan operation like you have done in the previous lab. A sample sequential pseudo code is shown below.

cdf[0] = p(histogram[0])
for ii from 1 to 256 do
	cdf[ii] = cdf[ii - 1] + p(histogram[ii])
end
Where p() calculates the probability of a pixel to be in a histogram bin

def p(x):
	return x / (width * height)
end*/
__global__ void compute_cdf(unsigned int *input, float *output, int len, int width, int height) { //len is the length of the histogram (# of bins)
  //Use one-level scan, to be convinient, launch only one block with 128 threads (256 elements)
  __shared__ float T[2 * BLOCK_SIZE];
  //Load a block into shared memory T, note that one thread should load two elements.
  int i = 2 * blockIdx.x * blockDim.x + threadIdx.x;
  if (i < len) {
    T[threadIdx.x] = input[i] / (float)(width * height);  //Get the probability
  }else{
    T[threadIdx.x] = 0.0f;  //For the last block, we fill in 0s for the rest of the block.
  }
  if (i + blockDim.x < len) {
    T[threadIdx.x + blockDim.x] = input[i + blockDim.x]  / (float)(width * height); //Get the probability
  }else{
    T[threadIdx.x + blockDim.x] = 0.0f;  //For the last block, we fill in 0s for the rest of the block.
  }

  //Do the reduction step.
  int stride = 1;
  while(stride < 2 * BLOCK_SIZE) {
    __syncthreads();
    int index = (threadIdx.x + 1) * stride * 2 - 1; //The index of the element for this thread
    //The "+ 1" is there to make sure the index is not zero-based in this calculation. 
    //The "- 1" at the end shifts the result to be zero-based again.
    if(index < 2*BLOCK_SIZE && (index - stride) >= 0) {
      //For every step, the threads used is a half of the previous step.
      T[index] += T[index - stride];  //Do the "tree addition": this thread will take care of element "index" and "index-stride"
    }
    stride *= 2;  //Double the stride to go to next loop iteration.
  }

  //Do the post reduction step.
  stride = BLOCK_SIZE / 2;
  while(stride > 0) {
    __syncthreads();
    int index = (threadIdx.x + 1) * stride * 2 - 1; //The index of the element for this thread
    if (index + stride < 2*BLOCK_SIZE) {
      T[index + stride] += T[index];  //Do the "tree addition": this thread will take care of element "index" and "index+stride"
    }
    stride /= 2;  //Half the stride to go to next loop iteration.
  }

  //Write the result to the output array.
  __syncthreads();  //Ensure that all scan blocks are done.
  if (i < len) {  //Note that the thread i is responsable for two elements (indexed i and i+blockDim.x).
    output[i] = T[threadIdx.x];
  }
  if (i + blockDim.x < len) {
    output[i + blockDim.x] = T[threadIdx.x + blockDim.x];
  }
}

/*Compute the minimum value of the CDF. The maximal value of the CDF should be 1.0.*/
//It is obvious that the first element is the minimum, so we return it as cdfmin
//So we can directly implement it in kernel correct.

/*Define the histogram equalization function
The histogram equalization function (correct) remaps the cdf of the histogram of the image to a linear function
and is defined as

def correct_color(val) 
	return clamp(255*(cdf[val] - cdfmin)/(1.0 - cdfmin), 0, 255.0)
end

def clamp(x, start, end)
	return min(max(x, start), end)
end*/
//Use __device__ to define the function, so that it can be called by the kernel in the device.
__device__ float clamp(float x, float start, float end) {
  float low_clamp = (x > start) ? x : start;
  float up_clamp = (low_clamp < end) ? low_clamp : end;
  return up_clamp;
}

__device__ float correct_color(float *cdf, unsigned int val, float cdfmin) {
  return clamp(255 * (cdf[val] - cdfmin) / (1.0 - cdfmin), 0.0, 255.0);
}

/*Apply the histogram equalization function
Once you have implemented all of the above, then you are ready to correct the input image. This can be done by
writing a kernel to apply the correct_color() function to the RGB pixel values in parallel.

for ii from 0 to (width * height * channels) do
	ucharImage[ii] = correct_color(ucharImage[ii])
end*/
__global__ void correct(unsigned char *input, float *cdf, unsigned char *output, int len) {
  int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i < len) {
    float cdfmin = cdf[0];  //It is obvious that the first element is the minimum, so we return it as cdfmin
    unsigned int val = (unsigned int) input[i];
    output[i] = (unsigned char)(correct_color(cdf, val, cdfmin));
  }
}

/*Cast back to float
for ii from 0 to (width * height * channels) do
	outputImage[ii] = (float) (ucharImage[ii]/255.0)
end*/
__global__ void uchar_to_float(unsigned char *input, float *output, int len) {
  int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i < len) {
    output[i] = (float)(input[i] / 255.0);
  }
}


int main(int argc, char **argv) {
  wbArg_t args;
  int imageWidth;
  int imageHeight;
  int imageChannels;
  wbImage_t inputImage;
  wbImage_t outputImage;
  float *hostInputImageData;
  float *hostOutputImageData;
  const char *inputImageFile;

  //@@ Insert more code here
  float *deviceInputImageData;
  unsigned char *deviceUcharImageData;
  unsigned char *deviceGrayImageData;
  unsigned int *deviceHistogram;
  float *deviceCDF;
  unsigned char *deviceCorrectedImageData;
  float *deviceOutputImageData;

  args = wbArg_read(argc, argv); /* parse the input arguments */

  inputImageFile = wbArg_getInputFile(args, 0);

  wbTime_start(Generic, "Importing data and creating memory on host");
  inputImage = wbImport(inputImageFile);
  imageWidth = wbImage_getWidth(inputImage);
  imageHeight = wbImage_getHeight(inputImage);
  imageChannels = wbImage_getChannels(inputImage);
  outputImage = wbImage_new(imageWidth, imageHeight, imageChannels);
  hostInputImageData = wbImage_getData(inputImage);
  hostOutputImageData = wbImage_getData(outputImage);
  wbTime_stop(Generic, "Importing data and creating memory on host");

  //@@ insert code here
  int imageSize = imageWidth * imageHeight * imageChannels;
  int imageSize_singlechannel = imageWidth * imageHeight;
  wbTime_start(GPU, "Allocating GPU memory.");
  wbCheck(cudaMalloc((void **)&deviceInputImageData, imageSize * sizeof(float)));
  wbCheck(cudaMalloc((void **)&deviceUcharImageData, imageSize * sizeof(unsigned char)));
  wbCheck(cudaMalloc((void **)&deviceGrayImageData, imageSize_singlechannel * sizeof(unsigned char)));
  wbCheck(cudaMalloc((void **)&deviceHistogram, HISTOGRAM_LENGTH * sizeof(unsigned int)));
  wbCheck(cudaMalloc((void **)&deviceCDF, HISTOGRAM_LENGTH * sizeof(float)));
  wbCheck(cudaMalloc((void **)&deviceCorrectedImageData, imageSize * sizeof(unsigned char)));
  wbCheck(cudaMalloc((void **)&deviceOutputImageData, imageSize * sizeof(float)));
  wbTime_stop(GPU, "Allocating GPU memory.");

  printf("imageWidth: %d, imageHeight: %d, imageChannels: %d\n", imageWidth, imageHeight, imageChannels);

  wbTime_start(GPU, "Clearing output memory.");
  wbCheck(cudaMemset(deviceHistogram, 0, HISTOGRAM_LENGTH * sizeof(unsigned int)));
  wbCheck(cudaMemset(deviceCDF, 0, HISTOGRAM_LENGTH * sizeof(float)));
  wbCheck(cudaMemset(deviceOutputImageData, 0, imageSize * sizeof(float)));
  wbTime_stop(GPU, "Clearing output memory.");

  wbTime_start(GPU, "Copying input memory to the GPU.");
  wbCheck(cudaMemcpy(deviceInputImageData, hostInputImageData, imageSize * sizeof(float), cudaMemcpyHostToDevice));
  wbTime_stop(GPU, "Copying input memory to the GPU.");

  // wbLog(TRACE, "Is deviceInputImageData available? ", deviceInputImageData!=NULL);
  //Note that you can only access host variables, if you want to access device variables, please first copy it back to the host.
  //Otherwise if you directly access device variables from the host, you will get a segmentation fault.

  wbTime_start(Compute, "Performing CUDA computation");
  /*Cast the image from float to unsigned char*/
  dim3 dimGrid_float_to_uchar(ceil(imageSize / (1.0 * BLOCK_SIZE)), 1, 1);
  dim3 dimBlock_float_to_uchar(BLOCK_SIZE, 1, 1);
  float_to_uchar<<<dimGrid_float_to_uchar, dimBlock_float_to_uchar>>>(deviceInputImageData, deviceUcharImageData, imageSize);
  cudaDeviceSynchronize();

  wbLog(TRACE, "Kernel float_to_uchar finished.");

  /*Convert the image from RGB to GrayScale*/
  dim3 dimGrid_rgb_to_gray(ceil(imageSize_singlechannel / (1.0 * BLOCK_SIZE)), 1, 1);
  dim3 dimBlock_rgb_to_gray(BLOCK_SIZE, 1, 1);
  rgb_to_gray<<<dimGrid_rgb_to_gray, dimBlock_rgb_to_gray>>>(deviceUcharImageData, deviceGrayImageData, imageWidth, imageHeight);
  cudaDeviceSynchronize();

  wbLog(TRACE, "Kernel rgb_to_gray finished.");

  /*Compute the histogram of grayImage*/
  dim3 dimGrid_histo(ceil(imageSize_singlechannel / (1.0 * HISTOGRAM_LENGTH)), 1, 1);
  dim3 dimBlock_histo(HISTOGRAM_LENGTH, 1, 1);  //Use HISTOGRAM_LENGTH as the blockDim.x to make sure there are enough threads for each bin
  compute_histogram<<<dimGrid_histo, dimBlock_histo>>>(deviceGrayImageData, deviceHistogram, imageSize_singlechannel);
  cudaDeviceSynchronize();

  wbLog(TRACE, "Kernel compute_histogram finished.");

  /*Compute the Cumulative Distribution Function of histogram*/
  dim3 dimGrid_cdf(1, 1, 1);
  dim3 dimBlock_cdf(BLOCK_SIZE, 1, 1);  //Use 128 threads to take care of 256 elements, use scan to get the prefix sum
  compute_cdf<<<dimGrid_cdf, dimBlock_cdf>>>(deviceHistogram, deviceCDF, HISTOGRAM_LENGTH, imageWidth, imageHeight);
  cudaDeviceSynchronize();

  wbLog(TRACE, "Kernel compute_cdf finished.");

  /*Compute the minimum value of the CDF. The maximal value of the CDF should be 1.0.*/
  // float cdfmin = find_cdfmin(deviceCDF);
  // printf("Minimum value of CDF = %f\n", cdfmin);

  /*Apply the histogram equalization function*/
  dim3 dimGrid_correct(ceil(imageSize / (1.0 * BLOCK_SIZE)), 1, 1);
  dim3 dimBlock_correct(BLOCK_SIZE, 1, 1);
  correct<<<dimGrid_correct, dimBlock_correct>>>(deviceUcharImageData, deviceCDF, deviceCorrectedImageData, imageSize);
  cudaDeviceSynchronize();

  wbLog(TRACE, "Kernel correct finished.");

  /*Cast back to float*/
  dim3 dimGrid_uchar_to_float(ceil(imageSize / (1.0 * BLOCK_SIZE)), 1, 1);
  dim3 dimBlock_uchar_to_float(BLOCK_SIZE, 1, 1);
  uchar_to_float<<<dimGrid_uchar_to_float, dimBlock_uchar_to_float>>>(deviceCorrectedImageData, deviceOutputImageData, imageSize);
  cudaDeviceSynchronize();

  wbLog(TRACE, "Kernel uchar_to_float finished.");

  wbTime_stop(Compute, "Performing CUDA computation");

  wbTime_start(Copy, "Copying output memory to the CPU");
  wbCheck(cudaMemcpy(hostOutputImageData, deviceOutputImageData, imageSize * sizeof(float), cudaMemcpyDeviceToHost));
  wbTime_stop(Copy, "Copying output memory to the CPU");

  wbTime_start(Generic, "Exporting data");
  wbImage_setData(outputImage, hostOutputImageData);
  wbTime_stop(Generic, "Exporting data");

  wbTime_start(GPU, "Freeing GPU Memory");
  cudaFree(deviceInputImageData);
  cudaFree(deviceUcharImageData);
  cudaFree(deviceGrayImageData);
  cudaFree(deviceHistogram);
  cudaFree(deviceCDF);
  cudaFree(deviceCorrectedImageData);
  cudaFree(deviceOutputImageData);
  wbTime_stop(GPU, "Freeing GPU Memory");

  wbSolution(args, outputImage);

  //@@ insert code here
  //Free host memory, only pointers need to be freed
  free(hostInputImageData);
  free(hostOutputImageData);

  return 0;
}
