// MP Scan
// Given a list (lst) of length n
// Output its prefix sum = {lst[0], lst[0] + lst[1], lst[0] + lst[1] + ...
// +
// lst[n-1]}

#include <wb.h>

#define BLOCK_SIZE 512 //@@ You can change this

#define wbCheck(stmt)                                                     \
  do {                                                                    \
    cudaError_t err = stmt;                                               \
    if (err != cudaSuccess) {                                             \
      wbLog(ERROR, "Failed to run stmt ", #stmt);                         \
      wbLog(ERROR, "Got CUDA error ...  ", cudaGetErrorString(err));      \
      return -1;                                                          \
    }                                                                     \
  } while (0)

// __global__ void scan_block_sums(float *block_sums, float *block_sums_scanned, int len) {

// }

__global__ void add_offset(float *block_sums_scanned, float *block_scanned, float *output, int len) {
  int i = 2 * blockIdx.x * blockDim.x + threadIdx.x;
  __shared__ float offset;  //One offset for each block.
  if (threadIdx.x == 0) {
    if (blockIdx.x == 0) {
      offset = 0.0f;  //For the first block, the offset is 0.
    }else{  //For other blocks, the offset is the accumulative sum of the previous blocks.
      offset = block_sums_scanned[blockIdx.x - 1];  
    }
  }
  __syncthreads();
  if (i < len) {  
    output[i] = block_scanned[i] + offset;
  }
  if (i + blockDim.x < len) {
    output[i + blockDim.x] = block_scanned[i + blockDim.x] + offset;
  }
}

__global__ void scan(float *input, float *output, float *aux, int len) {
  //@@ Modify the body of this function to complete the functionality of
  //@@ the scan on the device
  //@@ You may need multiple kernel calls; write your kernels before this
  //@@ function and call them from the host
  //Breut-Kung Parallel Scan
  
  __shared__ float T[2 * BLOCK_SIZE];
  //Load a block into shared memory T, note that one thread should load two elements.
  int i = 2 * blockIdx.x * blockDim.x + threadIdx.x;
  if (i < len) {
    T[threadIdx.x] = input[i];
  }else{
    T[threadIdx.x] = 0.0f;  //For the last block, we fill in 0s for the rest of the block.
  }
  if (i + blockDim.x < len) {
    T[threadIdx.x + blockDim.x] = input[i + blockDim.x];
  }else{
    T[threadIdx.x + blockDim.x] = 0.0f;  //For the last block, we fill in 0s for the rest of the block.
  }

  //Do the reduction step.
  int stride = 1;
  while(stride < 2*BLOCK_SIZE) {
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

  //Write the block sums to the auxiliary array if aux is not NULL.
  if (aux != NULL) {
    __syncthreads(); //Ensure that all output are done.
    if (threadIdx.x == 0) { //For each block, use one thread to copy the last element of each block into the auxiliary array.
      aux[blockIdx.x] = T[2*BLOCK_SIZE - 1];
    }
  }
}

int main(int argc, char **argv) {
  wbArg_t args;
  float *hostInput;  // The input 1D list
  float *hostOutput; // The output list
  float *deviceInput;
  float *deviceOutput;
  float *deviceScanBlock;       //The output of the scan on each block.
  float *deviceAuxiliaryArray;  //The auxiliary array used to store the block sums.
  float *deviceScanBlockSums;   //The output of the scan on an auxiliary array where each entry is the sum of each block above.
  int numElements; // number of elements in the list

  args = wbArg_read(argc, argv);

  wbTime_start(Generic, "Importing data and creating memory on host");
  hostInput = (float *)wbImport(wbArg_getInputFile(args, 0), &numElements);
  hostOutput = (float *)malloc(numElements * sizeof(float));
  wbTime_stop(Generic, "Importing data and creating memory on host");

  wbLog(TRACE, "The number of input elements in the input is ",
        numElements);

  int numBlocks = ceil(numElements/(2.0*BLOCK_SIZE)); //Calculate the number of blocks for convinience.
  // wbLog(TRACE, "The number of blocks in the input is ",
        // numBlocks); //
  
  wbTime_start(GPU, "Allocating GPU memory.");
  wbCheck(cudaMalloc((void **)&deviceInput, numElements * sizeof(float)));
  wbCheck(cudaMalloc((void **)&deviceOutput, numElements * sizeof(float)));
  wbCheck(cudaMalloc((void **)&deviceScanBlock, numElements * sizeof(float)));
  wbCheck(cudaMalloc((void **)&deviceAuxiliaryArray, numBlocks * sizeof(float)));
  wbCheck(cudaMalloc((void **)&deviceScanBlockSums, numBlocks * sizeof(float)));
  wbTime_stop(GPU, "Allocating GPU memory.");

  wbTime_start(GPU, "Clearing output memory.");
  wbCheck(cudaMemset(deviceOutput, 0, numElements * sizeof(float)));
  wbTime_stop(GPU, "Clearing output memory.");

  wbTime_start(GPU, "Copying input memory to the GPU.");
  wbCheck(cudaMemcpy(deviceInput, hostInput, numElements * sizeof(float),
                     cudaMemcpyHostToDevice));
  wbTime_stop(GPU, "Copying input memory to the GPU.");

  //@@ Initialize the grid and block dimensions here
  dim3 dimGrid(numBlocks, 1, 1);
  dim3 dimBlock(BLOCK_SIZE, 1, 1);
  dim3 dimGridAux(1, 1, 1);

  wbTime_start(Compute, "Performing CUDA computation");
  //@@ Modify this to complete the functionality of the scan
  //@@ on the deivce
  scan<<<dimGrid, dimBlock>>>(deviceInput, deviceScanBlock, deviceAuxiliaryArray, numElements);
  cudaDeviceSynchronize();

  // wbLog(TRACE, "Kernel 1 finished."); //

  scan<<<dimGridAux, dimGrid>>>(deviceAuxiliaryArray, deviceScanBlockSums, NULL, numBlocks);
  cudaDeviceSynchronize();

  // wbLog(TRACE, "Kernel 2 finished."); //

  add_offset<<<dimGrid, dimBlock>>>(deviceScanBlockSums, deviceScanBlock, deviceOutput, numElements);
  cudaDeviceSynchronize();

  // wbLog(TRACE, "Kernel 3 finished."); //

  wbTime_stop(Compute, "Performing CUDA computation");

  wbTime_start(Copy, "Copying output memory to the CPU");
  wbCheck(cudaMemcpy(hostOutput, deviceOutput, numElements * sizeof(float),
                     cudaMemcpyDeviceToHost));
  wbTime_stop(Copy, "Copying output memory to the CPU");

  wbTime_start(GPU, "Freeing GPU Memory");
  cudaFree(deviceInput);
  cudaFree(deviceOutput);
  cudaFree(deviceScanBlock);
  cudaFree(deviceAuxiliaryArray);
  cudaFree(deviceScanBlockSums);
  wbTime_stop(GPU, "Freeing GPU Memory");

  wbSolution(args, hostOutput, numElements);

  free(hostInput);
  free(hostOutput);

  return 0;
}
