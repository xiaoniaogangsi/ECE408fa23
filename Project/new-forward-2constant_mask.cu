#include <cmath>
#include <iostream>
#include "gpu-new-forward.h"

#define TILE_WIDTH 8
#define MAX_MASK_WIDTH 7
#define MAX_CHANNEL_NUM 4
#define MAX_STRIDE 4
#define MAX_OUT_FM 16

#define MAX_SIN_WIDTH  ((TILE_WIDTH-1) * MAX_STRIDE + MAX_MASK_WIDTH)    //The maximum width of the shared input
#define MAX_SIN_CHSIZE  (MAX_SIN_WIDTH * MAX_SIN_WIDTH)            //The maximum size of each channel in the shared input 
#define MAX_SIN_SIZE  (MAX_CHANNEL_NUM * MAX_SIN_CHSIZE)            //The maximum size of the shared input
#define MAX_SMASK_CHSIZE  (MAX_MASK_WIDTH * MAX_MASK_WIDTH)          //The maximum size of each channel in the shared mask
#define MAX_SMASK_SIZE  (MAX_CHANNEL_NUM * MAX_SMASK_CHSIZE)        //The maximum size of each channel in the shared mask

__constant__ float Mask_c[MAX_OUT_FM * MAX_CHANNEL_NUM * MAX_MASK_WIDTH * MAX_MASK_WIDTH];

__global__ void conv_forward_kernel(float *output, const float *input, const float *mask, const int B, const int M, const int C, const int H, const int W, const int K,const int S)
{
    /*
    Modify this function to implement the forward pass described in Chapter 16.
    We have added an additional dimension to the tensors to support an entire mini-batch
    The goal here is to be correct AND fast.

    Function paramter definitions:
    output - output
    input - input
    mask - convolution kernel
    B - batch_size (number of images in x)
    M - number of output feature maps
    C - number of input feature maps
    H - input height dimension
    W - input width dimension
    K - kernel height and width (K x K)
    S - stride step length
    */

    /*Optimization 1: Tiled shared memory convolution
    Inside each block, each output element needs K*K elements from the input feature and from the mask,
    so there are many loads from global memory. We can copy elements from input and mask into shared memory.*/

    
    __shared__ float subTile_input[MAX_SIN_SIZE];
    // __shared__ float subTile_mask[MAX_SMASK_SIZE];

    int s_in_width = (TILE_WIDTH-1) * S + K;
    int s_in_chsize = s_in_width * s_in_width;      //The size of each channel in the shared input 
    int s_in_size = C * s_in_chsize;                //The size of the shared input

    int s_mask_chsize = K * K;              //The size of each channel in the shared mask
    int s_mask_size = C * s_mask_chsize;    //The size of each channel in the shared mask

    const int H_out = (H - K)/S + 1;
    const int W_out = (W - K)/S + 1;
    // (void)H_out; // silence declared but never referenced warning. remove this line when you start working
    // (void)W_out; // silence declared but never referenced warning. remove this line when you start working
    const int H_grid = ceil(H_out / (1.0 * TILE_WIDTH));    //Number of tiles in height for one output feature map
    const int W_grid = ceil(W_out / (1.0 * TILE_WIDTH));    //Number of tiles in width for one output feature map

    // We have some nice #defs for you below to simplify indexing. Feel free to use them, or create your own.
    // An example use of these macros:
    // float a = in_4d(0,0,0,0)
    // out_4d(0,0,0,0) = a

    #define out_4d(i3, i2, i1, i0) output[(i3) * (M * H_out * W_out) + (i2) * (H_out * W_out) + (i1) * (W_out) + i0]
    #define in_4d(i3, i2, i1, i0) input[(i3) * (C * H * W) + (i2) * (H * W) + (i1) * (W) + i0]
    #define mask_4d(i3, i2, i1, i0) mask[(i3) * (C * K * K) + (i2) * (K * K) + (i1) * (K) + i0]
    //out_4d(b, m, h_out, w_out)
    //in_4d(b, c, h, w)
    //mask_4d(m, c, p, q)
    #define shared_in_3d(i2, i1, i0) subTile_input[(i2) * (s_in_width * s_in_width) + (i1) * (s_in_width) + i0]
    // #define shared_mask_3d(i2, i1, i0) subTile_mask[(i2) * (K * K) + (i1) * (K) + i0]
    //shared_in_3d(c, h, w)
    //shared_mask_3d(c, p, q)
    #define const_mask_4d(i3, i2, i1, i0) Mask_c[(i3) * (C * K * K) + (i2) * (K * K) + (i1) * (K) + i0]
    //const_mask_4d(m, c, p, q)

    // Insert your GPU convolution kernel code here
    int m = blockIdx.x; //The index of the output feature map.
    //We need to recombine the linearized tiles to a feature map with (W_grid * H_grid) tiles.
    //Then use (h, w) to index each element in the recombined feature maps.
    int h = (blockIdx.y / W_grid) * TILE_WIDTH + threadIdx.y; //The height index of the output tile.
    int w = (blockIdx.y % W_grid) * TILE_WIDTH + threadIdx.x; //The width index of the output tile.
    int b = blockIdx.z; //The index of the image in the batch.

    //Copy the input feature map and mask into shared memory.
    int tile_index = threadIdx.y * TILE_WIDTH + threadIdx.x;
    int tile_size = TILE_WIDTH * TILE_WIDTH;
    int h_base = (blockIdx.y / W_grid) * TILE_WIDTH;
    int w_base = (blockIdx.y % W_grid) * TILE_WIDTH;

    for (int index = tile_index; index < s_in_size; index += tile_size) {
        int c = index / s_in_chsize;
        int h_in = (index % s_in_chsize) / s_in_width;
        int w_in = (index % s_in_chsize) % s_in_width;
        if (h_base*S + h_in < H && w_base*S + w_in < W) {
            shared_in_3d(c, h_in, w_in) = in_4d(b, c, h_base*S + h_in, w_base*S + w_in);
        }
        else {
            shared_in_3d(c, h_in, w_in) = 0.0f;
        }
    }
    __syncthreads();

    // for (int index = tile_index; index < s_mask_size; index += tile_size) {
    //     int c = index / s_mask_chsize;
    //     int p = (index % s_mask_chsize) / K;
    //     int q = (index % s_mask_chsize) % K;
    //     shared_mask_3d(c, p, q) = mask_4d(m, c, p, q);
    // }
    // __syncthreads();

    if (h<H_out && w<W_out) {   //Check the boundary conditions of the output
        float acc = 0.0f;
        for (int c=0; c<C; c++) {   //Iterate on each channel
            for (int p=0; p<K; p++) {
                for (int q=0; q<K; q++) {   //Iterate on each mask element (there are K*K mask elements)
                    //Multiply the mask element with the corresponding input element and accumulate the result.
                    // if (h*S+p < H && w*S+q < W){    //Check the boundary condition of input, remember the stride S.
                    //     acc += in_4d(b, c, h*S+p, w*S+q) * mask_4d(m, c, p, q);
                    // }
                    if (threadIdx.y*S+p < s_in_width && threadIdx.x*S+q < s_in_width) {
                        // acc += shared_in_3d(c, threadIdx.y*S+p, threadIdx.x*S+q) * shared_mask_3d(c, p, q);
                        acc += shared_in_3d(c, threadIdx.y*S+p, threadIdx.x*S+q) * const_mask_4d(m, c, p, q);
                    }
                }
            }
        }
        out_4d(b, m, h, w) = acc;   //Store the result into output.
    }

    #undef out_4d
    #undef in_4d
    #undef mask_4d

    #undef shared_in_3d
    // #undef shared_mask_3d

    #undef const_mask_4d
}

	
__host__ void GPUInterface::conv_forward_gpu_prolog(const float *host_output, const float *host_input, const float *host_mask, float **device_output_ptr, float **device_input_ptr, float **device_mask_ptr, const int B, const int M, const int C, const int H, const int W, const int K, const int S)
{
    // Allocate memory and copy over the relevant data structures to the GPU

    // We pass double pointers for you to initialize the relevant device pointers,
    //  which are passed to the other two functions.

    // Useful snippet for error checking
    // cudaError_t error = cudaGetLastError();
    // if(error != cudaSuccess)
    // {
    //     std::cout<<"CUDA error: "<<cudaGetErrorString(error)<<std::endl;
    //     exit(-1);
    // }
    const int H_out = (H - K)/S + 1;
    const int W_out = (W - K)/S + 1;
   
    cudaMalloc((void **)device_output_ptr, B * M * H_out * W_out * sizeof(float));
    cudaMalloc((void **)device_input_ptr, B * C * H * W * sizeof(float));
    cudaMalloc((void **)device_mask_ptr, M * C * K * K * sizeof(float));

    cudaMemcpy(*device_input_ptr, host_input, B * C * H * W * sizeof(float), cudaMemcpyHostToDevice);
    // cudaMemcpy(*device_mask_ptr, host_mask, M * C * K * K * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpyToSymbol(Mask_c, host_mask, M * C * K * K * sizeof(float));
}


__host__ void GPUInterface::conv_forward_gpu(float *device_output, const float *device_input, const float *device_mask, const int B, const int M, const int C, const int H, const int W, const int K, const int S)
{
    // Set the kernel dimensions and call the kernel
    const int H_out = (H - K)/S + 1;    //Output height for one output feature map
    const int W_out = (W - K)/S + 1;    //Output width for one output feature map
    const int H_grid = ceil(H_out / (1.0 * TILE_WIDTH));    //Number of tiles in height for one output feature map
    const int W_grid = ceil(W_out / (1.0 * TILE_WIDTH));    //Number of tiles in width for one output feature map
    const int Y = H_grid * W_grid;      //Number of tiles for one output feature map (linearized)

    dim3 dimGrid(M, Y, B);  //There are total B images in a batch. And for each image, there are M output feature map, each with Y tiles.
    dim3 dimBlock(TILE_WIDTH, TILE_WIDTH, 1);   //The dimension of a block (a tile)
    
    //---Debug---
    // std::cout<<"H_out: "<<H_out<<std::endl;
    // std::cout<<"W_out: "<<W_out<<std::endl;
    // std::cout<<"H_grid: "<<H_grid<<std::endl;
    // std::cout<<"W_grid: "<<W_grid<<std::endl;
    // std::cout<<"Y: "<<Y<<std::endl;
    // std::cout<<"dimGrid: x="<<dimGrid.x<<", y="<<dimGrid.y<<", z="<<dimGrid.z<<std::endl;
    // std::cout<<"dimBlock: x="<<dimBlock.x<<", y="<<dimBlock.y<<", z="<<dimBlock.z<<std::endl;
    //---Debug---

    //Launch the convolution forward kernel.
    conv_forward_kernel<<<dimGrid, dimBlock>>>(device_output, device_input, device_mask, B, M, C, H, W, K, S);
    cudaDeviceSynchronize();
}


__host__ void GPUInterface::conv_forward_gpu_epilog(float *host_output, float *device_output, float *device_input, float *device_mask, const int B, const int M, const int C, const int H, const int W, const int K, const int S)
{
    // Copy the output back to host
    const int H_out = (H - K)/S + 1;
    const int W_out = (W - K)/S + 1;
    cudaMemcpy(host_output, device_output, B * M * H_out * W_out * sizeof(float), cudaMemcpyDeviceToHost);
   
    // Free device memory
    cudaFree(device_output);
    cudaFree(device_input);
    cudaFree(device_mask);
}


__host__ void GPUInterface::get_device_properties()
{
    int deviceCount;
    cudaGetDeviceCount(&deviceCount);

    for(int dev = 0; dev < deviceCount; dev++)
    {
        cudaDeviceProp deviceProp;
        cudaGetDeviceProperties(&deviceProp, dev);

        std::cout<<"Device "<<dev<<" name: "<<deviceProp.name<<std::endl;
        std::cout<<"Computational capabilities: "<<deviceProp.major<<"."<<deviceProp.minor<<std::endl;
        std::cout<<"Max Global memory size: "<<deviceProp.totalGlobalMem<<std::endl;
        std::cout<<"Max Constant memory size: "<<deviceProp.totalConstMem<<std::endl;
        std::cout<<"Max Shared memory size per block: "<<deviceProp.sharedMemPerBlock<<std::endl;
        std::cout<<"Max threads per block: "<<deviceProp.maxThreadsPerBlock<<std::endl;
        std::cout<<"Max block dimensions: "<<deviceProp.maxThreadsDim[0]<<" x, "<<deviceProp.maxThreadsDim[1]<<" y, "<<deviceProp.maxThreadsDim[2]<<" z"<<std::endl;
        std::cout<<"Max grid dimensions: "<<deviceProp.maxGridSize[0]<<" x, "<<deviceProp.maxGridSize[1]<<" y, "<<deviceProp.maxGridSize[2]<<" z"<<std::endl;
        std::cout<<"Warp Size: "<<deviceProp.warpSize<<std::endl;
    }
}
