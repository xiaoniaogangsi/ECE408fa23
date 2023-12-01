#include <cmath>
#include <iostream>
#include "gpu-new-forward.h"
#include <fstream>

#define TILE_WIDTH 16
#define MAX_MASK_WIDTH 7
#define MAX_CHANNEL_NUM 4
#define MAX_STRIDE 4
#define MAX_OUT_FM 16

#include <mma.h>
using namespace nvcuda;

__constant__ half Mask_c[MAX_OUT_FM * MAX_CHANNEL_NUM * MAX_MASK_WIDTH * MAX_MASK_WIDTH];

//Convert float (FP32) to half (FP16)
__global__ void convertFP32toFP16(half *out, const float *in, int n)
{
    int idx = blockDim.x * blockIdx.x + threadIdx.x;
    if (idx < n) {
       out[idx] = __float2half(in[idx]);
    }
}

//Convert half (FP16) to float (FP32)
__global__ void convertFP16toFP32(float *out, half *in, int n)
{
    int idx = blockDim.x * blockIdx.x + threadIdx.x;
    if (idx < n) {
       out[idx] = __half2float(in[idx]);
    }
}

__global__ void conv_forward_kernel(half *output, const half *input, const half *mask, const int B, const int M, const int C, const int H, const int W, const int K,const int S)
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

    /*Optimization 2: Weight matrix (kernel values) in constant memory*/

    /*Optimization 3: Shared memory matrix multiplication and input matrix unrolling*/

    /*Optimization 4: Kernel fusion for unrolling and matrix-multiplication*/

    /*Optimization 5: Fixed point (FP16) arithmetic (based on Optimization 3)*/

    __shared__ half subTile_input[TILE_WIDTH][TILE_WIDTH];
    __shared__ half subTile_mask[TILE_WIDTH][TILE_WIDTH];

    const int H_out = (H - K)/S + 1;
    const int W_out = (W - K)/S + 1;
    // const int H_grid = ceil(H_out / (1.0 * TILE_WIDTH));    //Number of tiles in height for one output feature map
    // const int W_grid = ceil(W_out / (1.0 * TILE_WIDTH));    //Number of tiles in width for one output feature map

    int num_input_rows = C * K * K;
    int num_input_columns = H_out * W_out;
    int num_mask_rows = M;
    int num_mask_columns = C * K * K;
    int num_output_rows = M;
    int num_output_columns = H_out * W_out;

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

    #define uin_3d(i2, i1, i0) input[(i2) * (C * K * K * H_out * W_out) + (i1) * (H_out * W_out) + (i0)]
    //uin_3d(b, h_unrolled, w_unrolled)

    // Insert your GPU convolution kernel code here
    int m = blockIdx.x; //The index of the output feature map.
    //We need to recombine the linearized tiles to a feature map with (W_grid * H_grid) tiles.
    //Then use (h, w) to index each element in the recombined feature maps.
    // int h = (blockIdx.y / W_grid) * TILE_WIDTH + threadIdx.y; //The height index of the output tile.
    // int w = (blockIdx.y % W_grid) * TILE_WIDTH + threadIdx.x; //The width index of the output tile.
    int b = blockIdx.z; //The index of the image in the batch.

    int bx = blockIdx.x;
    int by = blockIdx.y;
    int tx = threadIdx.x;
    int ty = threadIdx.y;
    int Row = by * TILE_WIDTH + ty; //The row number of the current element in matrix C
    int Col = bx * TILE_WIDTH + tx; //The column number of the current element in matrix C
    half output_val = __float2half(0.0f);

    //Loop on each pair of tiles from A and B, 
    //Note that here (num_mask_columns-1) / TILE_WIDTH + 1 == ceil(num_mask_columns/TILE_WIDTH)
    for (int q=0; q < (num_mask_columns-1) / TILE_WIDTH + 1; q++){  //num_mask_columns == num_input_rows
    //load subtiles of A and B from the global memory to the shared memory of this block
        if (Row < num_mask_rows && q*TILE_WIDTH+tx < num_mask_columns){ //Boundary condition
            subTile_mask[ty][tx] = Mask_c[Row * num_mask_columns + q * TILE_WIDTH + tx];
        }else{
            subTile_mask[ty][tx] = __float2half(0.0f);
        }
        if (q*TILE_WIDTH+ty < num_input_rows && Col < num_input_columns) { //Boundary condition condition
            subTile_input[ty][tx] = uin_3d(b, (q * TILE_WIDTH + ty), Col);  //B[(q * TILE_WIDTH + ty) * num_input_columns + Col];
        }else{
            subTile_input[ty][tx] = __float2half(0.0f);
        }
        __syncthreads();  //Synchronize the threads, we need to wait all threads finish loading the data
        //Use the subtiles to calculate the partial sum for the element, and accumulate into output_val
        if (Row < num_output_rows && Col < num_output_columns){ //Boundary condition
            for (int k=0; k<TILE_WIDTH; k++){
                output_val += subTile_mask[ty][k] * subTile_input[k][tx];
                // output_val += Mask_c[Row * num_mask_columns + q * TILE_WIDTH + k] * subTile_input[k][tx];
            }
        }
        __syncthreads();  //Synchronize the threads, we need to wait all threads finish computation, so that we can get the correct answer
    }

    if (Row < num_output_rows && Col < num_output_columns){ //Boundary condition
        output[b * (num_output_rows * num_output_columns) + Row * num_output_columns + Col] = output_val;  //Store the result element back
    }

    #undef out_4d
    #undef in_4d
    #undef mask_4d

    #undef uin_3d
}

__global__ void unroll_Kernel(const half* device_input, half* device_input_unrolled, const int B, const int M, const int C, const int H, const int W, const int K, const int S) {
    int t = blockIdx.x * blockDim.x + threadIdx.x;
    int b = blockIdx.y;     //image index
    const int H_out = (H - K)/S + 1;
    const int W_out = (W - K)/S + 1;
    // Width of the unrolled input feature matrix
    int W_unroll = H_out * W_out;

    #define in_4d(i3, i2, i1, i0) device_input[(i3) * (C * H * W) + (i2) * (H * W) + (i1) * (W) + i0]
    #define uin_3d(i2, i1, i0) device_input_unrolled[(i2) * (C * K * K * H_out * W_out) + (i1) * (H_out * W_out) + (i0)]
    //in_4d(b, c, h, w)
    //uin_3d(b, h_unrolled, w_unrolled)

    if (t < C * W_unroll) {
        // Channel of the input feature map being collected by the thread
        int c = t / W_unroll;
        // Column index of the unrolled matrix to write a strip of
        // input elements into (also, the linearized index of the output
        // element for which the thread is collecting input elements)
        int w_unroll = t % W_unroll;
        // Horizontal and vertical indices of the output element
        int h_out = w_unroll / W_out;
        int w_out = w_unroll % W_out;
        // Starting row index for the unrolled matrix section for channel c
        int w_base = c * K * K;
        for(int p = 0; p < K; p++){
            for(int q = 0; q < K; q++) {
                // Row index of the unrolled matrix for the thread to write
                // the input element into for the current iteration
                int h_unroll = w_base + p*K + q;
                if (h_out*S+p < H && w_out*S+q < W){
                    uin_3d(b, h_unroll, w_unroll) = in_4d(b, c, h_out*S + p, w_out*S + q);
                }else{
                    uin_3d(b, h_unroll, w_unroll) = __float2half(0.0f);
                }
            }
        }
    }

    #undef in_4d
    #undef uin_3d
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
    // cudaMalloc((void **)device_input_ptr, B * C * K * K * H_out * W_out * sizeof(float));
    cudaMalloc((void **)device_mask_ptr, M * C * K * K * sizeof(float));

    cudaMemcpy(*device_input_ptr, host_input, B * C * H * W * sizeof(float), cudaMemcpyHostToDevice);
    // cudaMemcpy(*device_input_ptr, host_input_unrolled, B * C * K * K * H_out * W_out * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(*device_mask_ptr, host_mask, M * C * K * K * sizeof(float), cudaMemcpyHostToDevice);
    // cudaMemcpyToSymbol(Mask_c, host_mask, M * C * K * K * sizeof(float));
    // cudaMemcpyToSymbol(Mask_c, host_mask_unrolled, M * C * K * K * sizeof(float));

    std::ofstream outfile;
	outfile.open("host_mask_org.txt");
    for (int i = 0; i < M; i++){
        for (int j = 0; j < C*K*K; j++){
            outfile << host_mask[i * (C*K*K) + j] << " ";
        }
        outfile << std::endl;
    }
    outfile.close();

    outfile.open("host_input_org.txt");
    for (int b=0; b<B; b++){
        outfile << "b = " << b << ":-----------------------------------------" << std::endl;
        for (int c=0; c<C; c++){
        outfile << "-- c= " << c << std::endl;
            for (int i = 0; i < H; i++){
                for (int j = 0; j < W; j++){
                    outfile << host_input[b*(C*H*W) + c*(H*W) + i * W + j] << " ";
                }
                outfile << std::endl;
            }
        }
    }
    outfile.close();
}


__host__ void GPUInterface::conv_forward_gpu(float *device_output, const float *device_input, const float *device_mask, const int B, const int M, const int C, const int H, const int W, const int K, const int S)
{
    // Set the kernel dimensions and call the kernel
    const int H_out = (H - K)/S + 1;    //Output height for one output feature map
    const int W_out = (W - K)/S + 1;    //Output width for one output feature map
    // const int H_grid = ceil(H_out / (1.0 * TILE_WIDTH));    //Number of tiles in height for one output feature map
    // const int W_grid = ceil(W_out / (1.0 * TILE_WIDTH));    //Number of tiles in width for one output feature map
    // const int Y = H_grid * W_grid;      //Number of tiles for one output feature map (linearized)

    // Run the kernel to convert float into half (FP16)
    half *device_output_fp16; 
    half *device_input_fp16;
    half *device_mask_fp16;
    cudaMalloc((void **)&device_output_fp16, B * M * H_out * W_out * sizeof(half));
    cudaMalloc((void **)&device_input_fp16, B * C * H * W * sizeof(half));
    cudaMalloc((void **)&device_mask_fp16, M * C * K * K * sizeof(half));

    convertFP32toFP16<<<(B * C * H * W + 255) / 256, 256>>>(device_input_fp16, device_input, B * C * H * W);
    convertFP32toFP16<<<(M * C * K * K + 255) / 256, 256>>>(device_mask_fp16, device_mask, M * C * K * K);
    cudaDeviceSynchronize();
    cudaMemcpyToSymbol(Mask_c, device_mask_fp16, M * C * K * K * sizeof(half));

    //---Debug---
    half *host_mask_fp16 = (half *)malloc(M*C*K*K * sizeof(half));
    cudaMemcpy(host_mask_fp16, device_mask_fp16, M*C*K*K * sizeof(half), cudaMemcpyDeviceToHost);
    
    std::ofstream outfile;
    outfile.open("host_mask_fp16.txt");
    for (int i = 0; i < M; i++){
        for (int j = 0; j < C*K*K; j++){
            outfile << __half2float(host_mask_fp16[i * (C*K*K) + j]) << " ";
        }
        outfile << std::endl;
    }
    outfile.close();

    free(host_mask_fp16);
    //---Debug---

    // The kernel to unroll the input feature maps-----------------------------------------------------
    half *device_input_unrolled_fp16;
    cudaMalloc((void **)&device_input_unrolled_fp16, B * C * K * K * H_out * W_out * sizeof(half));    

    //The 1D grid and block for GPU unrolling
    //Each CUDA thread will be responsible for gathering (K*K) input elements from one input feature map for one element of an output feature map.
    int num_threads = C * H_out * W_out;
    int num_blocks = ceil(num_threads / (1.0 * TILE_WIDTH));
    dim3 dimGrid_unroll(num_blocks, B, 1);     //Note we have B images
    dim3 dimBlock_unroll(TILE_WIDTH, 1, 1);
    unroll_Kernel<<<dimGrid_unroll, dimBlock_unroll>>>(device_input_fp16, device_input_unrolled_fp16, B, M, C, H, W, K, S);
    cudaDeviceSynchronize();
    //Kernel ends--------------------------------------------------------------------------------------------------

    //---Debug---
    half *host_input_unrolled_fp16 = (half *)malloc(B * C * K * K * H_out * W_out * sizeof(half));
    cudaMemcpy(host_input_unrolled_fp16, device_input_unrolled_fp16, B * C * K * K * H_out * W_out * sizeof(half), cudaMemcpyDeviceToHost);
    // std::ofstream outfile;
	outfile.open("host_input_unrolled_fp16.txt");
    for (int b=0; b<B; b++){
        outfile << "b = " << b << ":-----------------------------------------" << std::endl;
        for (int i = 0; i < C*K*K; i++){
            for (int j = 0; j < H_out*W_out; j++){
                outfile << __half2float(host_input_unrolled_fp16[b*(C * K * K * H_out * W_out) + i * (H_out*W_out) + j]) << " ";
            }
            outfile << std::endl;
        }
    }
    outfile.close();
    free(host_input_unrolled_fp16);
    //---Debug---

    const int X_grid = ceil((1.0 * H_out * W_out) / (1.0 * TILE_WIDTH));
    const int M_grid = ceil((1.0 * M) / (1.0 * TILE_WIDTH));    //Number of tiles in height for all output feature maps in an image

    // dim3 dimGrid(M, Y, B);  //There are total B images in a batch. And for each image, there are M output feature map, each with Y tiles.
    dim3 dimGrid(X_grid, M_grid, B);
    dim3 dimBlock(TILE_WIDTH, TILE_WIDTH, 1);   //The dimension of a block (a tile)
    
    //---Debug---
    // std::cout<<"H_out: "<<H_out<<std::endl;
    // std::cout<<"W_out: "<<W_out<<std::endl;
    // std::cout<<"H_grid: "<<H_grid<<std::endl;
    // std::cout<<"W_grid: "<<W_grid<<std::endl;
    // std::cout<<"Y: "<<Y<<std::endl;
    // std::cout<<"X_grid: "<<X_grid<<std::endl;
    // std::cout<<"M_grid: "<<M_grid<<std::endl;
    // std::cout<<"dimGrid: x="<<dimGrid.x<<", y="<<dimGrid.y<<", z="<<dimGrid.z<<std::endl;
    // std::cout<<"dimBlock: x="<<dimBlock.x<<", y="<<dimBlock.y<<", z="<<dimBlock.z<<std::endl;
    //---Debug---

    //Launch the convolution forward kernel.
    // conv_forward_kernel<<<dimGrid, dimBlock>>>(device_output, device_input, device_mask, B, M, C, H, W, K, S);
    conv_forward_kernel<<<dimGrid, dimBlock>>>(device_output_fp16, device_input_unrolled_fp16, device_mask_fp16, B, M, C, H, W, K, S);
    cudaDeviceSynchronize();

    //---Debug---
    half *host_output_fp16 = (half *)malloc(B * M * H_out * W_out * sizeof(half));
    cudaMemcpy(host_output_fp16, device_output_fp16, B * M * H_out * W_out * sizeof(half), cudaMemcpyDeviceToHost);
    // std::ofstream outfile;
	outfile.open("host_output_fp16.txt");
    for (int b=0; b<B; b++){
        outfile << "b = " << b << ":-----------------------------------------" << std::endl;
        for (int i = 0; i < M; i++){
            for (int j = 0; j < H_out * W_out; j++){
                outfile << __half2float(host_output_fp16[b*(M * H_out * W_out) + i * (H_out * W_out) + j]) << " ";
            }
            outfile << std::endl;
        }
    }
    outfile.close();
    free(host_output_fp16);
    //---Debug---
    
    // Run the kernel to convert half (FP16) back to float (FP32)
    convertFP16toFP32<<<(B * M * H_out * W_out + 255) / 256, 256>>>(device_output, device_output_fp16, B * M * H_out * W_out);
    cudaDeviceSynchronize();

    cudaFree(device_output_fp16);
    cudaFree(device_input_fp16);
    cudaFree(device_mask_fp16);

    cudaFree(device_input_unrolled_fp16);
}


__host__ void GPUInterface::conv_forward_gpu_epilog(float *host_output, float *device_output, float *device_input, float *device_mask, const int B, const int M, const int C, const int H, const int W, const int K, const int S)
{
    // Copy the output back to host
    const int H_out = (H - K)/S + 1;
    const int W_out = (W - K)/S + 1;
    cudaMemcpy(host_output, device_output, B * M * H_out * W_out * sizeof(float), cudaMemcpyDeviceToHost);
   
    //---Debug---
    std::ofstream outfile;
	outfile.open("host_output.txt");
    for (int b=0; b<B; b++){
        outfile << "b = " << b << ":-----------------------------------------" << std::endl;
        for (int i = 0; i < M; i++){
            for (int j = 0; j < H_out * W_out; j++){
                outfile << host_output[b*(M * H_out * W_out) + i * (H_out * W_out) + j] << " ";
            }
            outfile << std::endl;
        }
    }
    outfile.close();

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
