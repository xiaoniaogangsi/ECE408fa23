#include <cmath>
#include <iostream>
#include "gpu-new-forward.h"

#define TILE_WIDTH 16
#define MAX_MASK_WIDTH 7
#define MAX_CHANNEL_NUM 4
#define MAX_STRIDE 4
#define MAX_OUT_FM 16

__constant__ float Mask_c[MAX_OUT_FM * MAX_CHANNEL_NUM * MAX_MASK_WIDTH * MAX_MASK_WIDTH];

__global__ void conv_forward_kernel(float *output, const float *input, const float *mask, const int B, const int M, const int C, const int H, const int W, const int K,const int S)
// __global__ void conv_forward_kernel(float *output, const float *input, const float *mask, float *input_unrolled, const int B, const int M, const int C, const int H, const int W, const int K,const int S)
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

    __shared__ float subTile_input[TILE_WIDTH][TILE_WIDTH];
    __shared__ float subTile_mask[TILE_WIDTH][TILE_WIDTH];

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
    // #define mask_4d(i3, i2, i1, i0) mask[(i3) * (C * K * K) + (i2) * (K * K) + (i1) * (K) + i0]
    //out_4d(b, m, h_out, w_out)
    //in_4d(b, c, h, w)
    //mask_4d(m, c, p, q)

    //Every thread takes care of one output feature map element, which needs one column of unrolled input feature map.
    //So in a block, we can only unroll TILE_WIDTH columns of input feature map. 

    // Insert your GPU convolution kernel code here
    // int m = blockIdx.x; //The index of the output feature map.
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
    float output_val = 0;

    //Loop on each pair of tiles from A and B, 
    //Note that here (num_mask_columns-1) / TILE_WIDTH + 1 == ceil(num_mask_columns/TILE_WIDTH)
    for (int i=0; i < (num_mask_columns-1) / TILE_WIDTH + 1; i++){  //num_mask_columns == num_input_rows
    //load subtiles of A and B from the global memory to the shared memory of this block
        if (Row < num_mask_rows && i*TILE_WIDTH+tx < num_mask_columns){ //Boundary condition
            subTile_mask[ty][tx] = Mask_c[Row * num_mask_columns + i * TILE_WIDTH + tx];
        }else{
            subTile_mask[ty][tx] = 0;
        }
        if (i*TILE_WIDTH+ty < num_input_rows && Col < num_input_columns) { //Boundary condition condition
            int c = (i * TILE_WIDTH + ty) / (K * K);    //The channel index of the input feature map, not it is integer division
            int w_unroll = Col;             //The column index of the unrolled input feature map
            int h_out = Col / W_out;
            int w_out = Col % W_out;
            int w_base = c * K * K;
            int p = (i * TILE_WIDTH + ty - w_base) / K;
            int q = (i * TILE_WIDTH + ty - w_base) % K;
            // Row index of the unrolled matrix for the thread to write
            // the input element into for the current iteration
            int h_unroll = w_base + p*K + q;
            if ((h_out*S+p < H) && (w_out*S+q < W)){
                subTile_input[ty][tx] = in_4d(b, c, h_out*S + p, w_out*S + q);
            }else{
                subTile_input[ty][tx] = 0.0f;
            }
            // subTile_input[ty][tx] = uin_3d(b, (i * TILE_WIDTH + ty), Col);  //B[(i * TILE_WIDTH + ty) * num_input_columns + Col];
        }else{
            subTile_input[ty][tx] = 0;
        }
        __syncthreads();  //Synchronize the threads, we need to wait all threads finish loading the data
        //Use the subtiles to calculate the partial sum for the element, and accumulate into output_val
        if (Row < num_output_rows && Col < num_output_columns){ //Boundary condition
            for (int k=0; k<TILE_WIDTH; k++){
                output_val += subTile_mask[ty][k] * subTile_input[k][tx];
                // output_val += Mask_c[Row * num_mask_columns + i * TILE_WIDTH + k] * subTile_input[k][tx];
            }
        }
        __syncthreads();  //Synchronize the threads, we need to wait all threads finish computation, so that we can get the correct answer
    }

    if (Row < num_output_rows && Col < num_output_columns){ //Boundary condition
        output[b * (num_output_rows * num_output_columns) + Row * num_output_columns + Col] = output_val;  //Store the result element back
    }

    #undef out_4d
    #undef in_4d
    // #undef mask_4d
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

    const int X_grid = ceil((1.0 * H_out * W_out) / (1.0 * TILE_WIDTH));
    const int M_grid = ceil((1.0 * M) / (1.0 * TILE_WIDTH));    //Number of tiles in height for all output feature maps in an image

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
    // std::cout<<"Threads available in one image (ceiled): "<<X_grid*M_grid*TILE_WIDTH*TILE_WIDTH<<std::endl;
    // std::cout<<"Threads available in one image (not ceiled): "<<M*H_out*W_out<<std::endl;
    // std::cout<<"Unroll threads needed in one image = C*H_out*W_out: "<<C*H_out*W_out<<std::endl;
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
