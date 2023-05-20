
#include <cmath>
#include <iostream>
#include "gpu-new-forward.h"

#define TILE_WIDTH 10
#define MASK_WIDTH 7
#define RADIUS 3
#define CHANNEL_NUM 4
#define INPUT_TILE_WIDTH (TILE_WIDTH + MASK_WIDTH - 1)
__constant__ float Mc[4*16*7*7]; // channel * map * K * K

__global__ void conv_forward_kernel(float *output, const float *input, const float *mask, const int Batch, const int Map_out, const int Channel, const int Height, const int Width, const int K)
{
    /*
    Modify this function to implement the forward pass described in Chapter 16.
    We have added an additional dimension to the tensors to support an entire mini-batch
    The goal here is to be correct AND fast.

    Function paramter definitions:
    output - output
    input - input
    mask - convolution kernel
    Batch - batch_size (number of images in x)
    Map_out - number of output feature maps
    Channel - number of input feature maps
    Height - input height dimension
    Width - input width dimension
    K - kernel height and width (K x K)
    */

    const int Height_out = Height - K + 1;
    const int Width_out = Width - K + 1;
    (void)Height_out; // silence declared but never referenced warning. remove this line when you start working
    (void)Width_out; // silence declared but never referenced warning. remove this line when you start working

    // We have some nice #defs for you below to simplify indexing. Feel free to use them, or create your own.
    // An example use of these macros:
    // float a = in_4d(0,0,0,0)
    // out_4d(0,0,0,0) = a

    #define out_4d(i3, i2, i1, i0) output[(i3) * (Map_out * Height_out * Width_out) + (i2) * (Height_out * Width_out) + (i1) * (Width_out) + i0]
    #define in_4d(i3, i2, i1, i0) input[(i3) * (Channel * Height * Width) + (i2) * (Height * Width) + (i1) * (Width) + i0]
    #define mask_4d(i3, i2, i1, i0) Mc[(i3) * (Channel * K * K) + (i2) * (K * K) + (i1) * (K) + i0]

    // Insert your GPU convolution kernel code here
    __shared__ float sharedTile [INPUT_TILE_WIDTH][INPUT_TILE_WIDTH][CHANNEL_NUM];
    int batchDataIdx = blockIdx.z;
    int mapOutIdx = blockIdx.x;

    int W_grid = (int)ceil(Width_out / (float)TILE_WIDTH);
    int HgridIdx = blockIdx.y / (W_grid);
    int WgridIdx = blockIdx.y % (W_grid);
    int HdataIdx = HgridIdx * TILE_WIDTH + threadIdx.y;
    int WdataIdx = WgridIdx * TILE_WIDTH + threadIdx.x;

    // load shared mem
    int tx = threadIdx.x;
    int ty = threadIdx.y;
    int tz = threadIdx.z;
    if(WdataIdx < Width && HdataIdx < Height){
        sharedTile[ty][tx][tz] = in_4d(batchDataIdx, tz, HdataIdx, WdataIdx);
    }
    else{
        sharedTile[ty][tx][tz] = 0.0;
    }
    
    // if(tx < TILE_WIDTH && ty < TILE_WIDTH){
    //     if(HdataIdx < Height_out && WdataIdx < Width_out) {
    //         out_4d(batchDataIdx, mapOutIdx, HdataIdx, WdataIdx) = 0.0f;
    //     }
    // }
    __syncthreads();

    // only tx=0~TILEWIDTH-1, ty=0~TILEWIDTH-1 , tz=0～channel-1 are computing.
    // each thread computes channel values and add them up
    
    if(tx < TILE_WIDTH && ty < TILE_WIDTH){
        float val = 0.0f;
        for(int k1=0; k1<K; k1++){
            for(int k2=0; k2<K; k2++){
                val += sharedTile[ty+k1][tx+k2][tz] * mask_4d(mapOutIdx, tz, k1, k2);
            }
        }
        if(HdataIdx < Height_out && WdataIdx < Width_out) {
            atomicAdd(&(out_4d(batchDataIdx, mapOutIdx, HdataIdx, WdataIdx)), val);
        }
    }



    #undef out_4d
    #undef in_4d
    #undef mask_4d
}


__host__ void GPUInterface::conv_forward_gpu(float *device_output, const float *device_input, const float *device_mask, const int Batch, const int Map_out, const int Channel, const int Height, const int Width, const int K)
{
    // Set the kernel dimensions and call the kernel
    const int Height_out = Height - K + 1;
    const int Width_out = Width - K + 1;
    int H_grid = (int)ceil(Height_out / (float)TILE_WIDTH);
    int W_grid = (int)ceil(Width_out / (float)TILE_WIDTH);

    dim3 dim_block(INPUT_TILE_WIDTH, INPUT_TILE_WIDTH, Channel);
    dim3 dim_grid(Map_out, H_grid * W_grid, Batch);
    conv_forward_kernel<<<dim_grid, dim_block>>>(device_output, device_input, device_mask, Batch, Map_out, Channel , Height, Width, K);
}

	
__host__ void GPUInterface::conv_forward_gpu_prolog(const float *host_output, const float *host_input, const float *host_mask, float **device_output_ptr, float **device_input_ptr, float **device_mask_ptr, const int Batch, const int Map_out, const int Channel, const int Height, const int Width, const int K)
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
    const int Height_out = Height - K + 1;
    const int Width_out = Width - K + 1;

    int sizeInput = Batch * Channel * Height * Width * sizeof(float);
    int sizeOutput = Batch * Map_out * Height_out * Width_out * sizeof(float);
    int sizeMask = Channel * Map_out * K * K * sizeof(float);

    cudaMalloc((void**) device_input_ptr, sizeInput);
    cudaMalloc((void**) device_output_ptr, sizeOutput);
    // cudaMalloc((void**) device_mask_ptr, sizeMask);

    cudaMemcpy(*device_input_ptr, host_input, sizeInput, cudaMemcpyHostToDevice);
    cudaMemcpy(*device_output_ptr, host_output, sizeOutput, cudaMemcpyHostToDevice);
    cudaMemset(*device_output_ptr, 0.0, sizeOutput);

    // cudaMemcpy(*device_mask_ptr, host_mask, sizeMask, cudaMemcpyHostToDevice);
    cudaMemcpyToSymbol(Mc, host_mask, sizeMask);

}



__host__ void GPUInterface::conv_forward_gpu_epilog(float *host_output, float *device_output, float *device_input, float *device_mask, const int Batch, const int Map_out, const int Channel, const int Height, const int Width, const int K)
{
    // Copy the output back to host
    const int Height_out = Height - K + 1;
    const int Width_out = Width - K + 1;

    int sizeOutput = Batch * Map_out * Height_out * Width_out * sizeof(float);
    cudaMemcpy(host_output, device_output, sizeOutput ,cudaMemcpyDeviceToHost);
    // Free device memory
    cudaFree(device_input);
    cudaFree(device_output);
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
