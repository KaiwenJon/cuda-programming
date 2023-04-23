#include <cmath>
#include <iostream>
#include "gpu-new-forward.h"

#define TILE_WIDTH 18
#define MASK_WIDTH 7
#define RADIUS 3
#define CHANNEL_NUM 4
#define INPUT_TILE_WIDTH (TILE_WIDTH + MASK_WIDTH - 1)
#define STREAM_NUM 10
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
        for(int c=0; c<Channel; c++){
            sharedTile[ty][tx][c] = in_4d(batchDataIdx, c, HdataIdx, WdataIdx);
        }
    }
    else{
        for(int c=0; c<Channel; c++){
            sharedTile[ty][tx][c] = 0.0f;
        }
    }

    __syncthreads();

    // only tx=0~TILEWIDTH-1, ty=0~TILEWIDTH-1 , tz=0are computing.
    // each thread computes channel values and add them up

    if(tx < TILE_WIDTH && ty < TILE_WIDTH && tz == 0){
        float val = 0.0f;
        for(int c=0; c<Channel; c++){
            for(int k1=0; k1<K; k1++){
                for(int k2=0; k2<K; k2++){
                    val += sharedTile[ty+k1][tx+k2][c] * mask_4d(mapOutIdx, c, k1, k2);
                }
            }
        }
        if(HdataIdx < Height_out && WdataIdx < Width_out) {
            out_4d(batchDataIdx, mapOutIdx, HdataIdx, WdataIdx) = val;
        }
    }

    #undef out_4d
    #undef in_4d
    #undef mask_4d
}


__host__ void GPUInterface::conv_forward_gpu(float *device_output, const float *device_input, const float *device_mask, const int Batch, const int Map_out, const int Channel, const int Height, const int Width, const int K)
{
    return;
}

	
__host__ void GPUInterface::conv_forward_gpu_prolog(const float *host_output, const float *host_input, const float *host_mask, float **device_output_ptr, float **device_input_ptr, float **device_mask_ptr, const int Batch, const int Map_out, const int Channel, const int Height, const int Width, const int K)
{
    const int Height_out = Height - K + 1;
    const int Width_out = Width - K + 1;
    float *host_output_dummy = (float*)host_output;

    int sizeInput = Batch * Channel * Height * Width * sizeof(float);
    int sizeOutput = Batch * Map_out * Height_out * Width_out * sizeof(float);
    int sizeMask = Channel * Map_out * K * K * sizeof(float);
    
    cudaMalloc((void**) device_input_ptr, sizeInput);
    cudaMalloc((void**) device_output_ptr, sizeOutput);
    cudaMemcpyToSymbol(Mc, host_mask, sizeMask);
    
    int H_grid = (int)ceil(Height_out / (float)TILE_WIDTH);
    int W_grid = (int)ceil(Width_out / (float)TILE_WIDTH);
    dim3 dim_block(INPUT_TILE_WIDTH, INPUT_TILE_WIDTH, 1);
    dim3 dim_grid(Map_out, H_grid * W_grid, Batch/STREAM_NUM);

    int input_batch_size = Batch * Channel * Height * Width / STREAM_NUM;
    int output_batch_size = Batch * Map_out * Height_out * Width_out / STREAM_NUM;

    cudaStream_t stream[STREAM_NUM];
    for(int i=0; i<STREAM_NUM; i++){
        cudaStreamCreate(&stream[i]);
    }
    for(int i=0; i<STREAM_NUM; i++){
        int input_offset = input_batch_size * i;
        int output_offset = output_batch_size * i;
        cudaMemcpyAsync((*device_input_ptr) + input_offset, 
                        host_input + input_offset,
                        input_batch_size * sizeof(float),
                        cudaMemcpyHostToDevice,
                        stream[i]);
        conv_forward_kernel<<<dim_grid, dim_block, 0, stream[i]>>>(
            (*device_output_ptr) + output_offset, 
            (*device_input_ptr) + input_offset,
            (*device_mask_ptr),
            Batch, Map_out, Channel , Height, Width, K);
        cudaMemcpyAsync(host_output_dummy + output_offset,
                        (*device_output_ptr) + output_offset,
                        output_batch_size * sizeof(float),
                        cudaMemcpyDeviceToHost,
                        stream[i]);
    }
    cudaDeviceSynchronize();
    for(int i=0; i<STREAM_NUM; i++){
        cudaStreamDestroy(stream[i]);
    }
    // Free device memory
    cudaFree(device_input_ptr);
    cudaFree(device_output_ptr);
    cudaFree(device_mask_ptr);
}



__host__ void GPUInterface::conv_forward_gpu_epilog(float *host_output, float *device_output, float *device_input, float *device_mask, const int Batch, const int Map_out, const int Channel, const int Height, const int Width, const int K)
{
    return;
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
