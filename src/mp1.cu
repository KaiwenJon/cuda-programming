// MP 1
#include <wb.h>
#include <iostream>
#include <cuda_runtime.h>
#include <device_launch_parameters.h>

using namespace std;
__global__ void vecAdd(float *in1, float *in2, float *out, int len) {
  //@@ Insert code to implement vector addition here
  int id = (blockIdx.x * blockDim.x) + threadIdx.x;
  // printf("hello %d", id);
  if(id < len){
    out[id] = in1[id] + in2[id];
    // printf("out %2f\n", out[id]);
  }
}

int main(int argc, char **argv) {
  wbArg_t args;
  int inputLength;
  float *hostInput1;
  float *hostInput2;
  float *hostOutput;
  float *deviceInput1;
  float *deviceInput2;
  float *deviceOutput;
  args = wbArg_read(argc, argv);
  wbTime_start(Generic, "Importing data and creating memory on host");
  hostInput1 =
      (float *)wbImport(wbArg_getInputFile(args, 0), &inputLength);
  hostInput2 =
      (float *)wbImport(wbArg_getInputFile(args, 1), &inputLength);
  hostOutput = (float *)malloc(inputLength * sizeof(float));
  wbTime_stop(Generic, "Importing data and creating memory on host");
  wbLog(TRACE, "The input length is ", inputLength);

  wbTime_start(GPU, "Allocating GPU memory.");
  //@@ Allocate GPU memory here
  int total_bytes = inputLength * sizeof(float);
  cudaMalloc(&deviceInput1, total_bytes);
  cudaMalloc(&deviceInput2, total_bytes);
  cudaMalloc(&deviceOutput, total_bytes);

  wbTime_stop(GPU, "Allocating GPU memory.");

  wbTime_start(GPU, "Copying input memory to the GPU.");
  //@@ Copy memory to the GPU here
  cudaMemcpy(deviceInput1, hostInput1, total_bytes, cudaMemcpyHostToDevice);
  cudaMemcpy(deviceInput2, hostInput2, total_bytes, cudaMemcpyHostToDevice);

  wbTime_stop(GPU, "Copying input memory to the GPU.");

  //@@ Initialize the grid and block dimensions here
  int NUM_THREADS = 64;
  int NUM_BLOCKS = (int)ceil(inputLength / (float)NUM_THREADS);
  wbTime_start(Compute, "Performing CUDA computation");
  //@@ Launch the GPU Kernel here
  vecAdd<<<NUM_BLOCKS, NUM_THREADS>>>(deviceInput1, deviceInput2, deviceOutput, inputLength);
  cudaDeviceSynchronize();
  wbTime_stop(Compute, "Performing CUDA computation");

  wbTime_start(Copy, "Copying output memory to the CPU");
  //@@ Copy the GPU memory back to the CPU here
  cudaMemcpy(hostOutput, deviceOutput, total_bytes, cudaMemcpyDeviceToHost);
  wbTime_stop(Copy, "Copying output memory to the CPU");
  // for(int i=0; i<inputLength; i++){
  //   cout << hostInput1[i] << "\t" << hostInput2[i] << "\t" << hostOutput[i]<<endl;
  // }
  wbTime_start(GPU, "Freeing GPU Memory");
  //@@ Free the GPU memory here
  cudaFree(deviceInput1);
  cudaFree(deviceInput2);
  cudaFree(deviceOutput);

  wbTime_stop(GPU, "Freeing GPU Memory");

  wbSolution(args, hostOutput, inputLength);

  wbExport(wbArg_getOutputFile(args), hostOutput, inputLength);
  free(hostInput1);
  free(hostInput2);
  free(hostOutput);
  // ./bin/cuda -t vector -i ./inputs/data_mp1/1/input0.raw,./inputs/data_mp1/1/input1.raw -o ./outputs/data_mp1/1/output.raw -e ./inputs/data_mp1/1/output.raw 
  return 0;
}
