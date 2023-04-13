// MP 5.1 Reduction
// Given a list of length n
// Output its sum = lst[0] + lst[1] + ... + lst[n-1];

#include <wb.h>

#define BLOCK_SIZE 64 //@@ You can change this

#define wbCheck(stmt)                                                     \
  do {                                                                    \
    cudaError_t err = stmt;                                               \
    if (err != cudaSuccess) {                                             \
      wbLog(ERROR, "Failed to run stmt ", #stmt);                         \
      wbLog(ERROR, "Got CUDA error ...  ", cudaGetErrorString(err));      \
      return -1;                                                          \
    }                                                                     \
  } while (0)
  
__global__ void reduction(float *input, float *output, int len) {
  //@@ Load a segment of the input vector into shared memory
  //@@ Traverse the reduction tree
  //@@ Write the computed sum of the block to the output vector at the
  //@@ correct index
  printf("Wju!");
  __shared__ float partialSum[BLOCK_SIZE*2];
  // load data to shared mem
  int tx = threadIdx.x;
  int firstIdx = blockIdx.x * (blockDim.x*2) + threadIdx.x;
  int secondIdx = firstIdx + blockDim.x;
  if(firstIdx < len){
    partialSum[tx] = input[firstIdx];
  }
  else{
    partialSum[tx] = 0;
  }
  if(secondIdx < len){
    partialSum[tx + blockDim.x] = input[secondIdx];
  }
  else{
    partialSum[tx + blockDim.x] = 0;
  }
  for(int stride=blockDim.x; stride>=1; stride/=2){
    __syncthreads();
    if(tx < stride){
      partialSum[tx] += partialSum[tx + stride];
    }
    printf("Layer%d, first value:%f", stride, partialSum[0]);
  }
  if(tx == 0){
    output[blockIdx.x] = partialSum[0];
  }
}

int main(int argc, char **argv) {
  int ii;
  wbArg_t args;
  float *hostInput;  // The input 1D list
  float *hostOutput; // The output list
  float *deviceInput;
  float *deviceOutput;
  int numInputElements;  // number of elements in the input list
  int numOutputElements; // number of elements in the output list

  args = wbArg_read(argc, argv);

  wbTime_start(Generic, "Importing data and creating memory on host");
  hostInput =
      (float *)wbImport(wbArg_getInputFile(args, 0), &numInputElements);

  numOutputElements = (numInputElements - 1) / (BLOCK_SIZE << 1) + 1;
  hostOutput = (float *)malloc(numOutputElements * sizeof(float));

  wbTime_stop(Generic, "Importing data and creating memory on host");

  wbLog(TRACE, "The number of input elements in the input is ",
        numInputElements);
  wbLog(TRACE, "The number of output elements in the input is ",
        numOutputElements);

  wbTime_start(GPU, "Allocating GPU memory.");
  //@@ Allocate GPU memory here
  int bytesInput = numInputElements * sizeof(float);
  int bytesOutput = numOutputElements * sizeof(float);
  cudaMalloc(&deviceInput, bytesInput);
  cudaMalloc(&deviceOutput, bytesOutput);

  wbTime_stop(GPU, "Allocating GPU memory.");

  wbTime_start(GPU, "Copying input memory to the GPU.");
  //@@ Copy memory to the GPU here
  cudaMemcpy(deviceInput, hostInput, bytesInput, cudaMemcpyHostToDevice);

  wbTime_stop(GPU, "Copying input memory to the GPU.");
  //@@ Initialize the grid and block dimensions here
  int NUM_THREADS = BLOCK_SIZE;
  int NUM_BLOCKS = (int)ceil(numInputElements / (((float)BLOCK_SIZE)*2));
  printf("NUMBOLCOKS:%d\n", NUM_BLOCKS);
  dim3 dim_block(NUM_THREADS, 1, 1);
  dim3 dim_grid(NUM_BLOCKS, 1, 1);


  wbTime_start(Compute, "Performing CUDA computation");
  //@@ Launch the GPU Kernel here
  reduction<<<dim_grid, dim_block>>>(deviceInput, deviceOutput, numInputElements);

  cudaDeviceSynchronize();
  wbTime_stop(Compute, "Performing CUDA computation");

  wbTime_start(Copy, "Copying output memory to the CPU");
  //@@ Copy the GPU memory back to the CPU here
  cudaMemcpy(hostOutput, deviceOutput, bytesOutput, cudaMemcpyDeviceToHost);

  wbTime_stop(Copy, "Copying output memory to the CPU");

  /***********************************************************************
   * Reduce output vector on the host
   * NOTE: One could also perform the reduction of the output vector
   * recursively and support any size input.
   * For simplicity, we do not require that for this lab!
   ***********************************************************************/
  for (ii = 1; ii < numOutputElements; ii++) {
    printf("hostoutput%f\n", hostOutput[ii]);
    hostOutput[0] += hostOutput[ii];
  }

  wbTime_start(GPU, "Freeing GPU Memory");
  //@@ Free the GPU memory here
  cudaFree(deviceInput);
  cudaFree(deviceOutput);
  wbTime_stop(GPU, "Freeing GPU Memory");

  wbSolution(args, hostOutput, 1);

  free(hostInput);
  free(hostOutput);

  return 0;
}
//./bin/cuda -t vector -i ./inputs/MP5.1/data/0/input0.raw, -o ./outputs/MP5.1/data/0/output.raw -e ./inputs/MP5.1/data/0/output.raw