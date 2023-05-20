// MP Scan
// Given a list (lst) of length n
// Output its prefix sum = {lst[0], lst[0] + lst[1], lst[0] + lst[1] + ...
// +
// lst[n-1]}

#include <wb.h>

#define BLOCK_SIZE 256 //@@ You can change this

#define wbCheck(stmt)                                                     \
  do {                                                                    \
    cudaError_t err = stmt;                                               \
    if (err != cudaSuccess) {                                             \
      wbLog(ERROR, "Failed to run stmt ", #stmt);                         \
      wbLog(ERROR, "Got CUDA error ...  ", cudaGetErrorString(err));      \
      return -1;                                                          \
    }                                                                     \
  } while (0)

__global__ void scan(float *input, float *output, int len) {
  //@@ Modify the body of this function to complete the functionality of
  //@@ the scan on the device
  //@@ You may need multiple kernel calls; write your kernels before this
  //@@ function and call them from the host
  __shared__ float partialSum[2*BLOCK_SIZE];
  int tx = threadIdx.x;
  int firstIdx = blockIdx.x * (BLOCK_SIZE*2) + threadIdx.x*2;
  int secondIdx = firstIdx + 1;
  if(firstIdx < len){
    partialSum[2*tx] = input[firstIdx];
  }
  else{
    partialSum[2*tx] = 0;
  }
  if(secondIdx < len){
    partialSum[2*tx + 1] = input[secondIdx];
  }
  else{
    partialSum[2*tx + 1] = 0;
  }
  int stride = 1;
  while(stride <= 2*BLOCK_SIZE){
    int index = (tx + 1)*stride*2 - 1;
    if(index < 2*BLOCK_SIZE && (index-stride) >= 0){
      partialSum[index] += partialSum[index-stride];
    }
    stride *= 2;
    __syncthreads();
  }

  stride = BLOCK_SIZE/2;
  while(stride > 0){
    __syncthreads();
    int index = (tx + 1)*stride*2 - 1;
    if(index + stride < 2*BLOCK_SIZE){
      partialSum[index+stride] += partialSum[index];
    }
    stride = stride / 2;
  }

  __syncthreads();
  if(firstIdx < len){
    output[firstIdx] = partialSum[2*tx];
  }
  if(secondIdx < len){
    output[secondIdx] = partialSum[2*tx + 1];
  }
}

__global__ void addScannedBlock(float* output, float* tempSum, int len){
  // inplace calculation
  int index = blockIdx.x * blockDim.x + threadIdx.x;
  if(blockIdx.x > 0){
    if(index < len){
      output[index] += tempSum[blockIdx.x-1];
    }
  }
  
  
}

int main(int argc, char **argv) {
  wbArg_t args;
  float *hostInput;  // The input 1D list
  float *hostOutput; // The output list
  float *deviceInput;
  float *deviceOutput;
  int numElements; // number of elements in the list
  

  args = wbArg_read(argc, argv);

  wbTime_start(Generic, "Importing data and creating memory on host");
  hostInput = (float *)wbImport(wbArg_getInputFile(args, 0), &numElements);
  hostOutput = (float *)malloc(numElements * sizeof(float));
  wbTime_stop(Generic, "Importing data and creating memory on host");

  wbLog(TRACE, "The number of input elements in the input is ",
        numElements);

  wbTime_start(GPU, "Allocating GPU memory.");
  wbCheck(cudaMalloc((void **)&deviceInput, numElements * sizeof(float)));
  wbCheck(cudaMalloc((void **)&deviceOutput, numElements * sizeof(float)));
  wbTime_stop(GPU, "Allocating GPU memory.");

  wbTime_start(GPU, "Clearing output memory.");
  wbCheck(cudaMemset(deviceOutput, 0, numElements * sizeof(float)));
  wbTime_stop(GPU, "Clearing output memory.");

  wbTime_start(GPU, "Copying input memory to the GPU.");
  wbCheck(cudaMemcpy(deviceInput, hostInput, numElements * sizeof(float),
                     cudaMemcpyHostToDevice));
  wbTime_stop(GPU, "Copying input memory to the GPU.");

  //@@ Initialize the grid and block dimensions here
  int NUM_THREADS = BLOCK_SIZE;
  int NUM_BLOCKS = (int)ceil(numElements / (float)(BLOCK_SIZE*2));

  int final_blocks = NUM_BLOCKS;
  int final_threads = BLOCK_SIZE*2;

  dim3 dim_block(NUM_THREADS, 1, 1);
  dim3 dim_grid(NUM_BLOCKS, 1, 1);
  printf("Stage1, BlockSize: %d, num_blocks: %d\n", NUM_THREADS, NUM_BLOCKS);
  wbTime_start(Compute, "Performing CUDA computation");
  //@@ Modify this to complete the functionality of the scan
  //@@ on the deivce
  scan<<<dim_grid, dim_block>>>(deviceInput, deviceOutput, numElements);
  wbCheck(cudaMemcpy(hostOutput, deviceOutput, numElements * sizeof(float),
                     cudaMemcpyDeviceToHost));

  // for(int i=0; i<numElements; i++){
  //   if(i % 256 == 0){
  //     printf("Result %d: %f\n", i, hostOutput[i]);
  //   }
  // }
  float* hostTempSum = (float *)malloc(NUM_BLOCKS * sizeof(float));
  for(int i=0; i<NUM_BLOCKS; i++){
    if(i == NUM_BLOCKS-1){
      hostTempSum[i] = hostOutput[numElements-1];
    }
    else{
      hostTempSum[i] = hostOutput[BLOCK_SIZE*2*(i+1)-1];  
    }
    // printf("blocksum %d: %f\n", i, hostTempSum[i]);
  }
  float* deviceTempSum;
  wbCheck(cudaMalloc((void **)&deviceTempSum, NUM_BLOCKS * sizeof(float)));
  wbCheck(cudaMemcpy(deviceTempSum, hostTempSum, NUM_BLOCKS * sizeof(float),
                     cudaMemcpyHostToDevice));
  float* deviceTempSumOut;
  wbCheck(cudaMalloc((void **)&deviceTempSumOut, NUM_BLOCKS * sizeof(float)));

  // TODO: consider arbitrary length of host input, assign 
  int new_BLOCK_SIZE = ceil((float)NUM_BLOCKS/2); 
  #define BLOCK_SIZE new_BLOCK_SIZE //@@ You can change this
  dim3 dim_block_add(BLOCK_SIZE, 1, 1);
  dim3 dim_grid_add(1, 1, 1);
  printf("Stage2, BlockSize: %d, num_elements: %d\n", new_BLOCK_SIZE, NUM_BLOCKS);
  scan<<<dim_grid_add, dim_block_add>>>(deviceTempSum, deviceTempSumOut, NUM_BLOCKS);
  wbCheck(cudaMemcpy(hostTempSum, deviceTempSumOut, NUM_BLOCKS * sizeof(float),
                     cudaMemcpyDeviceToHost));
  // for(int i=0;i<NUM_BLOCKS; i++){
  //   printf("accum tempsum %d, %f\n", i, hostTempSum[i]);
  // }
  dim3 dim_block_final(final_threads, 1, 1);
  dim3 dim_grid_final(final_blocks, 1, 1);
  printf("Stage3, BlockSize: %d, num_blocks: %d\n", final_threads, final_blocks);
  
  addScannedBlock<<<dim_grid_final, dim_block_final>>>(deviceOutput, deviceTempSumOut, numElements);
  cudaDeviceSynchronize();
  wbTime_stop(Compute, "Performing CUDA computation");
  
  wbTime_start(Copy, "Copying output memory to the CPU");
  wbCheck(cudaMemcpy(hostOutput, deviceOutput, numElements * sizeof(float),
  cudaMemcpyDeviceToHost));
  // for(int i=0; i<numElements; i++){
  //   if(i % 256 == 0){
  //     printf("Result %d: %f\n", i, hostOutput[i]);
  //   }
  // }
  wbTime_stop(Copy, "Copying output memory to the CPU");

  wbTime_start(GPU, "Freeing GPU Memory");
  cudaFree(deviceInput);
  cudaFree(deviceOutput);
  cudaFree(deviceTempSum);
  cudaFree(deviceTempSumOut);
  wbTime_stop(GPU, "Freeing GPU Memory");

  wbSolution(args, hostOutput, numElements);

  free(hostInput);
  free(hostOutput);
  free(hostTempSum);

  return 0;
}
