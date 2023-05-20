#include <wb.h>


#define wbCheck(stmt)                                                     \
do {                                                                    \
  cudaError_t err = stmt;                                               \
  if (err != cudaSuccess) {                                             \
    wbLog(ERROR, "CUDA error: ", cudaGetErrorString(err));              \
    wbLog(ERROR, "Failed to run stmt ", #stmt);                         \
    return -1;                                                          \
  }                                                                     \
} while (0)

//@@ Define any useful program-wide constants here

#define TILE_WIDTH 5
#define MASK_WIDTH 3
#define INPUT_TILE_WIDTH (TILE_WIDTH + MASK_WIDTH - 1)
#define RADIUS 1

//@@ Define constant memory for device kernel here

__constant__ float Mc[MASK_WIDTH][MASK_WIDTH][MASK_WIDTH];

__global__ void conv3d(float *input, float *output, const int z_size,
                       const int y_size, const int x_size) {
  //@@ Insert kernel code here
  // Strategy 2!!!
  __shared__ float sharedTile [INPUT_TILE_WIDTH][INPUT_TILE_WIDTH][INPUT_TILE_WIDTH];
  int tx = threadIdx.x;
  int ty = threadIdx.y;
  int tz = threadIdx.z;

  int x_in = blockIdx.x * TILE_WIDTH + tx - RADIUS;
  int y_in = blockIdx.y * TILE_WIDTH + ty - RADIUS;
  int z_in = blockIdx.z * TILE_WIDTH + tz - RADIUS;
  if(x_in >= 0 && x_in < x_size &&
     y_in >= 0 && y_in < y_size &&
     z_in >= 0 && z_in < z_size){
    sharedTile[ty][tx][tz] = input[x_size*y_size*z_in + x_size*y_in + x_in];
  }
  else{
    sharedTile[ty][tx][tz] = 0.0;
  }

  __syncthreads();
  float p_value = 0.0;
  if(tx < TILE_WIDTH && ty < TILE_WIDTH && tz < TILE_WIDTH){
    for(int i=0; i<MASK_WIDTH; i++){
      for(int j=0; j<MASK_WIDTH; j++){
        for(int k=0; k<MASK_WIDTH; k++){
          p_value += sharedTile[ty+i][tx+j][tz+k] * Mc[i][j][k];
        }
      }
    }
  }
  __syncthreads();
  int x_out = blockIdx.x * TILE_WIDTH + tx;
  int y_out = blockIdx.y * TILE_WIDTH + ty;
  int z_out = blockIdx.z * TILE_WIDTH + tz;
  if(tx < TILE_WIDTH && ty < TILE_WIDTH && tz < TILE_WIDTH){
    if(x_out < x_size && y_out < y_size && z_out < z_size){
      output[x_size*y_size*z_out + x_size*y_out + x_out] = p_value;
    }
  }
}

int main(int argc, char *argv[]) {
  wbArg_t args;
  int z_size;
  int y_size;
  int x_size;
  int inputLength, kernelLength;
  float *hostInput;
  float *hostKernel;
  float *hostOutput;
  float *deviceInput;
  float *deviceOutput;

  args = wbArg_read(argc, argv);

  // Import data
  hostInput = (float *)wbImport(wbArg_getInputFile(args, 0), &inputLength);
  hostKernel =
      (float *)wbImport(wbArg_getInputFile(args, 1), &kernelLength);
  hostOutput = (float *)malloc(inputLength * sizeof(float));

  // First three elements are the input dimensions
  z_size = hostInput[0];
  y_size = hostInput[1];
  x_size = hostInput[2];
  wbLog(TRACE, "The input size is ", z_size, "x", y_size, "x", x_size);
  assert(z_size * y_size * x_size == inputLength - 3);
  assert(kernelLength == 27);

  wbTime_start(GPU, "Doing GPU Computation (memory + compute)");

  wbTime_start(GPU, "Doing GPU memory allocation");
  //@@ Allocate GPU memory here
  // Recall that inputLength is 3 elements longer than the input data
  // because the first  three elements were the dimensions
  int bytesInput = (inputLength - 3) * sizeof(float);
  int bytesOutput = (inputLength - 3) * sizeof(float);
  cudaMalloc(&deviceInput, bytesInput);
  cudaMalloc(&deviceOutput, bytesOutput);

  wbTime_stop(GPU, "Doing GPU memory allocation");

  wbTime_start(Copy, "Copying data to the GPU");
  //@@ Copy input and kernel to GPU here
  // Recall that the first three elements of hostInput are dimensions and
  // do
  // not need to be copied to the gpu
  cudaMemcpy(deviceInput, hostInput+3, bytesInput, cudaMemcpyHostToDevice);
  
  float Mask[MASK_WIDTH][MASK_WIDTH][MASK_WIDTH];
  for(unsigned int i=0; i<MASK_WIDTH; i++){
    for(unsigned int j=0; j<MASK_WIDTH; j++){
      for(unsigned int k=0; k<MASK_WIDTH; k++){
        Mask[i][j][k] = hostKernel[MASK_WIDTH*MASK_WIDTH*k + MASK_WIDTH*j + i];
      }
    }
  }
  cudaMemcpyToSymbol(Mc, Mask, MASK_WIDTH*MASK_WIDTH*MASK_WIDTH*sizeof(float));


  wbTime_stop(Copy, "Copying data to the GPU");

  wbTime_start(Compute, "Doing the computation on the GPU");
  //@@ Initialize grid and block dimensions here

  int NUM_THREADS = INPUT_TILE_WIDTH;
  int NUM_BLOCKS_Z = (int)ceil(z_size / (float)TILE_WIDTH);
  int NUM_BLOCKS_Y = (int)ceil(y_size / (float)TILE_WIDTH);
  int NUM_BLOCKS_X = (int)ceil(x_size / (float)TILE_WIDTH);

  dim3 dim_block(NUM_THREADS, NUM_THREADS, NUM_THREADS);
  dim3 dim_grid(NUM_BLOCKS_X, NUM_BLOCKS_Y, NUM_BLOCKS_Z);

  //@@ Launch the GPU kernel here
  conv3d<<<dim_grid, dim_block>>>(deviceInput, deviceOutput, z_size, y_size, x_size);
  cudaDeviceSynchronize();
  wbTime_stop(Compute, "Doing the computation on the GPU");

  wbTime_start(Copy, "Copying data from the GPU");
  //@@ Copy the device memory back to the host here
  // Recall that the first three elements of the output are the dimensions
  // and should not be set here (they are set below)
  cudaMemcpy(hostOutput+3, deviceOutput, bytesOutput, cudaMemcpyDeviceToHost);
  wbTime_stop(Copy, "Copying data from the GPU");

  wbTime_stop(GPU, "Doing GPU Computation (memory + compute)");

  // Set the output dimensions for correctness checking
  hostOutput[0] = z_size;
  hostOutput[1] = y_size;
  hostOutput[2] = x_size;
  wbSolution(args, hostOutput, inputLength);

  // Free device memory
  cudaFree(deviceInput);
  cudaFree(deviceOutput);

  // Free host memory
  free(hostInput);
  free(hostOutput);
  return 0;
}

// ./bin/cuda -t vector -i ./inputs/MP4/data/0/input.dat,./inputs/MP4/data/0/kernel.dat -o ./outputs/MP4/data/0/output.dat -e ./inputs/MP4/data/0/output.dat
