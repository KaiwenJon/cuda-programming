
#include <wb.h>

#include <iostream>
#include <cuda_runtime.h>
#include <device_launch_parameters.h>

using namespace std;

#define TILE_WIDTH 2
#define wbCheck(stmt)                                                     \
  do {                                                                    \
    cudaError_t err = stmt;                                               \
    if (err != cudaSuccess) {                                             \
      wbLog(ERROR, "Failed to run stmt ", #stmt);                         \
      wbLog(ERROR, "Got CUDA error ...  ", cudaGetErrorString(err));      \
      return -1;                                                          \
    }                                                                     \
  } while (0)

// Compute C = A * B
__global__ void matrixMultiply(float *A, float *B, float *C, int numARows,
                               int numAColumns, int numBRows,
                               int numBColumns, int numCRows,
                               int numCColumns) {

  __shared__ float TILE_A[TILE_WIDTH][TILE_WIDTH];
  __shared__ float TILE_B[TILE_WIDTH][TILE_WIDTH];
  //@@ Insert code to implement matrix multiplication here
  int ty = threadIdx.y;
  int tx = threadIdx.x;
  int ROW = blockIdx.y * blockDim.y + threadIdx.y;
  int COL = blockIdx.x * blockDim.x + threadIdx.x;
  if(ROW >= numCRows || COL >= numCColumns){
    return;
  }
  float p_value = 0;
  // Loop over tile across k(numAColumns)
  int n_TILE = (int)ceil(numAColumns / (float)TILE_WIDTH);
  for(int t=0; t<n_TILE; t++){
    TILE_A[ty][tx] = A[ROW * numAColumns + t*TILE_WIDTH+tx];
    TILE_B[ty][tx] = B[(t*TILE_WIDTH+ty)*numBColumns + COL];
    // if(ROW <=2 && COL <= 2){
    //     printf("Building...TILE_A[%d][%d] %f, TILE_B[%d][%d] %f\n", ty, tx, TILE_A[ty][tx], ty, tx, TILE_B[ty][tx]);
    // }
    __syncthreads();

    for(int i=0; i<TILE_WIDTH; i++){
        // if(ROW == 0 && COL == 0){
        //     printf("tile i: %d\n", i);
        //     printf("TILE_A[%d][%d] %f, TILE_B[%d][%d] %f, p_value %f\n", ty, i, TILE_A[ty][i], i, tx, TILE_B[i][tx],p_value);
        // }
        p_value += TILE_A[ty][i] * TILE_B[i][tx];
    }
    __syncthreads();
    // if(ROW == 0 && COL == 0){
    //     printf("Next Tile\n");
    // }
  }
  C[ROW * numCColumns + COL] = p_value;
}

int main(int argc, char **argv) {
  wbArg_t args;
  float *hostA; // The A matrix
  float *hostB; // The B matrix
  float *hostC; // The output C matrix
  float *deviceA;
  float *deviceB;
  float *deviceC;
  int numARows;    // number of rows in the matrix A
  int numAColumns; // number of columns in the matrix A
  int numBRows;    // number of rows in the matrix B
  int numBColumns; // number of columns in the matrix B
  int numCRows;    // number of rows in the matrix C (you have to set this)
  int numCColumns; // number of columns in the matrix C (you have to set
                   // this)

  args = wbArg_read(argc, argv);

  wbTime_start(Generic, "Importing data and creating memory on host");
  hostA = (float *)wbImport(wbArg_getInputFile(args, 0), &numARows,
                            &numAColumns);
  hostB = (float *)wbImport(wbArg_getInputFile(args, 1), &numBRows,
                            &numBColumns);
  //@@ Set numCRows and numCColumns
  numCRows = numARows;
  numCColumns = numBColumns;
  //@@ Allocate the hostC matrix
  hostC = (float *)malloc(numCRows * numCColumns * sizeof(float));
  wbTime_stop(Generic, "Importing data and creating memory on host");

  wbLog(TRACE, "The dimensions of A are ", numARows, " x ", numAColumns);
  wbLog(TRACE, "The dimensions of B are ", numBRows, " x ", numBColumns);

  wbTime_start(GPU, "Allocating GPU memory.");
  //@@ Allocate GPU memory here
  int bytesA = numARows * numAColumns * sizeof(float);
  int bytesB = numBRows * numBColumns * sizeof(float);
  int bytesC = numCRows * numCColumns * sizeof(float);
  cudaMalloc(&deviceA, bytesA);
  cudaMalloc(&deviceB, bytesB);
  cudaMalloc(&deviceC, bytesC);
  wbTime_stop(GPU, "Allocating GPU memory.");

  wbTime_start(GPU, "Copying input memory to the GPU.");
  //@@ Copy memory to the GPU here

  cudaMemcpy(deviceA, hostA, bytesA, cudaMemcpyHostToDevice);
  cudaMemcpy(deviceB, hostB, bytesB, cudaMemcpyHostToDevice);

  wbTime_stop(GPU, "Copying input memory to the GPU.");

  //@@ Initialize the grid and block dimensions here
  int NUM_THREADS = TILE_WIDTH;// 16; // per dimension
  int NUM_BLOCKS_ROW = (int)ceil(numCRows / (float)NUM_THREADS); // per dimension
  int NUM_BLOCKS_COL = (int)ceil(numCColumns / (float)NUM_THREADS);
  // cout << NUM_BLOCKS_COL << " " << NUM_BLOCKS_ROW << endl;
  // cout << ceil(numCColumns/16.0) << " " <<  ceil(numCRows/16.0) << endl;
  dim3 dim_block(NUM_THREADS, NUM_THREADS, 1);
  dim3 dim_grid(NUM_BLOCKS_COL, NUM_BLOCKS_ROW, 1);
  wbTime_start(Compute, "Performing CUDA computation");
  //@@ Launch the GPU Kernel here
  matrixMultiply<<<dim_grid, dim_block>>>(deviceA, deviceB, deviceC, numARows, numAColumns, numBRows, numBColumns, numCRows, numCColumns);
  
  cudaDeviceSynchronize();
  wbTime_stop(Compute, "Performing CUDA computation");

  wbTime_start(Copy, "Copying output memory to the CPU");
  //@@ Copy the GPU memory back to the CPU here
  cudaMemcpy(hostC, deviceC, bytesC, cudaMemcpyDeviceToHost);
  wbTime_stop(Copy, "Copying output memory to the CPU");

  wbTime_start(GPU, "Freeing GPU Memory");
  //@@ Free the GPU memory here
  cudaFree(deviceA);
  cudaFree(deviceB);
  cudaFree(deviceC);
  wbTime_stop(GPU, "Freeing GPU Memory");
  // for(int i=0; i<numCRows; i++){
  //   for(int j=0; j<numCColumns; j++){
  //     cout << hostC[i * numCColumns + j] << " ";
  //   }
  //   cout << endl;
  // }
  // cout << "finish freeing memory" <<endl;
  wbSolution(args, hostC, numCRows, numCColumns);

  free(hostA);
  free(hostB);
  free(hostC);
  //./bin/cuda -t vector -i ./inputs/MP2/data/0/input0.raw,./inputs/MP2/data/0/input1.raw -o ./outputs/MP2/data/0/output.raw -e ./inputs/MP2/data/0/output.raw 
  return 0;
}
