// Histogram Equalization

#include <wb.h>

typedef unsigned char uint8_t;
typedef unsigned int uint_t;
#define HISTOGRAM_LENGTH 256
#define BLOCK_SIZE 16

#define wbCheck(stmt)                                                     \
  do {                                                                    \
    cudaError_t err = stmt;                                               \
    if (err != cudaSuccess) {                                             \
      wbLog(ERROR, "Failed to run stmt ", #stmt);                         \
      wbLog(ERROR, "Got CUDA error ...  ", cudaGetErrorString(err));      \
      return -1;                                                          \
    }                                                                     \
  } while (0)
//@@ insert code here


__global__ void floatToUint8(uint8_t *deviceImageUint8, float *deviceImageFloat, int width, int height, int channels){
  int Col = blockIdx.x * blockDim.x + threadIdx.x;
  int Row = blockIdx.y * blockDim.y + threadIdx.y;
  if(Col < width && Row < height){
    int index =  (width * Row + Col) * channels + threadIdx.z;
    deviceImageUint8[index] = (uint8_t)(255 * deviceImageFloat[index]);
  }
}

__global__ void rgb2Gray(uint8_t *deviceImageUint8, uint8_t *deviceGrayImage, int width, int height, int channels){
  int Col = blockIdx.x * blockDim.x + threadIdx.x;
  int Row = blockIdx.y * blockDim.y + threadIdx.y;
  if(Col < width && Row < height){
    int outputIdx = Row * width + Col;
    uint8_t r = deviceImageUint8[outputIdx*channels];
    uint8_t g = deviceImageUint8[outputIdx*channels + 1];
    uint8_t b = deviceImageUint8[outputIdx*channels + 2];
    deviceGrayImage[outputIdx] = (uint8_t)(0.21f * (float)r + 0.71f * (float)g + 0.07f * (float)b);
  }
}

__global__ void image2Histogram(uint8_t *image, uint_t *histogram, int width, int height){
  __shared__ uint_t histo_private[HISTOGRAM_LENGTH];

  // use 256 threads to initialize histo_private
  int linearIdx = threadIdx.y * blockDim.x + threadIdx.x;
  if(linearIdx < HISTOGRAM_LENGTH){
    // if(blockIdx.x == 0){
    //   histogram[linearIdx] = 0;
    // }
    histo_private[linearIdx] = 0;
  }

  __syncthreads();

  int Col = blockIdx.x * blockDim.x + threadIdx.x;
  int Row = blockIdx.y * blockDim.y + threadIdx.y;
  if(Col < width && Row < height){
    int index = Row * width + Col;
    atomicAdd(&(histo_private[image[index]]), 1);
  }

  __syncthreads();

  if(linearIdx < HISTOGRAM_LENGTH){
    atomicAdd(&(histogram[linearIdx]), histo_private[linearIdx]);
  }
}

__global__ void scan(uint_t *histo, float *cdf, int numData) {
  //@@ Modify the body of this function to complete the functionality of
  //@@ the scan on the device
  //@@ You may need multiple kernel calls; write your kernels before this
  //@@ function and call them from the host
  __shared__ float partialSum[HISTOGRAM_LENGTH];
  int tx = threadIdx.x;
  int firstIdx = blockIdx.x * blockDim.x * 2 + threadIdx.x*2;
  int secondIdx = firstIdx + 1;
  if(firstIdx < HISTOGRAM_LENGTH){
    partialSum[2*tx] = ((float)histo[firstIdx]) / numData;
  }
  else{
    partialSum[2*tx] = 0;
  }
  if(secondIdx < HISTOGRAM_LENGTH){
    partialSum[2*tx + 1] = ((float)histo[secondIdx]) / numData;
  }
  else{
    partialSum[2*tx + 1] = 0;
  }
  int stride = 1;
  while(stride < HISTOGRAM_LENGTH){
    __syncthreads();
    int index = (tx + 1)*stride*2 - 1;
    if(index < HISTOGRAM_LENGTH && (index-stride) >= 0){
      partialSum[index] += partialSum[index-stride];
    }
    stride *= 2;
  }

  stride = HISTOGRAM_LENGTH / 4;
  while(stride > 0){
    __syncthreads();
    int index = (tx + 1)*stride*2 - 1;
    if(index + stride < HISTOGRAM_LENGTH){
      partialSum[index+stride] += partialSum[index];
    }
    stride = stride / 2;
  }

  __syncthreads();
  if(firstIdx < HISTOGRAM_LENGTH){
    cdf[firstIdx] = partialSum[2*tx];
  }
  if(secondIdx < HISTOGRAM_LENGTH){
    cdf[secondIdx] = partialSum[2*tx + 1];
  }
}

__global__ void HistoEqualization(uint8_t *deviceImageUint8, float* cdf, int width, int height, int channels){
  __shared__ float cdfmin;

  // every block, one thread is responsible of getting this global min.
  if(threadIdx.x == 0 && threadIdx.y ==0 && threadIdx.z == 0){
    cdfmin = cdf[0];
  }
  __syncthreads();
  int Col = blockIdx.x * blockDim.x + threadIdx.x;
  int Row = blockIdx.y * blockDim.y + threadIdx.y;
  if(Col < width && Row < height){
    int index =  (width * Row + Col) * channels + threadIdx.z;
    deviceImageUint8[index] = min(max(255*(cdf[deviceImageUint8[index]] - cdfmin)/(1.0f - cdfmin), 0.0f), 255.0f);
  }
}

__global__ void uint8ToFloat(uint8_t *deviceImageUint8, float* deviceOutputImage, int width, int height, int channels){
  int Col = blockIdx.x * blockDim.x + threadIdx.x;
  int Row = blockIdx.y * blockDim.y + threadIdx.y;
  if(Col < width && Row < height){
    int index =  (width * Row + Col) * channels + threadIdx.z;
    deviceOutputImage[index] = ((float)deviceImageUint8[index]) / 255.0f;
  }
}





int main(int argc, char **argv) {
  wbArg_t args;
  int imageWidth;
  int imageHeight;
  int imageChannels;
  wbImage_t inputImage;
  wbImage_t outputImage;
  float *hostInputImageData;
  float *hostOutputImageData;
  const char *inputImageFile;

  //@@ Insert more code here

  args = wbArg_read(argc, argv); /* parse the input arguments */

  inputImageFile = wbArg_getInputFile(args, 0);

  wbTime_start(Generic, "Importing data and creating memory on host");
  inputImage = wbImport(inputImageFile);
  imageWidth = wbImage_getWidth(inputImage);
  imageHeight = wbImage_getHeight(inputImage);
  imageChannels = wbImage_getChannels(inputImage);
  outputImage = wbImage_new(imageWidth, imageHeight, imageChannels);
  hostInputImageData = wbImage_getData(inputImage);
  hostOutputImageData = wbImage_getData(outputImage);
  wbTime_stop(Generic, "Importing data and creating memory on host");

  //@@ insert code here
  float *deviceImageFloat;
  uint8_t *deviceImageUint8;

  int totalNumDataRGB = imageWidth * imageHeight * imageChannels;
  int totalNumDataGray = imageWidth * imageHeight;
  wbCheck(cudaMalloc((void**)&deviceImageFloat, totalNumDataRGB * sizeof(float)));
  wbCheck(cudaMalloc((void**)&deviceImageUint8, totalNumDataRGB * sizeof(uint8_t)));
  wbCheck(cudaMemcpy(deviceImageFloat, hostInputImageData, totalNumDataRGB * sizeof(float), cudaMemcpyHostToDevice));

  // float *hostTestFloat = (float *)malloc(totalNumDataRGB * sizeof(float));
  // cudaMemcpy(hostTestFloat, deviceImageFloat, totalNumDataRGB * sizeof(float), cudaMemcpyDeviceToHost);
  // printf("float ");
  // for(int i=0; i<totalNumDataRGB; i++){
  //   if(hostTestFloat[i] != 0){
  //     printf(" %f", hostTestFloat[i]);
  //   }
  // }

  // Cast to uint8_t
  int gridDim1 = (int)ceil(((float)imageWidth)/BLOCK_SIZE);
  int gridDim2 = (int)ceil(((float)imageHeight)/BLOCK_SIZE);
  dim3 dim_grid(gridDim1, gridDim2, 1);
  dim3 dim_block_rgb(BLOCK_SIZE, BLOCK_SIZE, 3);

  floatToUint8<<<dim_grid, dim_block_rgb>>>(deviceImageUint8, deviceImageFloat, imageWidth, imageHeight, imageChannels);
  cudaDeviceSynchronize();
  // uint8_t *hostTestuint8 = (uint8_t *)malloc(totalNumDataRGB * sizeof(uint8_t));
  // cudaMemcpy(hostTestuint8, deviceImageUint8, totalNumDataRGB * sizeof(uint8_t), cudaMemcpyDeviceToHost);
  // printf("uint8 ");
  // for(int i=0; i<totalNumDataRGB; i++){
  //   if(hostTestuint8[i] != 0){
  //     printf(" %d", hostTestuint8[i]);
  //   }
  // }

  // convert to gray
  uint8_t *deviceGrayImage;
  cudaMalloc((void**)&deviceGrayImage, totalNumDataGray * sizeof(uint8_t));
  dim3 dim_block_gray(BLOCK_SIZE, BLOCK_SIZE, 1);
  rgb2Gray<<<dim_grid, dim_block_gray>>>(deviceImageUint8, deviceGrayImage, imageWidth, imageHeight, imageChannels);
  cudaDeviceSynchronize();
  uint8_t *hostTestGray = (uint8_t *)malloc(totalNumDataGray * sizeof(uint8_t));
  cudaMemcpy(hostTestGray, deviceGrayImage, totalNumDataGray * sizeof(uint8_t), cudaMemcpyDeviceToHost);
  // printf("gray ");
  // for(int i=0; i<totalNumDataGray; i++){
  //   if(hostTestGray[i] != 0){
  //     printf(" %d", hostTestGray[i]);
  //   }
  // }


  // compute histo for gray image
  uint_t *deviceHisto;
  cudaMalloc((void**)&deviceHisto, HISTOGRAM_LENGTH * sizeof(uint_t));
  image2Histogram<<<dim_grid, dim_block_gray>>>(deviceGrayImage, deviceHisto, imageWidth, imageHeight);
  cudaDeviceSynchronize();
  // uint_t *hostTestHist = (uint_t *)malloc(HISTOGRAM_LENGTH * sizeof(uint_t));
  // cudaMemcpy(hostTestHist, deviceHisto, HISTOGRAM_LENGTH * sizeof(uint_t), cudaMemcpyDeviceToHost);
  // printf("Hist ");
  // for(int i=0; i<HISTOGRAM_LENGTH; i++){
  //   printf(" %d", hostTestHist[i]);
  // }

  // scan histo to get CDF
  float *deviceCDF;
  cudaMalloc((void**)&deviceCDF, HISTOGRAM_LENGTH * sizeof(float));
  dim3 scan_grid(1, 1, 1);
  dim3 scan_block(HISTOGRAM_LENGTH/2, 1, 1);
  scan<<<scan_grid, scan_block>>>(deviceHisto, deviceCDF, totalNumDataGray);
  cudaDeviceSynchronize();

  // float *hostTest = (float *)malloc(HISTOGRAM_LENGTH * sizeof(float));
  // cudaMemcpy(hostTest, deviceCDF, HISTOGRAM_LENGTH * sizeof(float), cudaMemcpyDeviceToHost);
  // printf("CDF: ");
  // for(int i=0; i<HISTOGRAM_LENGTH; i++){
  //   printf(" %f", hostTest[i]);
  // }

  // transform image based on cdf
  HistoEqualization<<<dim_grid, dim_block_rgb>>>(deviceImageUint8, deviceCDF, imageWidth, imageHeight, imageChannels);
  cudaDeviceSynchronize();
  // convert back to float
  float* deviceOutputImage;
  cudaMalloc((void**)&deviceOutputImage, totalNumDataRGB * sizeof(float));
  uint8ToFloat<<<dim_grid, dim_block_rgb>>>(deviceImageUint8, deviceOutputImage, imageWidth, imageHeight, imageChannels);
  
  cudaDeviceSynchronize();

  cudaMemcpy(hostOutputImageData, deviceOutputImage, totalNumDataRGB * sizeof(float), cudaMemcpyDeviceToHost);

  cudaFree(deviceImageFloat);
  cudaFree(deviceImageUint8);
  cudaFree(deviceGrayImage);
  cudaFree(deviceHisto);
  cudaFree(deviceCDF);
  cudaFree(deviceOutputImage);
  
  wbSolution(args, outputImage);

  // //@@ insert code here

  return 0;
}
