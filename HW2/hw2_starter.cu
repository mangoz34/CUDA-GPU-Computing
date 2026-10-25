
#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include <stdio.h>
#include <stdlib.h>
#include <Windows.h>
#include <profileapi.h>

#define STB_IMAGE_IMPLEMENTATION // this is needed
#include "stb_image.h"  // download from class website files
#define STB_IMAGE_WRITE_IMPLEMENTATION
#include "stb_image_write.h"  // download from class website files

#include "error_check.h"

// global gaussian blur filter coefficients array here
#define BLUR_FILTER_WIDTH 9  // 9x9 (square) Gaussian blur filter
const float BLUR_FILT[81] = { 0.1084,0.1762,0.2494,0.3071,0.3292,0.3071,0.2494,0.1762,0.1084,0.1762,0.2865,0.4054,0.4994,0.5353,0.4994,0.4054,0.2865,0.1762,0.2494,0.4054,0.5738,0.7066,0.7575,0.7066,0.5738,0.4054,0.2494,0.3071,0.4994,0.7066,0.8703,0.9329,0.8703,0.7066,0.4994,0.3071,0.3292,0.5353,0.7575,0.9329,1.0000,0.9329,0.7575,0.5353,0.3292,0.3071,0.4994,0.7066,0.8703,0.9329,0.8703,0.7066,0.4994,0.3071,0.2494,0.4054,0.5738,0.7066,0.7575,0.7066,0.5738,0.4054,0.2494,0.1762,0.2865,0.4054,0.4994,0.5353,0.4994,0.4054,0.2865,0.1762,0.1084,0.1762,0.2494,0.3071,0.3292,0.3071,0.2494,0.1762,0.1084 };

// blur kernel #1 - global memory only
__global__ void gaussBlurKernel_globalmem(const unsigned char* input, unsigned char* output, const float* filter, int width, int height) { int col = blockIdx.x * blockDim.x + threadIdx.x;int row = blockIdx.y * blockDim.y + threadIdx.y;
    
    if (col >= width || row >= height) {
        return;
    }

    int pixelSum = 0;
    float weightSum = 0.0f;

    const int radius = BLUR_FILTER_WIDTH / 2;

    for (int fy = 0; fy < BLUR_FILTER_WIDTH; fy++) {
        for (int fx = 0; fx < BLUR_FILTER_WIDTH; fx++) {
            int inputRow = row + fy - radius;
            int inputCol = col + fx - radius;

            if (inputRow >= 0 && inputRow < height &&
                inputCol >= 0 && inputCol < width) {

                float weight =
                    filter[fy * BLUR_FILTER_WIDTH + fx];

                unsigned char pixel =
                    input[inputRow * width + inputCol];

                pixelSum += __float2int_rn(pixel * weight);
                weightSum += weight;
            }
        }
    }

    int result = __float2int_rn(pixelSum / weightSum);

    // Keep the output within the unsigned-char range.
    if (result < 0) result = 0;
    if (result > 255) result = 255;

    output[row * width + col] =
        static_cast<unsigned char>(result);
}


// blur kernel #2 - device shared memory (static alloc)
__global__ void gaussBlurKernel_staticShared(const unsigned char* input, unsigned char* output, const float* filter, int width, int height) {
    __shared__ float sharedFilter[BLUR_FILTER_WIDTH * BLUR_FILTER_WIDTH];

    int tid = threadIdx.y * blockDim.x + threadIdx.x;
    int threadsPerBlock = blockDim.x * blockDim.y;

    // Cooperatively copy filter coefficients into shared memory.
    for (int i = tid;i < BLUR_FILTER_WIDTH * BLUR_FILTER_WIDTH; i += threadsPerBlock) {
        sharedFilter[i] = filter[i];
    }

    __syncthreads();

    int col = blockIdx.x * blockDim.x + threadIdx.x;
    int row = blockIdx.y * blockDim.y + threadIdx.y;

    if (col >= width || row >= height) {
        return;
    }

    int pixelSum = 0;
    float weightSum = 0.0f;
    const int radius = BLUR_FILTER_WIDTH / 2;

    for (int fy = 0; fy < BLUR_FILTER_WIDTH; fy++) {
        for (int fx = 0; fx < BLUR_FILTER_WIDTH; fx++) {
            int inputRow = row + fy - radius;
            int inputCol = col + fx - radius;

            if (inputRow >= 0 && inputRow < height &&
                inputCol >= 0 && inputCol < width) {

                float weight = sharedFilter[fy * BLUR_FILTER_WIDTH + fx];

                unsigned char pixel = input[inputRow * width + inputCol];

                pixelSum += __float2int_rn(pixel * weight);
                weightSum += weight;
            }
        }
    }

    int result = __float2int_rn(pixelSum / weightSum);

    if (result < 0) result = 0;
    if (result > 255) result = 255;

    output[row * width + col] = static_cast<unsigned char>(result);
}


// blur kernel #3 - device shared memory (dynamic alloc)
__global__ void gaussBlurKernel_dynamicShared(const unsigned char* input, unsigned char* output, const float* filter, int width, int height) {
    extern __shared__ float sharedFilter[];

    int tid = threadIdx.y * blockDim.x + threadIdx.x;
    int threadsPerBlock = blockDim.x * blockDim.y;
    
    for (int i = tid; i < BLUR_FILTER_WIDTH * BLUR_FILTER_WIDTH; i += threadsPerBlock) {
        sharedFilter[i] = filter[i];
    }

    __syncthreads();

    int col = blockIdx.x * blockDim.x + threadIdx.x;
    int row = blockIdx.y * blockDim.y + threadIdx.y;

    if (col >= width || row >= height) {
        return;
    }

    int pixelSum = 0;
    float weightSum = 0.0f;
    const int radius = BLUR_FILTER_WIDTH / 2;

    for (int fy = 0; fy < BLUR_FILTER_WIDTH; fy++) {
        for (int fx = 0; fx < BLUR_FILTER_WIDTH; fx++) {
            int inputRow = row + fy - radius;
            int inputCol = col + fx - radius;

            if (inputRow >= 0 && inputRow < height &&
                inputCol >= 0 && inputCol < width) {

                float weight =
                    sharedFilter[fy * BLUR_FILTER_WIDTH + fx];

                unsigned char pixel =
                    input[inputRow * width + inputCol];

                pixelSum += __float2int_rn(pixel * weight);
                weightSum += weight;
            }
        }
    }

    int result = __float2int_rn(pixelSum / weightSum);

    if (result < 0) result = 0;
    if (result > 255) result = 255;

    output[row * width + col] =
        static_cast<unsigned char>(result);
}


int main() {
    // read input image from file - be aware of image pixel bit-depth and resolution (horiz x vertical)
    const char filename[] = "hw2_testimage1.png";
    int x_cols = 0;
    int y_rows = 0;
    int n_pixdepth = 0;
    unsigned char* imgData = stbi_load(
        filename, &x_cols, &y_rows, &n_pixdepth, 1);

    if (imgData == NULL) {
        fprintf(stderr, "Failed to load image: %s\n",
            stbi_failure_reason());
        return 1;
    }

    size_t imgSize = (size_t)x_cols * y_rows * sizeof(unsigned char);

    printf("Image: %d x %d\n", x_cols, y_rows);
    printf("Loaded image size: %zu bytes\n", imgSize);

    // setup additional host variables, allocate host memory as needed
    cudaError_t cudaStatus = cudaSuccess;
    int exitCode = 1;

    unsigned char* h_imgOut = NULL;
    unsigned char* d_input = NULL;
    unsigned char* d_output = NULL;
    float* d_filter = NULL;

    size_t filterSize = sizeof(BLUR_FILT);

    dim3 block(16, 16);
    dim3 grid(
        (x_cols + block.x - 1) / block.x,
        (y_rows + block.y - 1) / block.y);

    LARGE_INTEGER frequency;
    LARGE_INTEGER start1, stop1;
    LARGE_INTEGER start2, stop2;

    double totalTimeMs = 0.0;
    double kernelTimeMs = 0.0;
    
    if (!QueryPerformanceFrequency(&frequency)) {
        fprintf(stderr, "High-resolution timer is unavailable.\n");
        goto Error;
    }

    h_imgOut = (unsigned char*)malloc(imgSize);
    if (h_imgOut == NULL) {
        fprintf(stderr, "Failed to allocate host output memory.\n");
        goto Error;
    }

    printf("Block: %u x %u\n", block.x, block.y);
    printf("Grid: %u x %u\n", grid.x, grid.y);

    // START timer #1
    QueryPerformanceCounter(&start1);

    // allocate device memory
    CUDA_CHECK(cudaMalloc((void**)&d_input, imgSize));
    CUDA_CHECK(cudaMalloc((void**)&d_output, imgSize));
    CUDA_CHECK(cudaMalloc((void**)&d_filter, filterSize));

    // copy host data to device
    CUDA_CHECK(cudaMemcpy(
        d_input, imgData, imgSize, cudaMemcpyHostToDevice));

    CUDA_CHECK(cudaMemcpy(
        d_filter, BLUR_FILT, filterSize, cudaMemcpyHostToDevice));

    // START timer #2
    QueryPerformanceCounter(&start2);

    // launch kernel --- use appropriate heuristics to determine #threads/block and #blocks/grid to ensure coverage of your 2D data range
    gaussBlurKernel_globalmem << <grid, block>> > (
        d_input, d_output, d_filter, x_cols, y_rows);

    // Check for any errors launching the kernel
    cudaStatus = cudaGetLastError();
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "Blur kernel launch failed: %s\n",
            cudaGetErrorString(cudaStatus));
        goto Error;
    }

    // call cudaDeviceSynchronize() to wait for the kernel to finish, and return
    // any errors encountered during the launch.
    CUDA_CHECK(cudaDeviceSynchronize());

    // STOP timer #2
    QueryPerformanceCounter(&stop2);

    // retrieve result data from device back to host
    CUDA_CHECK(cudaMemcpy(
        h_imgOut, d_output, imgSize, cudaMemcpyDeviceToHost));

    // STOP timer #1
    QueryPerformanceCounter(&stop1);

    // cudaDeviceReset( ) must be called in order for profiling and
    // tracing tools such as Nsight and Visual Profiler to show complete traces.
    // Called below, after freeing device memory.

    const char imgFileOut[] = "hw2_outimage1_exp.png";
    // save result output image data to file
    if (!stbi_write_png(
        imgFileOut, x_cols, y_rows, 1, h_imgOut, x_cols)) {
        fprintf(stderr, "Failed to save output image.\n");
        goto Error;
    }

    printf("Saved output: %s\n", imgFileOut);

    // EXTRA CREDIT:
    // start timer #3
    // run host sequential blur routine
    // stop timer #3

    // retrieve and save timer results (write to console or file)
    totalTimeMs =
        (double)(stop1.QuadPart - start1.QuadPart)
        / frequency.QuadPart * 1000.0;

    kernelTimeMs =
        (double)(stop2.QuadPart - start2.QuadPart)
        / frequency.QuadPart * 1000.0;

    printf("Timer #1 - GPU allocation + transfers + kernel: %.6f ms\n",
        totalTimeMs);

    printf("Timer #2 - Kernel launch + synchronization: %.6f ms\n",
        kernelTimeMs);

    exitCode = 0;

Error:  // assumes error macro has a goto Error statement

    // free host and device memory
    if (d_input != NULL) cudaFree(d_input);
    if (d_output != NULL) cudaFree(d_output);
    if (d_filter != NULL) cudaFree(d_filter);

    if (h_imgOut != NULL) free(h_imgOut);
    if (imgData != NULL) stbi_image_free(imgData);

    cudaStatus = cudaDeviceReset();
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaDeviceReset failed: %s\n",
            cudaGetErrorString(cudaStatus));
        exitCode = 1;
    }

    return exitCode;
}

// EXTRA CREDIT
// define host sequential blur-kernel routine

/*
int main() {
     // read input image from file - be aware of image pixel bit-depth and resolution (horiz x vertical)
    const char filename[] = "hw2_testimage1.png";
    int x_cols = 0;
    int y_rows = 0;
    int n_pixdepth = 0;
    unsigned char* imgData = stbi_load(filename, &x_cols, &y_rows, &n_pixdepth, 1);
    if (imgData == NULL) {
        fprintf(stderr, "Failed to load image: %s\n",
            stbi_failure_reason());
        return 1;
    }

    int imgSize = x_cols * y_rows * sizeof(unsigned char);

    printf("Image: %d x %d\n", x_cols, y_rows);
    printf("Loaded image size: %zu bytes\n", imgSize);

    // setup additional host variables, allocate host memory as needed

    // START timer #1

    // allocate device memory

    // copy host data to device

    // START timer #2
    // launch kernel --- use appropriate heuristics to determine #threads/block and #blocks/grid to ensure coverage of your 2D data range

    // Check for any errors launching the kernel
    cudaStatus = cudaGetLastError();
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "addKernel launch failed: %s\n", cudaGetErrorString(cudaStatus));
        goto Error;
    }

    // call cudaDeviceSynchronize() to wait for the kernel to finish, and return
    // any errors encountered during the launch.
    
    // STOP timer #2
    // 
    // retrieve result data from device back to host

    // STOP timer #1

    // cudaDeviceReset( ) must be called in order for profiling and
    // tracing tools such as Nsight and Visual Profiler to show complete traces.

       // save result output image data to file
    const char imgFileOut[] = "c:/Temp/AUT22_GPUCompute/dat/hw2_outimage1.png";
    stbi_write_png(imgFileOut, x_cols, y_rows, 1, h_imgOut, x_cols * n_pixdepth);


    // EXTRA CREDIT:
    // start timer #3
    // run host sequential blur routine
    // stop timer #3

    // retrieve and save timer results (write to console or file)
 
Error:  // assumes error macro has a goto Error statement

    // free host and device memory

    return 0;
}


*/