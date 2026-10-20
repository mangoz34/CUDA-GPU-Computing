#include <cstdio>
#include <cstdlib>
#include <cmath>
#include <cuda_runtime.h>
#include "device_launch_parameters.h"

#include "checkCuda.h"
#include "test.h"


__global__ void saxpyKernel(float a, const float* x, float* y, int N) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;

    if (i < N) {
        y[i] = a * x[i] + y[i];
    }
}

void saxpyCPU(float a, const float* x, float* y, int N) {
    for (int i = 0; i < N; ++i) {
        y[i] = a * x[i] + y[i];
    }
}


static void runSaxpyTest(int N, float tolerance) {
    const float a = 1.23f;
    const size_t bytes = static_cast<size_t>(N) * sizeof(float);

    float* h_x = (float*)std::malloc(bytes);
    float* h_yCPU = (float*)std::malloc(bytes);
    float* h_yGPU = (float*)std::malloc(bytes);

    if (!h_x || !h_yCPU || !h_yGPU) {
        std::fprintf(stderr, "Host memory allocation failed.\n");

        std::free(h_x);
        std::free(h_yCPU);
        std::free(h_yGPU);

        std::exit(EXIT_FAILURE);
    }

    for (int i = 0; i < N; ++i) {
        h_x[i] = 0.1f * (i + 1);
        h_yCPU[i] = 0.2f * (i + 1);
        h_yGPU[i] = h_yCPU[i];
    }

    float* d_x = nullptr;
    float* d_y = nullptr;

    checkCuda(cudaMalloc((void**)&d_x, bytes));
    checkCuda(cudaMalloc((void**)&d_y, bytes));

    checkCuda(cudaMemcpy(d_x, h_x, bytes, cudaMemcpyHostToDevice));
    checkCuda(cudaMemcpy(d_y, h_yGPU, bytes, cudaMemcpyHostToDevice));

    saxpyCPU(a, h_x, h_yCPU, N);

    const int threadsPerBlock = 256;
    const int blocks = (N + threadsPerBlock - 1) / threadsPerBlock;

    saxpyKernel<<<blocks, threadsPerBlock>>>(a, d_x, d_y, N);

    checkCuda(cudaGetLastError());
    checkCuda(cudaDeviceSynchronize());

    checkCuda(cudaMemcpy(h_yGPU, d_y, bytes, cudaMemcpyDeviceToHost));

    int mismatchCount = 0;
    float maxError = 0.0f;

    const int printInterval = N >= 10 ? N / 10 : 1;

    for (int i = 0; i < N; ++i) {
        float error = std::fabs(h_yCPU[i] - h_yGPU[i]);

        if (!std::isfinite(h_yCPU[i]) ||
            !std::isfinite(h_yGPU[i]) ||
            error > tolerance) {
            ++mismatchCount;
        }

        if (error > maxError) {
            maxError = error;
        }

        if (i % printInterval == 0) {
            printf("i=%d, CPU=%.9g, GPU=%.9g, error=%.9g\n", i, h_yCPU[i], h_yGPU[i], error);
        }
    }

    printf("\nSAXPY N=%d: %s\n", N, mismatchCount == 0 ? "PASS" : "FAIL");
    printf("Tolerance: %.9g\n", tolerance);
    printf("Mismatches: %d\n", mismatchCount);
    printf("Maximum absolute error: %.9g\n\n", maxError);

    checkCuda(cudaFree(d_x));
    checkCuda(cudaFree(d_y));

    std::free(h_x);
    std::free(h_yCPU);
    std::free(h_yGPU);
}

void runSaxpyTests() {
    runSaxpyTest(10, 1e-5f);

    runSaxpyTest(1000, 1e-5f);
    runSaxpyTest(1000, 1e-4f);

    runSaxpyTest(10000000, 1e-4f);
    runSaxpyTest(10000000, 0.1f);
    runSaxpyTest(10000000, 1.0f);
}