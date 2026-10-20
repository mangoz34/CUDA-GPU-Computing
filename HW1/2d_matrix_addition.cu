#include <cstdio>
#include <cstdlib>
#include <cmath>
#include <cuda_runtime.h>
#include "device_launch_parameters.h"

#include "checkCuda.h"
#include "test.h"


__global__ void matrixAddKernel(const float* A, const float* B, float* C, int M, int N) {
    int col = blockIdx.x * blockDim.x + threadIdx.x;
    int row = blockIdx.y * blockDim.y + threadIdx.y;

    if (row < M && col < N) {
        int i = row * N + col;
        C[i] = A[i] + B[i];
    }
}


void matrixAddCPU(const float* A, const float* B, float* C, int M, int N) {
    for (int row = 0; row < M; ++row) {
        for (int col = 0; col < N; ++col) {
            int i = row * N + col;
            C[i] = A[i] + B[i];
        }
    }
}


static void runMatrixAddTest(int M, int N, float tolerance) {
    const size_t count = static_cast<size_t>(M) * N;
    const size_t bytes = count * sizeof(float);

    float* h_A = (float*)std::malloc(bytes);
    float* h_B = (float*)std::malloc(bytes);
    float* h_CCPU = (float*)std::malloc(bytes);
    float* h_CGPU = (float*)std::malloc(bytes);

    if (!h_A || !h_B || !h_CCPU || !h_CGPU) {
        std::fprintf(stderr, "Host memory allocation failed.\n");

        std::free(h_A);
        std::free(h_B);
        std::free(h_CCPU);
        std::free(h_CGPU);

        std::exit(EXIT_FAILURE);
    }

    for (size_t i = 0; i < count; ++i) {
        h_A[i] = 0.1f * static_cast<float>(i + 1);
        h_B[i] = 0.2f * static_cast<float>(i + 1);
    }

    float* d_A = nullptr;
    float* d_B = nullptr;
    float* d_C = nullptr;

    checkCuda(cudaMalloc((void**)&d_A, bytes));
    checkCuda(cudaMalloc((void**)&d_B, bytes));
    checkCuda(cudaMalloc((void**)&d_C, bytes));

    checkCuda(cudaMemcpy(d_A, h_A, bytes, cudaMemcpyHostToDevice));
    checkCuda(cudaMemcpy(d_B, h_B, bytes, cudaMemcpyHostToDevice));

    matrixAddCPU(h_A, h_B, h_CCPU, M, N);

    dim3 block(16, 16);
    dim3 grid(
        (N + block.x - 1) / block.x,
        (M + block.y - 1) / block.y
    );

    matrixAddKernel<<<grid, block>>>(d_A, d_B, d_C, M, N);

    checkCuda(cudaGetLastError());
    checkCuda(cudaDeviceSynchronize());

    checkCuda(cudaMemcpy(h_CGPU, d_C, bytes, cudaMemcpyDeviceToHost));

    size_t mismatchCount = 0;
    float maxError = 0.0f;

    for (size_t i = 0; i < count; ++i) {
        float error = std::fabs(h_CCPU[i] - h_CGPU[i]);

        if (!std::isfinite(h_CCPU[i]) || !std::isfinite(h_CGPU[i]) || error > tolerance) {
            ++mismatchCount;
        }

        if (error > maxError) {
            maxError = error;
        }

        if (count <= 16 || i % 1000000 == 0) {
            size_t row = i / N;
            size_t col = i % N;

            printf("row=%zu, col=%zu, CPU=%.9g, GPU=%.9g, error=%.9g\n", row, col, h_CCPU[i], h_CGPU[i], error);
        }
    }

    printf("\n2D matrix addition %dx%d: %s\n", M, N, mismatchCount == 0 ? "PASS" : "FAIL");
    printf("Tolerance: %.9g\n", tolerance);
    printf("Mismatches: %zu\n", mismatchCount);
    printf("Maximum absolute error: %.9g\n\n", maxError);

    checkCuda(cudaFree(d_A));
    checkCuda(cudaFree(d_B));
    checkCuda(cudaFree(d_C));

    std::free(h_A);
    std::free(h_B);
    std::free(h_CCPU);
    std::free(h_CGPU);
}

void runMatrixAddTests() {
    runMatrixAddTest(4, 4, 1e-5f);
    runMatrixAddTest(500, 500, 1e-5f);
    runMatrixAddTest(3024, 4032, 1e-5f);
    runMatrixAddTest(3024, 4032, 0.0f);
}