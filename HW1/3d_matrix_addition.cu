#include <cstdio>
#include <cstdlib>
#include <cmath>
#include <cuda_runtime.h>
#include "device_launch_parameters.h"

#include "checkCuda.h"
#include "test.h"


__global__ void volumeAddKernel(const float* A, const float* B, float* C, int M, int N, int P) {
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;
    int z = blockIdx.z * blockDim.z + threadIdx.z;

    if (x < P && y < N && z < M) {
        size_t i = (static_cast<size_t>(z) * N + y) * P + x;
        C[i] = A[i] + B[i];
    }
}

void volumeAddCPU(const float* A, const float* B, float* C, int M, int N, int P) {
    for (int z = 0; z < M; ++z) {
        for (int y = 0; y < N; ++y) {
            for (int x = 0; x < P; ++x) {
                size_t i = (static_cast<size_t>(z) * N + y) * P + x;
                C[i] = A[i] + B[i];
            }
        }
    }
}



static void runVolumeAddTest(int M, int N, int P, float tolerance) {
    const size_t count = static_cast<size_t>(M) * N * P;
    const size_t bytes = count * sizeof(float);

    size_t freeBytes = 0;
    size_t totalBytes = 0;
    checkCuda(cudaMemGetInfo(&freeBytes, &totalBytes));

    float* h_A = (float*)std::malloc(bytes);
    float* h_B = (float*)std::malloc(bytes);
    float* h_CCPU = (float*)std::malloc(bytes);

    if (!h_A || !h_B || !h_CCPU) {
        std::fprintf(stderr, "Host memory allocation failed.\n");

        std::free(h_A);
        std::free(h_B);
        std::free(h_CCPU);

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


    volumeAddCPU(h_A, h_B, h_CCPU, M, N, P);

    std::free(h_A);
    std::free(h_B);
    h_A = nullptr;
    h_B = nullptr;

    dim3 block(8, 8, 8);
    dim3 grid(
        (P + block.x - 1) / block.x,
        (N + block.y - 1) / block.y,
        (M + block.z - 1) / block.z
    );

    volumeAddKernel<<<grid, block>>>(d_A, d_B, d_C, M, N, P);

    checkCuda(cudaGetLastError());
    checkCuda(cudaDeviceSynchronize());

    checkCuda(cudaFree(d_A));
    checkCuda(cudaFree(d_B));
    d_A = nullptr;
    d_B = nullptr;

    float* h_CGPU = (float*)std::malloc(bytes);

    if (!h_CGPU) {
        std::fprintf(stderr, "Host GPU-output allocation failed.\n");

        checkCuda(cudaFree(d_C));
        std::free(h_CCPU);

        std::exit(EXIT_FAILURE);
    }

    checkCuda(cudaMemcpy(h_CGPU, d_C, bytes, cudaMemcpyDeviceToHost));
    checkCuda(cudaFree(d_C));
    d_C = nullptr;

    size_t mismatchCount = 0;
    float maxError = 0.0f;

    for (size_t i = 0; i < count; ++i) {
        float error = std::fabs(h_CCPU[i] - h_CGPU[i]);

        if (!std::isfinite(h_CCPU[i]) ||
            !std::isfinite(h_CGPU[i]) ||
            error > tolerance) {
            ++mismatchCount;
        }

        if (error > maxError) {
            maxError = error;
        }

        if (count <= 27 || i % 100000000 == 0) {
            size_t x = i % P;
            size_t y = (i / P) % N;
            size_t z = i / (static_cast<size_t>(N) * P);

            printf("z=%zu, y=%zu, x=%zu, CPU=%.9g, GPU=%.9g, error=%.9g\n", z, y, x, h_CCPU[i], h_CGPU[i], error);
        }
    }

    printf("\n3D volume addition %dx%dx%d: %s\n", M, N, P, mismatchCount == 0 ? "PASS" : "FAIL");
    printf("Elements: %zu\n", count);
    printf("Tolerance: %.9g\n", tolerance);
    printf("Mismatches: %zu\n", mismatchCount);
    printf("Maximum absolute error: %.9g\n", maxError);

    std::free(h_CCPU);
    std::free(h_CGPU);
}


void runVolumeAddTests() {
    runVolumeAddTest(3, 3, 3, 1e-5f);
    runVolumeAddTest(100, 100, 100, 1e-5f);
    runVolumeAddTest(100, 100, 100, 0.0f);
    runVolumeAddTest(1000, 1000, 1000, 0.0f);
}