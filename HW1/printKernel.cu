#include <cstdio>
#include <cuda_runtime.h>
#include "device_launch_parameters.h"

#include "checkCuda.h"
#include "test.h"


__global__ void printKernel() {
    unsigned int tid = threadIdx.x;
    unsigned int bid = blockIdx.x;
    unsigned int gtid = bid * blockDim.x + tid;

    printf("thread=%u, block=%u, GTID=%u\n", tid, bid, gtid);
}


void runPrintfTests() {
    const int blocks[] = {1, 4, 32};
    const int threads[] = {32, 8, 1};

    for (int config = 0; config < 3; ++config) {
        for (int run = 0; run < 10; ++run) {
            printf("\n=== <<<%d, %d>>> Run %d ===\n", blocks[config], threads[config], run + 1);

            printKernel<<<blocks[config], threads[config]>>>();

            checkCuda(cudaGetLastError());
            checkCuda(cudaDeviceSynchronize());
        }
    }
}