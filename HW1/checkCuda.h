#pragma once
#pragma once

#include <cstdio>
#include <cstdlib>
#include <cuda_runtime.h>

#define checkCuda(call)                                         \
    do {                                                        \
        cudaError_t error = (call);                             \
        if (error != cudaSuccess) {                             \
            std::fprintf(stderr, "CUDA error at %s:%d: %s\n",   \
                         __FILE__, __LINE__,                    \
                         cudaGetErrorString(error));            \
            std::exit(EXIT_FAILURE);                            \
        }                                                       \
    } while (0)