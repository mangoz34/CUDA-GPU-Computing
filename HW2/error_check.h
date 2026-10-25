#pragma once
#define CUDA_CHECK(call)                                      \
    do {                                                      \
        cudaStatus = (call);                                  \
        if (cudaStatus != cudaSuccess) {                      \
            fprintf(stderr, "%s failed: %s\n",                 \
                    #call, cudaGetErrorString(cudaStatus));   \
            goto Error;                                       \
        }                                                     \
    } while (0)