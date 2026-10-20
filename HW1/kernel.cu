#include <cstdlib>
#include <cuda_runtime.h>

#include "checkCuda.h"
#include "test.h"

int main() {
    checkCuda(cudaSetDevice(0));

    runPrintfTests();
    runSaxpyTests();
    runMatrixAddTests();
    runVolumeAddTests();

    return EXIT_SUCCESS;
}