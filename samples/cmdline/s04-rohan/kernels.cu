#include "kernels.h"

// CUDA kernel for element-wise multiplication
__global__ void multiplyKernel(const float* a, const float* b, float* result, int size) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;

    // Check if the thread index is within the array bounds
    if (idx < size) {
        result[idx] = a[idx] * b[idx];
    }
}