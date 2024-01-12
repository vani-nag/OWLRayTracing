#ifndef KERNELS_H
#define KERNELS_H

// CUDA kernel for element-wise multiplication
__global__ void multiplyKernel(const float* a, const float* b, float* result, int size);

#endif  // KERNELS_H