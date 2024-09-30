#include <cuda_runtime.h>
#include <iostream>
#include "kernel.h"

// CUDA kernel for matrix multiplication
__global__ void matrixMulKernel(float* input1, float* input2, int M, int N, int K, float* output) {
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;

    // Compute one element of the output matrix
    if (row < M && col < K) {
        float sum = 0.0f;
        for (int i = 0; i < N; ++i) {
            sum += input1[row * N + i] * input2[i * K + col];
        }
        output[row * K + col] = sum;
    }
}

void matrixMul(array2d_t<float>& input1, array2d_t<float>& input2, array2d_t<float>& output) {
    int M = input1.row_count;
    int N = input1.col_count;
    int K = input2.col_count;
    dim3 threadsPerBlock(16, 16);
    dim3 numBlocks((K + threadsPerBlock.x - 1) / threadsPerBlock.x, (M + threadsPerBlock.y - 1) / threadsPerBlock.y);
    matrixMulKernel<<<numBlocks, threadsPerBlock>>>(input1.data_ptr, input2.data_ptr, M, N, K, output.data_ptr);
}