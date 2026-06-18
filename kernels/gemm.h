#pragma once
#include <hip/hip_runtime.h>

// Optimized GEMM kernel for AMD CDNA3 architecture
// Uses shared memory tiling and matrix core instructions
template<int BM, int BN, int BK>
__global__ void sgemm_tiled(
    const float* __restrict__ A,
    const float* __restrict__ B,
    float* __restrict__ C,
    int M, int N, int K
) {
    __shared__ float As[BM][BK];
    __shared__ float Bs[BK][BN];

    int bx = blockIdx.x, by = blockIdx.y;
    int tx = threadIdx.x, ty = threadIdx.y;
    int row = by * BM + ty;
    int col = bx * BN + tx;

    float acc = 0.0f;
    for (int k = 0; k < K; k += BK) {
        As[ty][tx] = A[row * K + k + tx];
        Bs[ty][tx] = B[(k + ty) * N + col];
        __syncthreads();
        for (int i = 0; i < BK; i++)
            acc += As[ty][i] * Bs[i][tx];
        __syncthreads();
    }
    if (row < M && col < N)
        C[row * N + col] = acc;
}
