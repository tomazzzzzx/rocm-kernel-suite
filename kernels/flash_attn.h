#pragma once
#include <hip/hip_runtime.h>

// Flash Attention v2 for AMD CDNA3
template<int BLOCK_SIZE, int D>
__global__ void flash_attn_fwd(
    const float* Q, const float* K, const float* V,
    float* O, float scale, int seq_len
) {
    extern __shared__ float smem[];
    int tid = threadIdx.x;
    int batch = blockIdx.x;
    int head = blockIdx.y;

    float m_prev = -INFINITY;
    float l_prev = 0.0f;

    for (int j = 0; j < seq_len; j += BLOCK_SIZE) {
        // Tile-based compute with online softmax
        // S = Q @ K^T / sqrt(d)
        // Update running max and sum
        // Accumulate output
    }
}
