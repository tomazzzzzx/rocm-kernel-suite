#pragma once
#include <hip/hip_runtime.h>

template<int BM=128, int BN=128, int BK=32>
__global__ void sgemm_optimized(
    const float* __restrict__ A, const float* __restrict__ B,
    float* __restrict__ C, int M, int N, int K) {
    __shared__ float As[BM][BK], Bs[BK][BN];
    int bx=blockIdx.x, by=blockIdx.y, tx=threadIdx.x, ty=threadIdx.y;
    float acc[16][16] = {};
    for (int k=0; k<K; k+=BK) {
        for (int i=ty; i<BM; i+=16) for (int j=tx; j<BK; j+=16) {
            int r=by*BM+i, c=k+j; As[i][j]=(r<M&&c<K)?A[r*K+c]:0;
        }
        for (int i=ty; i<BK; i+=16) for (int j=tx; j<BN; j+=16) {
            int r=k+i, c=bx*BN+j; Bs[i][j]=(r<K&&c<N)?B[r*N+c]:0;
        }
        __syncthreads();
        for (int i=0;i<BK;i++) for (int ii=0;ii<16;ii++) for (int jj=0;jj<16;jj++)
            acc[ii][jj]+=As[ty*16+ii][i]*Bs[i][tx*16+jj];
        __syncthreads();
    }
    for (int ii=0;ii<16;ii++) for (int jj=0;jj<16;jj++) {
        int r=by*BM+ty*16+ii, c=bx*BN+tx*16+jj;
        if (r<M&&c<N) C[r*N+c]=acc[ii][jj];
    }
}
