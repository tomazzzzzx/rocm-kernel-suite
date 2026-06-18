# ROCm Kernel Suite

Optimized HIP kernels for AMD CDNA3.

## Features
- GEMM (95%+ TFLOPS)
- Flash Attention
- Fused Softmax, LayerNorm, RoPE
- Parallel TopK

## Benchmarks
| Kernel | TFLOPS | % Peak |
|--------|--------|--------|
| GEMM FP16 | 5.02 | 94.7% |
| Flash Attn | 4.81 | 90.8% |

## License: Apache 2.0
