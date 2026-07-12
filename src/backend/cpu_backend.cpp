//
// Created by tambiyusuf on 5.07.2026.
//
#include "smallm/backend/cpu_backend.h"
#include "smallm/core/ops.h"

namespace smallm {

    // each method forwards to the corresponding scalar op; this keeps the tested
    // reference math in one place while giving forward a backend-shaped interface

    void CPUBackend::matmul(const float* W, const float* x,
                            uint32_t rows, uint32_t cols, float* y) {
        ops::matmul(W, x, rows, cols, y);
    }

    void CPUBackend::matmul_quantized(const uint8_t* W, uint32_t type,
                                  const float* x, uint32_t rows, uint32_t cols,
                                  float* y) {
        ops::matmul_quantized(W, type, x, rows, cols, y);
    }

    void CPUBackend::rmsnorm(const float* x, const float* weight,
                             uint32_t size, float eps, float* y) {
        ops::rmsnorm(x, weight, size, eps, y);
    }

    void CPUBackend::softmax(float* x, uint32_t size) {
        ops::softmax(x, size);
    }

    void CPUBackend::silu(float* x, uint32_t size) {
        ops::silu(x, size);
    }

    void CPUBackend::rope(float* x, uint32_t n_heads, uint32_t head_dim,
                          uint32_t pos, float freq_base) {
        ops::rope(x, n_heads, head_dim, pos, freq_base);
    }

} // namespace smallm