//
// Created by tambiyusuf on 5.07.2026.
//
#pragma once

#include "smallm/backend/common/backend.h"

namespace smallm {

    // CPU backend: delegates each operation to the scalar ops already written
    // and tested. later this is where SIMD / OpenMP versions can live.
    class CPUBackend : public Backend {
    public:
        void matmul(const float* W, const float* x,
                    uint32_t rows, uint32_t cols, float* y) override;
        
        void matmul_quantized(const uint8_t* W, uint32_t type,
                          const float* x, uint32_t rows, uint32_t cols,
                          float* y) override;
        void rmsnorm(const float* x, const float* weight,
                     uint32_t size, float eps, float* y) override;

        void softmax(float* x, uint32_t size) override;

        void silu(float* x, uint32_t size) override;

        void rope(float* x, uint32_t n_heads, uint32_t head_dim,
                  uint32_t pos, float freq_base) override;
    };

} // namespace smallm
