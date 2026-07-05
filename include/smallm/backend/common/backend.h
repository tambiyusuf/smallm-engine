//
// Created by tambiyusuf on 5.07.2026.
//
#pragma once

#include <cstdint>

namespace smallm {

    // abstract compute backend; the forward pass talks only to this interface,
    // never to raw ops. CPU implements it now; CUDA/ROCm slot in later untouched.
    class Backend {
    public:
        virtual ~Backend() = default;

        // y = W * x, with W row-major [rows, cols], x of size cols, y of size rows
        virtual void matmul(const float* W, const float* x,
                            uint32_t rows, uint32_t cols, float* y) = 0;

        // y = rmsnorm(x) * weight, over `size` elements
        virtual void rmsnorm(const float* x, const float* weight,
                            uint32_t size, float eps, float* y) = 0;

        // softmax in place over `size` elements
        virtual void softmax(float* x, uint32_t size) = 0;

        // SiLU in place over `size` elements
        virtual void silu(float* x, uint32_t size) = 0;

        // rotary position embedding in place, per head
        virtual void rope(float* x, uint32_t n_heads, uint32_t head_dim,
                         uint32_t pos, float freq_base) = 0;
    };

} // namespace smallm