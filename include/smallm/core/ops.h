//
// Created by tambiyusuf on 4.07.2026.
//

#pragma once
#include <cstdint>
#include <vector>

namespace smallm::ops {

    // matrix-vector product: y = W + x
    // W is row-major with shape [rows, columns], x has 'columns' elements,
    // result y has 'rows' elements. this is the shape most weigth tensors need.
    void matmul(const float* W, const float* x, uint32_t rows, uint32_t cols, float* y);

    // y = W * x, where W is a quantized weight matrix [rows, cols] dequantized
    // row-by-row on the fly. x is f32 of length cols, y is f32 of length rows.
    void matmul_quantized(const uint8_t* W, uint32_t type, const float* x, uint32_t rows, uint32_t cols,
                          float* y);
    // RMS normalization: y (x / rms(x) * weight, where rms(x) = sqrt (mean(x^2) + eps )
    // x, weight and y all have `size` elements. normalizes over the whole vector.
    void rmsnorm(const float* x, const float* weight, uint32_t size, float eps, float* y);

    // softmax in place over `size` elements: turns scores into a probability distribution
    void softmax(float* x, uint32_t size);

    // SiLU activation in place: x[i] = x[i] * sigmoid(x[i]); used in the gated MLP
    void silu(float* x, uint32_t size);

    // applies rotary position embedding (RoPE) to a vector split into heads.
    // rotates each pair of dimensions by an angle that depends on position and frequency.
    // x has `n_heads * head_dim` elements; head_dim must be even.
    void rope(float*x, uint32_t n_heads, uint32_t head_dim, uint32_t pos, float freq_base);

    // dot product of two f32 vectors of length n
    float dot(const float* a, const float* b, uint32_t n);

    // y += scale * v, elementwise over n floats (accumulating weighted sum)
    void axpy(float* y, const float* v, float scale, uint32_t n);


    void mul(float* y, const float* v, uint32_t n);   // y[i] *= v[i]

}