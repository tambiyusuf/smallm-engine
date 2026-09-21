#include "smallm/backend/cuda_backend.h"

#include <cuda_runtime.h>
#include <cmath>
#include <cstring>
#include <stdexcept>

namespace smallm {

namespace {

    void cuda_check(cudaError_t err, const char* what) {
        if (err != cudaSuccess) {
            throw std::runtime_error(std::string(what) + ": " + cudaGetErrorString(err));
        }
    }

    __device__ float f16_to_f32_dev(uint16_t h) {
        uint32_t sign = (h & 0x8000u) << 16;
        uint32_t exp = (h >> 10) & 0x1F;
        uint32_t mant = h & 0x3FF;
        uint32_t bits;
        if (exp == 0) {
            bits = (mant == 0) ? sign : 0; // subnormals simplified
            if (mant != 0) {
                exp = 127 - 15 + 1;
                while ((mant & 0x400) == 0) {
                    mant <<= 1;
                    --exp;
                }
                mant &= 0x3FF;
                bits = sign | (exp << 23) | (mant << 13);
            }
        } else if (exp == 0x1F) {
            bits = sign | 0x7F800000u | (mant << 13);
        } else {
            bits = sign | ((exp - 15 + 127) << 23) | (mant << 13);
        }
        float out;
        memcpy(&out, &bits, sizeof(out));
        return out;
    }

    __global__ void matmul_q4_0_kernel(const uint8_t* W, const float* x, float* y,
                                       uint32_t rows, uint32_t cols) {
        const int block = 32;
        const uint32_t nblocks = cols / block;
        const uint32_t row_bytes = nblocks * (2 + block / 2);

        uint32_t r = blockIdx.x * blockDim.x + threadIdx.x;
        if (r >= rows) {
            return;
        }

        const uint8_t* row = W + static_cast<uint64_t>(r) * row_bytes;
        float acc = 0.0f;

        for (uint32_t b = 0; b < nblocks; ++b) {
            uint16_t sb;
            memcpy(&sb, row, 2);
            float scale = f16_to_f32_dev(sb);
            const uint8_t* q = row + 2;
            const float* xb = x + b * block;

            for (int i = 0; i < 16; ++i) {
                uint8_t packed = q[i];
                int lo = (packed & 0x0F) - 8;
                int hi = (packed >> 4) - 8;
                acc += scale * static_cast<float>(lo) * xb[i];
                acc += scale * static_cast<float>(hi) * xb[16 + i];
            }
            row += 2 + block / 2;
        }
        y[r] = acc;
    }

} // namespace

CUDABackend::CUDABackend() : cpu_fallback_(std::make_unique<CPUBackend>()) {
    cuda_check(cudaSetDevice(0), "cudaSetDevice");
}

CUDABackend::~CUDABackend() {
    for (auto& entry : device_weights_) {
        cudaFree(entry.second);
    }
}

void* CUDABackend::device_weight(const uint8_t* host, uint64_t nbytes) {
    auto it = device_weights_.find(host);
    if (it != device_weights_.end()) {
        return it->second;
    }
    void* dev = nullptr;
    cuda_check(cudaMalloc(&dev, nbytes), "cudaMalloc weight");
    cuda_check(cudaMemcpy(dev, host, nbytes, cudaMemcpyHostToDevice), "cudaMemcpy weight");
    device_weights_[host] = dev;
    return dev;
}

void CUDABackend::matmul(const float* W, const float* x,
                         uint32_t rows, uint32_t cols, float* y) {
    cpu_fallback_->matmul(W, x, rows, cols, y);
}

void CUDABackend::matmul_quantized(const uint8_t* W, uint32_t type,
                                   const float* x, uint32_t rows, uint32_t cols,
                                   float* y) {
    if (type != 2) {
        cpu_fallback_->matmul_quantized(W, type, x, rows, cols, y);
        return;
    }

    const int block = 32;
    const uint32_t nblocks = cols / block;
    const uint64_t weight_bytes = static_cast<uint64_t>(rows) * nblocks * (2 + block / 2);
    const uint8_t* W_dev = static_cast<const uint8_t*>(device_weight(W, weight_bytes));

    float* x_dev = nullptr;
    float* y_dev = nullptr;
    cuda_check(cudaMalloc(&x_dev, cols * sizeof(float)), "cudaMalloc x");
    cuda_check(cudaMalloc(&y_dev, rows * sizeof(float)), "cudaMalloc y");
    cuda_check(cudaMemcpy(x_dev, x, cols * sizeof(float), cudaMemcpyHostToDevice), "cudaMemcpy x");

    int threads = 256;
    int blocks = (static_cast<int>(rows) + threads - 1) / threads;
    matmul_q4_0_kernel<<<blocks, threads>>>(W_dev, x_dev, y_dev, rows, cols);
    cuda_check(cudaGetLastError(), "matmul_q4_0_kernel");
    cuda_check(cudaMemcpy(y, y_dev, rows * sizeof(float), cudaMemcpyDeviceToHost), "cudaMemcpy y");

    cudaFree(x_dev);
    cudaFree(y_dev);
}

void CUDABackend::rmsnorm(const float* x, const float* weight,
                          uint32_t size, float eps, float* y) {
    cpu_fallback_->rmsnorm(x, weight, size, eps, y);
}

void CUDABackend::softmax(float* x, uint32_t size) {
    cpu_fallback_->softmax(x, size);
}

void CUDABackend::silu(float* x, uint32_t size) {
    cpu_fallback_->silu(x, size);
}

void CUDABackend::rope(float* x, uint32_t n_heads, uint32_t head_dim,
                       uint32_t pos, float freq_base) {
    cpu_fallback_->rope(x, n_heads, head_dim, pos, freq_base);
}

} // namespace smallm
