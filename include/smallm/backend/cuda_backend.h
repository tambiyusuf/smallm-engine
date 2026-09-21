#pragma once

#include "smallm/backend/common/backend.h"
#include "smallm/backend/cpu_backend.h"

#include <cstdint>
#include <memory>
#include <unordered_map>

namespace smallm {

    // GPU backend: quantized matmul on device; other ops use GPU with host sync for now
    class CUDABackend : public Backend {
    public:
        CUDABackend();
        ~CUDABackend() override;

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

    private:
        std::unique_ptr<CPUBackend> cpu_fallback_;
        // lazily uploads each distinct weight pointer once for the model lifetime
        std::unordered_map<const uint8_t*, void*> device_weights_;

        void* device_weight(const uint8_t* host, uint64_t nbytes);
    };

} // namespace smallm
