//
// Created by tambiyusuf on 13.07.2026.
//

#pragma once

#include <cstdint>
#include <vector>

namespace smallm {

    // per-layer key/value cache, pre-allocated to the model's context length.
    // storage is flat and position-indexed: k[pos * kv_dim + d].
    // this is the structure prefix caching builds on.
    struct KVCache {
        std::vector<float> k;
        std::vector<float> v;
        uint32_t kv_dim = 0;    // n_kv_heads * head_dim
        uint32_t max_seq = 0;   // allocated capacity in tokens

        void allocate(uint32_t kv_dim_, uint32_t max_seq_) {
            kv_dim = kv_dim_;
            max_seq = max_seq_;
            k.assign(static_cast<size_t>(max_seq) * kv_dim, 0.0f);
            v.assign(static_cast<size_t>(max_seq) * kv_dim, 0.0f);
        }
    };

} // namespace smallm