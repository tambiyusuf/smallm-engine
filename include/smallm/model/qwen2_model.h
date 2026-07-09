#pragma once

#include "smallm/model/common/model.h"
#include "smallm/config/qwen2_config.h"
#include "smallm/core/gguf.h"
#include "smallm/core/tensor.h"
#include "smallm/backend/common/backend.h"

#include <cstdint>
#include <memory>
#include <vector>

#include "common/model.h"

namespace smallm {

    // all dequantized weights for a single transformer layer
    struct LayerWeights {
        Tensor attn_norm;
        Tensor attn_q_w, attn_q_b;
        Tensor attn_k_w, attn_k_b;
        Tensor attn_v_w, attn_v_b;
        Tensor attn_out_w;

        Tensor ffn_norm;
        Tensor ffn_gate;
        Tensor ffn_up;
        Tensor ffn_down;
    };

    // per-layer key/value cache; grown one token at a time, pre-allocated up front.
    // this is the structure prefix caching will later build on.
    struct KVCache {
        // flat storage: [pos * kv_dim + d], where kv_dim = n_kv_heads * head_dim
        std::vector<float> k;
        std::vector<float> v;
        uint32_t kv_dim = 0;      // n_kv_heads * head_dim
        uint32_t max_seq = 0;     // allocated capacity in tokens

        void allocate(uint32_t layers, uint32_t kv_dim_, uint32_t max_seq_);
    };

    class Qwen2Model : public Model {
    public:
        explicit Qwen2Model(GGUFModel gguf);

        std::vector<float> forward(uint32_t token_id, uint32_t pos) override;
        void reset_cache() override;
        void truncate_cache(uint32_t) override;

    private:
        GGUFModel gguf_;
        std::unique_ptr<Backend> backend_;

        Tensor token_embd_;
        Tensor output_norm_;
        Tensor output_w_;

        std::vector<LayerWeights> layers_;

        // one KV cache per layer, pre-allocated to context_length
        std::vector<KVCache> kv_;

        void load_weights();
        void allocate_kv();

        // forward broken into pieces for clarity and debugging
        std::vector<float> attention(uint32_t layer, const std::vector<float>& x, uint32_t pos);
        std::vector<float> feed_forward(uint32_t layer, const std::vector<float>& x);
    };

} // namespace smallm