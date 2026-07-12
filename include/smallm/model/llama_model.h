//
// Created by tambiyusuf on 12.07.2026.
//
#pragma once

#include "smallm/model/common/model.h"
#include "smallm/config/llama_config.h"
#include "smallm/core/gguf.h"
#include "smallm/core/tensor.h"
#include "smallm/backend/common/backend.h"
#include "smallm/model/common/kv_cache.h"

#include <cstdint>
#include <memory>
#include <vector>

namespace smallm {

    // weights for one Llama transformer layer. unlike Qwen2, attention has no biases.
    struct LlamaLayerWeights {
        Tensor attn_norm;
        Tensor ffn_norm;

        QuantizedTensor attn_q_w, attn_k_w, attn_v_w, attn_out_w;
        QuantizedTensor ffn_gate, ffn_up, ffn_down;
    };


    class LlamaModel : public Model {
    public:
        explicit LlamaModel(GGUFModel gguf);

        std::vector<float> forward(uint32_t token_id, uint32_t pos) override;
        void reset_cache() override;
        void truncate_cache(uint32_t n) override;

    private:
        GGUFModel gguf_;
        std::unique_ptr<Backend> backend_;

        Tensor token_embd_;
        Tensor output_norm_;
        QuantizedTensor output_w_;

        std::vector<LlamaLayerWeights> layers_;
        std::vector<KVCache> kv_;

        void load_weights();
        void allocate_kv();

        std::vector<float> attention(uint32_t layer, const std::vector<float>& x, uint32_t pos);
        std::vector<float> feed_forward(uint32_t layer, const std::vector<float>& x);
    };

} // namespace smallm