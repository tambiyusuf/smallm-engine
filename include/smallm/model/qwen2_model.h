#pragma once

#include "smallm/model/common/model.h"
#include "smallm/config/qwen2_config.h"
#include "smallm/core/gguf.h"
#include "smallm/core/tensor.h"
#include "smallm/backend/common/backend.h"
#include "smallm/model/common/kv_cache.h"
#include <cstdint>
#include <memory>
#include <vector>

namespace smallm {

    // weights for a single transformer layer. small tensors (norms, biases) are
    // kept dequantized; large matmul weights stay quantized and are dequantized
    // on the fly during matmul.
    struct LayerWeights {
        // small f32 weights
        Tensor attn_norm;
        Tensor attn_q_b, attn_k_b, attn_v_b;
        Tensor ffn_norm;

        // large quantized weights
        QuantizedTensor attn_q_w, attn_k_w, attn_v_w, attn_out_w;
        QuantizedTensor ffn_gate, ffn_up, ffn_down;
    };

    class Qwen2Model : public Model {
    public:
        explicit Qwen2Model(GGUFModel gguf);

        std::vector<float> forward(uint32_t token_id, uint32_t pos) override;
        void reset_cache() override;
        void truncate_cache(uint32_t n) override;

    private:
        GGUFModel gguf_;
        std::unique_ptr<Backend> backend_;

        // model-wide weights
        Tensor token_embd_;          // f32, used by row lookup (not matmul)
        Tensor output_norm_;         // f32
        QuantizedTensor output_w_;   // large output projection, kept quantized

        std::vector<LayerWeights> layers_;
        std::vector<KVCache> kv_;

        void load_weights();
        void allocate_kv();

        std::vector<float> attention(uint32_t layer, const std::vector<float>& x, uint32_t pos);
        std::vector<float> feed_forward(uint32_t layer, const std::vector<float>& x);
    };

} // namespace smallm