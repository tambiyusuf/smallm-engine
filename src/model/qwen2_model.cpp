//
// Created by tambiyusuf on 5.07.2026.
//
#include "smallm/model/qwen2_model.h"
#include "smallm/config/qwen2_config.h"
#include "smallm/backend/cpu_backend.h"

#include <cmath>
#include <algorithm>
#include <string>

#include "smallm/core/ops.h"

namespace smallm {

// ---- KVCache ----

    void Qwen2Model::allocate_kv() {
        uint32_t kv_dim = config_->n_kv_heads * config_->head_dim();
        kv_.resize(config_->n_layers);
        for (auto& c : kv_) {
            c.allocate(kv_dim, config_->context_length);   // was: c.allocate(n_layers, kv_dim, ...)
        }
    }

// ---- construction ----

Qwen2Model::Qwen2Model(GGUFModel gguf)
    : Model(read_qwen2_config(gguf)),
      gguf_(std::move(gguf)),
      backend_(std::make_unique<CPUBackend>()) {
    load_weights();
    allocate_kv();
}

void Qwen2Model::load_weights() {
    // small f32 weights stay dequantized
    token_embd_  = dequantize_tensor(gguf_, "token_embd.weight");
    output_norm_ = dequantize_tensor(gguf_, "output_norm.weight");

    // large output projection kept quantized
    output_w_ = get_quantized_tensor(gguf_, "output.weight");

    uint32_t n = config_->n_layers;
    layers_.resize(n);
    for (uint32_t i = 0; i < n; ++i) {
        LayerWeights& L = layers_[i];

        // small: norms and biases dequantized
        L.attn_norm = dequantize_tensor(gguf_, layer_tensor_name(i, "attn_norm.weight"));
        L.attn_q_b  = dequantize_tensor(gguf_, layer_tensor_name(i, "attn_q.bias"));
        L.attn_k_b  = dequantize_tensor(gguf_, layer_tensor_name(i, "attn_k.bias"));
        L.attn_v_b  = dequantize_tensor(gguf_, layer_tensor_name(i, "attn_v.bias"));
        L.ffn_norm  = dequantize_tensor(gguf_, layer_tensor_name(i, "ffn_norm.weight"));

        // large: kept quantized, dequantized on the fly during matmul
        L.attn_q_w   = get_quantized_tensor(gguf_, layer_tensor_name(i, "attn_q.weight"));
        L.attn_k_w   = get_quantized_tensor(gguf_, layer_tensor_name(i, "attn_k.weight"));
        L.attn_v_w   = get_quantized_tensor(gguf_, layer_tensor_name(i, "attn_v.weight"));
        L.attn_out_w = get_quantized_tensor(gguf_, layer_tensor_name(i, "attn_output.weight"));
        L.ffn_gate   = get_quantized_tensor(gguf_, layer_tensor_name(i, "ffn_gate.weight"));
        L.ffn_up     = get_quantized_tensor(gguf_, layer_tensor_name(i, "ffn_up.weight"));
        L.ffn_down   = get_quantized_tensor(gguf_, layer_tensor_name(i, "ffn_down.weight"));
    }
}

// ---- cache management ----

void Qwen2Model::reset_cache() {
    cached_length_ = 0;
}

void Qwen2Model::truncate_cache(uint32_t n) {
    if (n < cached_length_) cached_length_ = n;
}

// ---- attention ----
std::vector<float> Qwen2Model::attention(uint32_t layer,
                                         const std::vector<float>& x,
                                         uint32_t pos) {
    const auto& L = layers_[layer];
    const uint32_t embd    = config_->embd_dim;
    const uint32_t n_head  = config_->n_heads;
    const uint32_t n_kv    = config_->n_kv_heads;
    const uint32_t hd      = config_->head_dim();
    const uint32_t q_dim   = n_head * hd;
    const uint32_t kv_dim  = n_kv * hd;
    const uint32_t group   = n_head / n_kv;

    // project x into Q, K, V via on-the-fly dequantized matmul.
    // weights are [out, in] as consumed here (rows=out_dim, cols=embd).
    std::vector<float> q(q_dim), k(kv_dim), v(kv_dim);
    backend_->matmul_quantized(L.attn_q_w.data, L.attn_q_w.type, x.data(), q_dim, embd, q.data());
    backend_->matmul_quantized(L.attn_k_w.data, L.attn_k_w.type, x.data(), kv_dim, embd, k.data());
    backend_->matmul_quantized(L.attn_v_w.data, L.attn_v_w.type, x.data(), kv_dim, embd, v.data());

    // Qwen2 adds biases to Q, K, V (vectorized add)
    ops::axpy(q.data(), L.attn_q_b.data.data(), 1.0f, q_dim);
    ops::axpy(k.data(), L.attn_k_b.data.data(), 1.0f, kv_dim);
    ops::axpy(v.data(), L.attn_v_b.data.data(), 1.0f, kv_dim);

    // rotary embedding on Q and K
    backend_->rope(q.data(), n_head, hd, pos, config_->rope_freq_base);
    backend_->rope(k.data(), n_kv,   hd, pos, config_->rope_freq_base);

    // write this token's K and V into the cache at position `pos`
    KVCache& cache = kv_[layer];
    std::copy(k.begin(), k.end(), cache.k.begin() + static_cast<size_t>(pos) * kv_dim);
    std::copy(v.begin(), v.end(), cache.v.begin() + static_cast<size_t>(pos) * kv_dim);

    std::vector<float> out(q_dim, 0.0f);
    float scale = 1.0f / std::sqrt(static_cast<float>(hd));

    for (uint32_t h = 0; h < n_head; ++h) {
        const float* qh = q.data() + h * hd;
        uint32_t kvh = h / group;   // which kv head this query head shares (GQA)

        // scores over cached positions 0..pos (vectorized dot product)
        std::vector<float> scores(pos + 1);
        for (uint32_t t = 0; t <= pos; ++t) {
            const float* kt = cache.k.data() + static_cast<size_t>(t) * kv_dim + kvh * hd;
            scores[t] = ops::dot(qh, kt, hd) * scale;
        }

        backend_->softmax(scores.data(), pos + 1);

        // weighted sum of V (vectorized accumulate)
        float* oh = out.data() + h * hd;
        for (uint32_t t = 0; t <= pos; ++t) {
            const float* vt = cache.v.data() + static_cast<size_t>(t) * kv_dim + kvh * hd;
            ops::axpy(oh, vt, scores[t], hd);
        }
    }

    // output projection back to embd width (on-the-fly dequantized)
    std::vector<float> proj(embd);
    backend_->matmul_quantized(L.attn_out_w.data, L.attn_out_w.type,
                               out.data(), embd, q_dim, proj.data());
    return proj;
}

    // ---- feed-forward (SwiGLU) ----
    std::vector<float> Qwen2Model::feed_forward(uint32_t layer,
                                                const std::vector<float>& x) {
    const auto& L = layers_[layer];
    const uint32_t embd = config_->embd_dim;
    const uint32_t ffn  = config_->ffn_dim;

    std::vector<float> gate(ffn), up(ffn);
    backend_->matmul_quantized(L.ffn_gate.data, L.ffn_gate.type, x.data(), ffn, embd, gate.data());
    backend_->matmul_quantized(L.ffn_up.data,   L.ffn_up.type,   x.data(), ffn, embd, up.data());

    // silu on gate, then elementwise multiply with up (vectorized)
    backend_->silu(gate.data(), ffn);
    ops::mul(gate.data(), up.data(), ffn);   // was: scalar loop

    // down projection back to embd width
    std::vector<float> down(embd);
    backend_->matmul_quantized(L.ffn_down.data, L.ffn_down.type, gate.data(), embd, ffn, down.data());
    return down;
}
    // ---- full forward ----
    std::vector<float> Qwen2Model::forward(uint32_t token_id, uint32_t pos) {
    const uint32_t embd = config_->embd_dim;

    // embedding lookup: copy the token's row from the embedding table
    std::vector<float> x(embd);
    const float* emb_row = token_embd_.data.data() + static_cast<size_t>(token_id) * embd;
    std::copy(emb_row, emb_row + embd, x.begin());

    // transformer layers
    std::vector<float> normed(embd);
    for (uint32_t layer = 0; layer < config_->n_layers; ++layer) {
        // attention block with pre-norm and residual
        backend_->rmsnorm(x.data(), layers_[layer].attn_norm.data.data(),
                          embd, config_->rms_eps, normed.data());
        std::vector<float> attn = attention(layer, normed, pos);
        ops::axpy(x.data(), attn.data(), 1.0f, embd);   // residual: x += attn

        // feed-forward block with pre-norm and residual
        backend_->rmsnorm(x.data(), layers_[layer].ffn_norm.data.data(),
                          embd, config_->rms_eps, normed.data());
        std::vector<float> ff = feed_forward(layer, normed);
        ops::axpy(x.data(), ff.data(), 1.0f, embd);     // residual: x += ff
    }

    // final norm
    backend_->rmsnorm(x.data(), output_norm_.data.data(),
                      embd, config_->rms_eps, normed.data());

    // output projection to logits (on-the-fly dequantized)
    std::vector<float> logits(config_->vocab_size);
    backend_->matmul_quantized(output_w_.data, output_w_.type,
                               normed.data(), config_->vocab_size, embd, logits.data());

    cached_length_ = pos + 1;
    return logits;
}

} // namespace smallm