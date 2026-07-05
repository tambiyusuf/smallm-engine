//
// Created by tambiyusuf on 5.07.2026.
//
#include "smallm/model/qwen2_model.h"
#include "smallm/config/qwen2_config.h"
#include "smallm/backend/cpu_backend.h"

#include <cmath>
#include <stdexcept>
#include <string>

namespace smallm {

// ---- KVCache ----

void KVCache::allocate(uint32_t /*layers*/, uint32_t kv_dim_, uint32_t max_seq_) {
    kv_dim = kv_dim_;
    max_seq = max_seq_;
    k.assign(static_cast<size_t>(max_seq) * kv_dim, 0.0f);
    v.assign(static_cast<size_t>(max_seq) * kv_dim, 0.0f);
}

// ---- construction ----

Qwen2Model::Qwen2Model(GGUFModel gguf)
    : Model(read_qwen2_config(gguf)),
      gguf_(std::move(gguf)),
      backend_(std::make_unique<CPUBackend>()) {
    load_weights();
    allocate_kv();
}

static std::string layer_name(uint32_t i, const std::string& suffix) {
    return "blk." + std::to_string(i) + "." + suffix;
}

void Qwen2Model::load_weights() {
    token_embd_  = dequantize_tensor(gguf_, "token_embd.weight");
    output_norm_ = dequantize_tensor(gguf_, "output_norm.weight");
    try {
        output_w_ = dequantize_tensor(gguf_, "output.weight");
    } catch (const std::exception&) {
        output_w_ = Tensor{};  // tied to token_embd_
    }

    uint32_t n = config_->n_layers;
    layers_.resize(n);
    for (uint32_t i = 0; i < n; ++i) {
        LayerWeights& L = layers_[i];
        L.attn_norm  = dequantize_tensor(gguf_, layer_name(i, "attn_norm.weight"));
        L.attn_q_w   = dequantize_tensor(gguf_, layer_name(i, "attn_q.weight"));
        L.attn_q_b   = dequantize_tensor(gguf_, layer_name(i, "attn_q.bias"));
        L.attn_k_w   = dequantize_tensor(gguf_, layer_name(i, "attn_k.weight"));
        L.attn_k_b   = dequantize_tensor(gguf_, layer_name(i, "attn_k.bias"));
        L.attn_v_w   = dequantize_tensor(gguf_, layer_name(i, "attn_v.weight"));
        L.attn_v_b   = dequantize_tensor(gguf_, layer_name(i, "attn_v.bias"));
        L.attn_out_w = dequantize_tensor(gguf_, layer_name(i, "attn_output.weight"));
        L.ffn_norm   = dequantize_tensor(gguf_, layer_name(i, "ffn_norm.weight"));
        L.ffn_gate   = dequantize_tensor(gguf_, layer_name(i, "ffn_gate.weight"));
        L.ffn_up     = dequantize_tensor(gguf_, layer_name(i, "ffn_up.weight"));
        L.ffn_down   = dequantize_tensor(gguf_, layer_name(i, "ffn_down.weight"));
    }
}

void Qwen2Model::allocate_kv() {
    uint32_t kv_dim = config_->n_kv_heads * config_->head_dim();
    kv_.resize(config_->n_layers);
    for (auto& c : kv_) {
        c.allocate(config_->n_layers, kv_dim, config_->context_length);
    }
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
    const uint32_t q_dim   = n_head * hd;   // full query width
    const uint32_t kv_dim  = n_kv * hd;     // key/value width (smaller, GQA)
    const uint32_t group   = n_head / n_kv; // query heads per kv head

    // project x into Q, K, V. weights are [out, in] row-major, so matmul fits directly.
    std::vector<float> q(q_dim), k(kv_dim), v(kv_dim);
    backend_->matmul(L.attn_q_w.data.data(), x.data(), q_dim, embd, q.data());
    backend_->matmul(L.attn_k_w.data.data(), x.data(), kv_dim, embd, k.data());
    backend_->matmul(L.attn_v_w.data.data(), x.data(), kv_dim, embd, v.data());

    // Qwen2 adds biases to Q, K, V
    for (uint32_t i = 0; i < q_dim; ++i)  q[i] += L.attn_q_b.data[i];
    for (uint32_t i = 0; i < kv_dim; ++i) k[i] += L.attn_k_b.data[i];
    for (uint32_t i = 0; i < kv_dim; ++i) v[i] += L.attn_v_b.data[i];

    // rotary embedding on Q and K, per head
    backend_->rope(q.data(), n_head, hd, pos, config_->rope_freq_base);
    backend_->rope(k.data(), n_kv,   hd, pos, config_->rope_freq_base);

    // write this token's K and V into the cache at position `pos`
    KVCache& cache = kv_[layer];
    std::copy(k.begin(), k.end(), cache.k.begin() + static_cast<size_t>(pos) * kv_dim);
    std::copy(v.begin(), v.end(), cache.v.begin() + static_cast<size_t>(pos) * kv_dim);

    // attention output accumulates here, per query head
    std::vector<float> out(q_dim, 0.0f);
    float scale = 1.0f / std::sqrt(static_cast<float>(hd));

    // for each query head, attend over all cached positions up to pos
    for (uint32_t h = 0; h < n_head; ++h) {
        const float* qh = q.data() + h * hd;
        uint32_t kvh = h / group;   // which kv head this query head uses

        // scores over positions 0..pos
        std::vector<float> scores(pos + 1);
        for (uint32_t t = 0; t <= pos; ++t) {
            const float* kt = cache.k.data() + static_cast<size_t>(t) * kv_dim + kvh * hd;
            float dot = 0.0f;
            for (uint32_t d = 0; d < hd; ++d) dot += qh[d] * kt[d];
            scores[t] = dot * scale;
        }

        // softmax over the scores
        backend_->softmax(scores.data(), pos + 1);

        // weighted sum of V
        float* oh = out.data() + h * hd;
        for (uint32_t t = 0; t <= pos; ++t) {
            const float* vt = cache.v.data() + static_cast<size_t>(t) * kv_dim + kvh * hd;
            float w = scores[t];
            for (uint32_t d = 0; d < hd; ++d) oh[d] += w * vt[d];
        }
    }

    // output projection back to embd width
    std::vector<float> proj(embd);
    backend_->matmul(L.attn_out_w.data.data(), out.data(), embd, q_dim, proj.data());
    return proj;
}

// ---- feed-forward (SwiGLU) ----

std::vector<float> Qwen2Model::feed_forward(uint32_t layer,
                                            const std::vector<float>& x) {
    const auto& L = layers_[layer];
    const uint32_t embd = config_->embd_dim;
    const uint32_t ffn  = config_->ffn_dim;

    std::vector<float> gate(ffn), up(ffn);
    backend_->matmul(L.ffn_gate.data.data(), x.data(), ffn, embd, gate.data());
    backend_->matmul(L.ffn_up.data.data(),   x.data(), ffn, embd, up.data());

    // silu on gate, then elementwise multiply with up
    backend_->silu(gate.data(), ffn);
    for (uint32_t i = 0; i < ffn; ++i) gate[i] *= up[i];

    // down projection back to embd width
    std::vector<float> down(embd);
    backend_->matmul(L.ffn_down.data.data(), gate.data(), embd, ffn, down.data());
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
        for (uint32_t i = 0; i < embd; ++i) x[i] += attn[i];

        // feed-forward block with pre-norm and residual
        backend_->rmsnorm(x.data(), layers_[layer].ffn_norm.data.data(),
                          embd, config_->rms_eps, normed.data());
        std::vector<float> ff = feed_forward(layer, normed);
        for (uint32_t i = 0; i < embd; ++i) x[i] += ff[i];
    }

    // final norm
    backend_->rmsnorm(x.data(), output_norm_.data.data(),
                      embd, config_->rms_eps, normed.data());

    // output projection to logits; fall back to tied embedding if no output.weight
    const float* out_w = output_w_.data.empty()
                       ? token_embd_.data.data()
                       : output_w_.data.data();
    std::vector<float> logits(config_->vocab_size);
    backend_->matmul(out_w, normed.data(), config_->vocab_size, embd, logits.data());
    return logits;
}

} // namespace smallm} // namespace smallm