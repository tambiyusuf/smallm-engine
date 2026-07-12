//
// Created by tambiyusuf on 6.07.2026.
//
#include "smallm/runtime/generator.h"

#include <algorithm>
#include <chrono>

namespace smallm {

    static uint32_t common_prefix_length(const std::vector<uint32_t>& a,
                                     const std::vector<uint32_t>& b) {
        uint32_t n = 0;
        uint32_t limit = std::min<uint32_t>(a.size(), b.size());
        while (n < limit && a[n] == b[n]) ++n;
        return n;
    }

    Generator::Generator(Model& model, const Tokenizer& tokenizer, Sampler& sampler)
        : model_(model), tokenizer_(tokenizer), sampler_(sampler) {}

std::string Generator::generate(const std::string& prompt, uint32_t max_new_tokens) {
    using clock = std::chrono::steady_clock;
    auto ms = [](auto a, auto b) {
        return std::chrono::duration<double, std::milli>(b - a).count();
    };

    stats_ = GenerationStats{};   // reset for this call

    std::vector<uint32_t> tokens = tokenizer_.encode(prompt);
    if (tokens.empty()) return "";

    stats_.prompt_tokens = static_cast<uint32_t>(tokens.size());

    uint32_t shared = common_prefix_length(cached_tokens_, tokens);
    if (shared >= tokens.size()) shared = static_cast<uint32_t>(tokens.size()) - 1;
    stats_.reused_tokens = shared;

    model_.truncate_cache(shared);

    std::string output;
    uint32_t pos = shared;

    // prefill phase
    auto prefill_start = clock::now();
    std::vector<float> logits;
    for (uint32_t i = shared; i < tokens.size(); ++i) {
        logits = model_.forward(tokens[i], pos);
        ++pos;
        ++stats_.prefill_tokens;
    }
    stats_.prefill_ms = ms(prefill_start, clock::now());

    int32_t eos = tokenizer_.eos_id();

    // decode phase
    auto decode_start = clock::now();
    std::vector<uint32_t> generated = tokens;
    for (uint32_t step = 0; step < max_new_tokens; ++step) {
        uint32_t next = sampler_.sample(logits);
        if (eos >= 0 && next == static_cast<uint32_t>(eos)) break;
        output += tokenizer_.decode({next});
        generated.push_back(next);
        logits = model_.forward(next, pos);
        ++pos;
        ++stats_.generated_tokens;
    }
    stats_.decode_ms = ms(decode_start, clock::now());

    cached_tokens_ = std::move(generated);
    return output;
}

} // namespace smallm
