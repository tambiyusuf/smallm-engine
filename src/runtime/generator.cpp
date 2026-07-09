//
// Created by tambiyusuf on 6.07.2026.
//
#include "smallm/runtime/generator.h"

#include <iostream>

namespace smallm {

    Generator::Generator(Model& model, const BPETokenizer& tokenizer, Sampler& sampler)
        : model_(model), tokenizer_(tokenizer), sampler_(sampler) {}

    // finds how many leading tokens two sequences share
static uint32_t common_prefix_length(const std::vector<uint32_t>& a,
                                     const std::vector<uint32_t>& b) {
    uint32_t n = 0;
    uint32_t limit = std::min<uint32_t>(a.size(), b.size());
    while (n < limit && a[n] == b[n]) ++n;
    return n;
}

std::string Generator::generate(const std::string& prompt, uint32_t max_new_tokens) {
    std::vector<uint32_t> tokens = tokenizer_.encode(prompt);
    if (tokens.empty()) return "";

    // prefix caching: reuse the cached positions this prompt shares with the last
    uint32_t shared = common_prefix_length(cached_tokens_, tokens);

    // never reuse the entire sequence — we need at least the last token's logits,
    // so keep at most tokens.size() - 1 as reused prefix
    if (shared >= tokens.size()) shared = static_cast<uint32_t>(tokens.size()) - 1;

    // keep the shared portion of the cache, discard the rest
    model_.truncate_cache(shared);

    std::string output;
    uint32_t pos = shared;   // start prefilling from the first non-shared token

    // prefill the remaining prompt tokens
    std::vector<float> logits;
    for (uint32_t i = shared; i < tokens.size(); ++i) {
        logits = model_.forward(tokens[i], pos);
        ++pos;
    }

    // report how much we skipped (helps verify prefix caching works)
    // (comment out in production; useful during testing)
    std::cerr << "[prefix] reused " << shared << " tokens, prefilled "
               << (tokens.size() - shared) << "\n";

    int32_t eos = tokenizer_.eos_id();

    // decode loop
    std::vector<uint32_t> generated = tokens;   // track full sequence for next call's cache
    for (uint32_t step = 0; step < max_new_tokens; ++step) {
        uint32_t next = sampler_.sample(logits);
        if (eos >= 0 && next == static_cast<uint32_t>(eos)) break;

        output += tokenizer_.decode({next});
        generated.push_back(next);

        logits = model_.forward(next, pos);
        ++pos;
    }

    // remember the full sequence so the next call can reuse its prefix
    cached_tokens_ = std::move(generated);
    return output;
}

} // namespace smallm