//
// Created by tambiyusuf on 6.07.2026.
//
#include "smallm/runtime/generator.h"

namespace smallm {

    Generator::Generator(Model& model, const BPETokenizer& tokenizer, Sampler& sampler)
        : model_(model), tokenizer_(tokenizer), sampler_(sampler) {}

    std::string Generator::generate(const std::string& prompt, uint32_t max_new_tokens) {
        // encode the prompt into token ids
        std::vector<uint32_t> tokens = tokenizer_.encode(prompt);
        if (tokens.empty()) return "";

        std::string output;
        uint32_t pos = 0;

        // prefill: feed every prompt token, advancing the KV cache.
        // only the last token's logits matter for the first prediction.
        std::vector<float> logits;
        for (uint32_t t : tokens) {
            logits = model_.forward(t, pos);
            ++pos;
        }

        int32_t eos = tokenizer_.eos_id();

        // decode loop: sample next token, append, feed it back
        for (uint32_t step = 0; step < max_new_tokens; ++step) {
            uint32_t next = sampler_.sample(logits);

            // stop at end-of-sequence
            if (eos >= 0 && next == static_cast<uint32_t>(eos)) break;

            // decode just this token and append to the output text
            output += tokenizer_.decode({next});

            // feed the new token back to get the following logits
            logits = model_.forward(next, pos);
            ++pos;
        }

        return output;
    }

} // namespace smallm