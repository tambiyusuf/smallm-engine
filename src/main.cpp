//
// Created by tambiyusuf on 4.07.2026.
//
#include "smallm/core/gguf.h"
#include "smallm/model/llama_model.h"
#include "smallm/tokenizer/common/tokenizer.h"
#include "smallm/runtime/generator.h"
#include "smallm/runtime/sampler/greedy_sampler.h"

#include <iostream>

int main(int argc, char** argv) {
    if (argc < 2) { std::cerr << "usage: smallm <model.gguf>\n"; return 1; }
    try {
        smallm::GGUFModel model = smallm::load_gguf(argv[1]);

        auto tokenizer = smallm::build_tokenizer(model);
        smallm::LlamaModel llama(std::move(model));

        // quick tokenizer round-trip check
        std::string text = "The capital of France is";
        auto ids = tokenizer->encode(text);
        std::cout << "encode: [";
        for (size_t i = 0; i < ids.size(); ++i) {
            std::cout << ids[i];
            if (i + 1 < ids.size()) std::cout << ", ";
        }
        std::cout << "]\n";
        std::cout << "round-trip: \"" << tokenizer->decode(ids) << "\"\n\n";

        smallm::GreedySampler sampler;
        smallm::Generator gen(llama, *tokenizer, sampler);

        std::string out = gen.generate(text, 40);
        std::cout << text << out << "\n\n";

        const auto& s = gen.last_stats();
        std::cout << "decode: " << s.generated_tokens << " tok ("
                  << s.decode_tokens_per_sec() << " tok/s)\n";

    } catch (const std::exception& e) {
        std::cerr << "error: " << e.what() << "\n";
        return 1;
    }
    return 0;
}