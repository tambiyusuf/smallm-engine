//
// Created by tambiyusuf on 4.07.2026.
//
#include "smallm/core/gguf.h"
#include "smallm/model/qwen2_model.h"
#include "smallm/tokenizer/common/tokenizer.h"
#include "smallm/runtime/generator.h"
#include "smallm/runtime/sampler/greedy_sampler.h"

#include <iostream>

int main(int argc, char** argv) {
    if (argc < 2) { std::cerr << "usage: smallm <model.gguf>\n"; return 1; }
    try {
        smallm::GGUFModel model = smallm::load_gguf(argv[1]);
        auto tokenizer = smallm::build_tokenizer(model);
        smallm::Qwen2Model qwen(std::move(model));

        smallm::GreedySampler sampler;
        smallm::Generator gen(qwen, *tokenizer, sampler);

        std::string prompt = "The capital of France is";
        std::string out = gen.generate(prompt, 30);
        std::cout << prompt << out << "\n\n";

        const auto& s = gen.last_stats();
        std::cout << "decode: " << s.generated_tokens << " tok ("
                  << s.decode_tokens_per_sec() << " tok/s)\n";

    } catch (const std::exception& e) {
        std::cerr << "error: " << e.what() << "\n";
        return 1;
    }
    return 0;
}