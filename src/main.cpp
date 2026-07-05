//
// Created by tambiyusuf on 4.07.2026.
//
#include "smallm/core/gguf.h"
#include "smallm/model/qwen2_model.h"
#include "smallm/tokenizer/bpe_tokenizer.h"
#include "smallm/runtime/generator.h"
#include "smallm/runtime/sampler/greedy_sampler.h"

#include <iostream>

int main(int argc, char** argv) {
    if (argc < 2) { std::cerr << "usage: smallm <model.gguf>\n"; return 1; }
    try {
        smallm::GGUFModel model = smallm::load_gguf(argv[1]);

        // build tokenizer BEFORE moving model into the engine
        smallm::BPETokenizer tokenizer(model);
        smallm::Qwen2Model qwen(std::move(model));

        smallm::GreedySampler sampler;
        smallm::Generator gen(qwen, tokenizer, sampler);

        std::string prompt = "The capital of France is";
        std::cout << "prompt: " << prompt << "\n";
        std::cout << "output: " << prompt << gen.generate(prompt, 20) << "\n";

    } catch (const std::exception& e) {
        std::cerr << "error: " << e.what() << "\n";
        return 1;
    }
    return 0;
}