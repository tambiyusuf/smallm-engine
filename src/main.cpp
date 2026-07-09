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

        // same system prefix, two different questions:
        // the second request should reuse the shared prefix from the first
        std::string sys = "You are a helpful assistant. ";

        std::cout << "--- first request ---\n";
        std::string p1 = sys + "What is the capital of France?";
        std::cout << p1 << gen.generate(p1, 15) << "\n";

        std::cout << "\n--- second request (same prefix) ---\n";
        std::string p2 = sys + "What is the capital of Japan?";
        std::cout << p2 << gen.generate(p2, 15) << "\n";

    } catch (const std::exception& e) {
        std::cerr << "error: " << e.what() << "\n";
        return 1;
    }
    return 0;
}