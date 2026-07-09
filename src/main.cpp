//
// Created by tambiyusuf on 4.07.2026.
//
#include "smallm/core/gguf.h"
#include "smallm/model/qwen2_model.h"
#include "smallm/tokenizer/bpe_tokenizer.h"
#include "smallm/runtime/generator.h"
#include "smallm/runtime/sampler/greedy_sampler.h"
#include "smallm/runtime/sampler/temperature_sampler.h"
#include "smallm/runtime/sampler/top_k_sampler.h"
#include "smallm/runtime/sampler/top_p_sampler.h"

#include <iostream>

int main(int argc, char** argv) {
    if (argc < 2) { std::cerr << "usage: smallm <model.gguf>\n"; return 1; }
    try {
        smallm::GGUFModel model = smallm::load_gguf(argv[1]);

        // build tokenizer BEFORE moving model into the engine
        smallm::BPETokenizer tokenizer(model);
        smallm::Qwen2Model qwen(std::move(model));

        std::string prompt = "The capital of France is";

        // greedy: deterministic, always the highest-logit token
        {
            smallm::GreedySampler s;
            smallm::Generator g(qwen, tokenizer, s);
            std::cout << "[greedy]    " << prompt << g.generate(prompt, 20) << "\n";
        }

        // temperature: sample from the softened distribution
        {
            smallm::TemperatureSampler s(0.8f, 42);   // temp=0.8, seed=42
            smallm::Generator g(qwen, tokenizer, s);
            std::cout << "[temp=0.8]  " << prompt << g.generate(prompt, 20) << "\n";
        }

        // top-k: sample among the k most likely tokens
        {
            smallm::TopKSampler s(40, 0.8f, 42);      // k=40, temp=0.8, seed=42
            smallm::Generator g(qwen, tokenizer, s);
            std::cout << "[top-k=40]  " << prompt << g.generate(prompt, 20) << "\n";
        }

        // top-p (nucleus): sample from the smallest set covering probability p
        {
            smallm::TopPSampler s(0.9f, 0.8f, 42);    // p=0.9, temp=0.8, seed=42
            smallm::Generator g(qwen, tokenizer, s);
            std::cout << "[top-p=0.9] " << prompt << g.generate(prompt, 20) << "\n";
        }

    } catch (const std::exception& e) {
        std::cerr << "error: " << e.what() << "\n";
        return 1;
    }
    return 0;
}