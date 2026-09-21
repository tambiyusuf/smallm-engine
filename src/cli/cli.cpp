#include "smallm/cli/cli.h"

#include "smallm/config/common/config.h"
#include "smallm/config/common/meta_read.h"
#include "smallm/core/gguf.h"
#include "smallm/i18n/locale.h"
#include "smallm/i18n/messages.h"
#include "smallm/model/model_factory.h"
#include "smallm/runtime/generator.h"
#include "smallm/runtime/sampler/greedy_sampler.h"
#include "smallm/runtime/sampler/temperature_sampler.h"
#include "smallm/runtime/sampler/top_k_sampler.h"
#include "smallm/runtime/sampler/top_p_sampler.h"
#include "smallm/tokenizer/common/tokenizer.h"

#include <cstring>
#include <iostream>
#include <memory>
#include <string>

namespace smallm::cli {

namespace {

    struct GenerateOptions {
        std::string prompt = "The capital of France is";
        uint32_t max_new = 128;
        float temp = 0.0f;
        uint32_t top_k = 0;
        float top_p = 0.0f;
        uint64_t seed = 0;
        bool greedy = false;
        bool seed_set = false;
    };

    bool parse_generate_flags(int& i, int argc, char** argv, GenerateOptions& opt) {
        for (; i < argc; ++i) {
            std::string arg = argv[i];
            if (arg == "-p" || arg == "--prompt") {
                if (i + 1 >= argc) {
                    return false;
                }
                opt.prompt = argv[++i];
                continue;
            }
            if (arg == "-n" || arg == "--max-new") {
                if (i + 1 >= argc) {
                    return false;
                }
                opt.max_new = static_cast<uint32_t>(std::stoul(argv[++i]));
                continue;
            }
            if (arg == "--temp") {
                if (i + 1 >= argc) {
                    return false;
                }
                opt.temp = std::stof(argv[++i]);
                continue;
            }
            if (arg == "--top-k") {
                if (i + 1 >= argc) {
                    return false;
                }
                opt.top_k = static_cast<uint32_t>(std::stoul(argv[++i]));
                continue;
            }
            if (arg == "--top-p") {
                if (i + 1 >= argc) {
                    return false;
                }
                opt.top_p = std::stof(argv[++i]);
                continue;
            }
            if (arg == "--seed") {
                if (i + 1 >= argc) {
                    return false;
                }
                opt.seed = std::stoull(argv[++i]);
                opt.seed_set = true;
                continue;
            }
            if (arg == "--greedy") {
                opt.greedy = true;
                continue;
            }
            return false;
        }
        return true;
    }

    std::unique_ptr<Sampler> make_sampler(const GenerateOptions& opt) {
        const uint64_t seed = opt.seed_set ? opt.seed : 42;

        if (opt.greedy || (opt.temp <= 0.0f && opt.top_k == 0 && opt.top_p <= 0.0f)) {
            return std::make_unique<GreedySampler>();
        }
        if (opt.top_k > 0) {
            return std::make_unique<TopKSampler>(opt.top_k, opt.temp > 0 ? opt.temp : 1.0f,
                                                 seed);
        }
        if (opt.top_p > 0.0f) {
            return std::make_unique<TopPSampler>(opt.top_p, opt.temp > 0 ? opt.temp : 1.0f,
                                                 seed);
        }
        return std::make_unique<TemperatureSampler>(opt.temp > 0 ? opt.temp : 1.0f, seed);
    }

    int run_generate(const std::string& model_path, GenerateOptions opt, bool bench_mode) {
        GGUFModel file = load_gguf(model_path);
        auto tokenizer = build_tokenizer(file);
        auto engine = build_model(std::move(file));

        auto sampler = make_sampler(opt);
        Generator gen(*engine, *tokenizer, *sampler);

        std::string out = gen.generate(opt.prompt, opt.max_new);
        if (!bench_mode) {
            std::cout << opt.prompt << out << "\n";
        }

        const auto& s = gen.last_stats();
        std::cout << i18n::message(i18n::MessageId::StatPrefill) << ": "
                  << s.prefill_tokens << " token  " << s.prefill_tokens_per_sec() << " "
                  << i18n::message(i18n::MessageId::StatTokensPerSec) << "\n";
        std::cout << i18n::message(i18n::MessageId::StatDecode) << ": "
                  << s.generated_tokens << " token  " << s.decode_tokens_per_sec() << " "
                  << i18n::message(i18n::MessageId::StatTokensPerSec) << "\n";
        if (s.reused_tokens > 0) {
            std::cout << i18n::message(i18n::MessageId::StatCacheReused) << ": "
                      << s.reused_tokens << "\n";
        }
        return 0;
    }

    int run_inspect(const std::string& model_path) {
        GGUFModel file = load_gguf(model_path);
        auto cfg = read_config(file);

        std::string tok_kind;
        try {
            tok_kind = meta::get<std::string>(file, "tokenizer.ggml.model");
        } catch (...) {
            tok_kind = "?";
        }

        std::cout << i18n::message(i18n::MessageId::LabelArchitecture) << ": "
                  << cfg->architecture << "\n";
        std::cout << i18n::message(i18n::MessageId::LabelVocab) << ": "
                  << cfg->vocab_size << "\n";
        std::cout << i18n::message(i18n::MessageId::LabelLayers) << ": "
                  << cfg->n_layers << "\n";
        std::cout << i18n::message(i18n::MessageId::LabelContext) << ": "
                  << cfg->context_length << "\n";
        std::cout << i18n::message(i18n::MessageId::LabelTokenizer) << ": "
                  << tok_kind << "\n";
        std::cout << i18n::message(i18n::MessageId::LabelTensors) << ": "
                  << file.tensors.size() << "\n";
        return 0;
    }

} // namespace

int run(int argc, char** argv) {
    bool save_lang = false;
    bool lang_from_cli = false;
    i18n::apply_cli_language_flags(argc, argv, save_lang, lang_from_cli);
    if (!lang_from_cli) {
        i18n::init_locale();
    } else if (save_lang) {
        i18n::save_language_preference(i18n::current_language());
    }

    if (argc < 2) {
        i18n::print_help();
        return argc < 2 ? 1 : 0;
    }

    std::string command = argv[1];
    if (command == "help" || command == "-h" || command == "--help") {
        i18n::print_help();
        return 0;
    }

    if (command != "generate" && command != "inspect" && command != "bench") {
        i18n::throw_error(i18n::MessageId::ErrUnknownCliCommand, command);
    }

    if (argc < 3) {
        i18n::throw_error(i18n::MessageId::ErrMissingModelPath);
    }

    const std::string model_path = argv[2];

    if (command == "inspect") {
        return run_inspect(model_path);
    }

    GenerateOptions opt;
    int flag_index = 3;
    if (!parse_generate_flags(flag_index, argc, argv, opt)) {
        i18n::print_help();
        return 1;
    }

    const bool bench_mode = (command == "bench");
    return run_generate(model_path, opt, bench_mode);
}

} // namespace smallm::cli
