#include "smallm/i18n/locale.h"
#include "smallm/i18n/messages.h"

#include <cstdlib>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <string>

namespace smallm::i18n {

namespace {

    Language g_language = Language::English;

    std::filesystem::path language_config_path() {
        if (const char* xdg = std::getenv("XDG_CONFIG_HOME")) {
            return std::filesystem::path(xdg) / "smallm" / "language";
        }
        if (const char* home = std::getenv("HOME")) {
            return std::filesystem::path(home) / ".config" / "smallm" / "language";
        }
        return std::filesystem::path(".smallm_language");
    }

    void prompt_and_save() {
        std::cerr << message(MessageId::SetupChooseLanguage);
        std::string line;
        if (!std::getline(std::cin, line)) {
            g_language = Language::English;
            return;
        }
        try {
            g_language = parse_language_tag(line);
        } catch (...) {
            g_language = Language::English;
        }
        save_language_preference(g_language);
    }

} // namespace

Language current_language() { return g_language; }

void set_language(Language lang) { g_language = lang; }

Language parse_language_tag(const std::string& tag) {
    if (tag == "en" || tag == "EN" || tag == "english") {
        return Language::English;
    }
    if (tag == "tr" || tag == "TR" || tag == "turkish" || tag == "turkce") {
        return Language::Turkish;
    }
    throw_error(MessageId::ErrInvalidLanguage, tag);
}

void save_language_preference(Language lang) {
    const auto path = language_config_path();
    std::error_code ec;
    std::filesystem::create_directories(path.parent_path(), ec);
    std::ofstream out(path);
    if (!out) {
        return;
    }
    out << (lang == Language::Turkish ? "tr" : "en") << '\n';
}

bool load_language_preference(Language& out) {
    std::ifstream in(language_config_path());
    if (!in) {
        return false;
    }
    std::string tag;
    if (!std::getline(in, tag)) {
        return false;
    }
    try {
        out = parse_language_tag(tag);
    } catch (...) {
        return false;
    }
    return true;
}

void apply_cli_language_flags(int& argc, char** argv, bool& save_language,
                              bool& language_from_cli) {
    save_language = false;
    language_from_cli = false;
    int write = 1;
    for (int read = 1; read < argc; ++read) {
        std::string arg = argv[read];
        if (arg == "--save-lang") {
            save_language = true;
            continue;
        }
        if (arg.rfind("--lang=", 0) == 0) {
            set_language(parse_language_tag(arg.substr(7)));
            language_from_cli = true;
            continue;
        }
        if (arg == "--lang" && read + 1 < argc) {
            set_language(parse_language_tag(argv[++read]));
            language_from_cli = true;
            continue;
        }
        argv[write++] = argv[read];
    }
    argc = write;
    argv[argc] = nullptr;
}

void init_locale() {
    if (const char* env = std::getenv("SMALLM_LANG")) {
        try {
            set_language(parse_language_tag(env));
            return;
        } catch (...) {
            // fall through to file or prompt
        }
    }

    Language from_file;
    if (load_language_preference(from_file)) {
        set_language(from_file);
        return;
    }

    if (!std::cin.good()) {
        set_language(Language::English);
        return;
    }

    prompt_and_save();
}

void print_help() {
    std::cout << message(MessageId::HelpTitle) << "\n\n";
    std::cout << message(MessageId::HelpUsage) << "\n";
    std::cout << "  smallm [options] <command> <model.gguf> ...\n\n";
    std::cout << message(MessageId::HelpGenerate) << "\n";
    std::cout << message(MessageId::HelpInspect) << "\n";
    std::cout << message(MessageId::HelpBench) << "\n";
    std::cout << "  help                             Show this message\n\n";
    std::cout << message(MessageId::HelpLangFlag) << "\n\n";
    std::cout << "generate / bench options:\n";
    std::cout << "  -p, --prompt <text>    Input prompt\n";
    std::cout << "  -n, --max-new <N>      New tokens to generate (default: 128)\n";
    std::cout << "      --temp <T>         Sampling temperature (0 = greedy)\n";
    std::cout << "      --top-k <K>        Top-k sampling\n";
    std::cout << "      --top-p <P>        Nucleus top-p (0-1)\n";
    std::cout << "      --seed <S>         RNG seed\n";
    std::cout << "      --greedy           Force greedy decoding\n";
}

} // namespace smallm::i18n
