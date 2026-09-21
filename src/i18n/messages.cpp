#include "smallm/i18n/messages.h"
#include "smallm/i18n/locale.h"

#include <stdexcept>

namespace smallm::i18n {

namespace {

    struct Entry {
        const char* en;
        const char* tr;
    };

    Entry table(MessageId id) {
        switch (id) {
        case MessageId::SetupChooseLanguage:
            return {"Choose display language [en/tr]: ",
                    "Gösterim dili seçin [en/tr]: "};
        case MessageId::ErrCannotOpenFile:
            return {"cannot open file", "dosya açılamadı"};
        case MessageId::ErrBadGgufMagic:
            return {"bad magic, not a GGUF file", "geçersiz dosya, GGUF değil"};
        case MessageId::ErrUnsupportedArchitecture:
            return {"unsupported architecture", "desteklenmeyen mimari"};
        case MessageId::ErrUnsupportedTokenizer:
            return {"unsupported tokenizer type", "desteklenmeyen tokenizer türü"};
        case MessageId::ErrUnsupportedTensorType:
            return {"unsupported tensor type", "desteklenmeyen tensor türü"};
        case MessageId::ErrTensorNotFound:
            return {"tensor not found", "tensor bulunamadı"};
        case MessageId::ErrConfigMissingKey:
            return {"missing metadata key", "metadata anahtarı eksik"};
        case MessageId::ErrConfigWrongType:
            return {"wrong type for metadata key", "metadata anahtarı için yanlış tür"};
        case MessageId::ErrTokenizerMissingTokens:
            return {"missing tokenizer.ggml.tokens", "tokenizer.ggml.tokens eksik"};
        case MessageId::ErrTokenizerBadMetadata:
            return {"tokenizer metadata has wrong type", "tokenizer metadata türü hatalı"};
        case MessageId::ErrUnknownCliCommand:
            return {"unknown command", "bilinmeyen komut"};
        case MessageId::ErrMissingModelPath:
            return {"missing model path", "model dosya yolu eksik"};
        case MessageId::ErrMissingSubcommand:
            return {"missing subcommand (generate, inspect, bench, help)",
                    "alt komut eksik (generate, inspect, bench, help)"};
        case MessageId::ErrInvalidLanguage:
            return {"invalid language, use en or tr", "geçersiz dil, en veya tr kullanın"};
        case MessageId::ErrUnknownBackend:
            return {"unknown backend (use cpu or cuda)", "bilinmeyen backend (cpu veya cuda)"};
        case MessageId::ErrCudaNotBuilt:
            return {"CUDA backend not built; recompile with -DSMALLM_CUDA=ON",
                    "CUDA backend derlenmedi; -DSMALLM_CUDA=ON ile yeniden derleyin"};
        case MessageId::ErrGeneric:
            return {"error", "hata"};
        case MessageId::HelpTitle:
            return {"smallm — local GGUF inference engine",
                    "smallm — yerel GGUF çıkarım motoru"};
        case MessageId::HelpUsage:
            return {"Usage:", "Kullanım:"};
        case MessageId::HelpGenerate:
            return {"  generate <model.gguf> [options]  Run text generation",
                    "  generate <model.gguf> [seçenekler]  Metin üret"};
        case MessageId::HelpInspect:
            return {"  inspect <model.gguf>             Show model metadata",
                    "  inspect <model.gguf>             Model bilgisini göster"};
        case MessageId::HelpBench:
            return {"  bench <model.gguf> [options]     Measure prefill/decode speed",
                    "  bench <model.gguf> [seçenekler]    Hız ölç"};
        case MessageId::HelpLangFlag:
            return {"  --lang en|tr   UI language (errors, help). --save-lang writes config",
                    "  --lang en|tr   Arayüz dili. --save-lang ayarı dosyaya yazılır"};
        case MessageId::LabelArchitecture:
            return {"architecture", "mimari"};
        case MessageId::LabelVocab:
            return {"vocab_size", "sözlük"};
        case MessageId::LabelLayers:
            return {"layers", "katman"};
        case MessageId::LabelContext:
            return {"context", "bağlam"};
        case MessageId::LabelTokenizer:
            return {"tokenizer", "tokenizer"};
        case MessageId::LabelTensors:
            return {"tensors", "tensor"};
        case MessageId::StatPrefill:
            return {"prefill", "ön-doldurma"};
        case MessageId::StatDecode:
            return {"decode", "üretim"};
        case MessageId::StatCacheReused:
            return {"cache reused (tokens)", "önbellekte yeniden kullanılan (token)"};
        case MessageId::StatTokensPerSec:
            return {"tok/s", "token/sn"};
        }
        return {"?", "?"};
    }

} // namespace

const char* message(MessageId id) {
    const Entry e = table(id);
    return current_language() == Language::Turkish ? e.tr : e.en;
}

std::string format(MessageId id, const std::string& detail) {
    std::string out = message(id);
    if (!detail.empty()) {
        out += ": ";
        out += detail;
    }
    return out;
}

void throw_error(MessageId id, const std::string& detail) {
    throw std::runtime_error(format(id, detail));
}

} // namespace smallm::i18n
