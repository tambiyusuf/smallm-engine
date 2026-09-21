#include "smallm/model/model_factory.h"

#include "smallm/config/common/meta_read.h"
#include "smallm/i18n/messages.h"
#include "smallm/model/llama_model.h"
#include "smallm/model/qwen2_model.h"

namespace smallm {

std::unique_ptr<Model> build_model(GGUFModel gguf, std::unique_ptr<Backend> backend) {
    std::string arch = meta::get<std::string>(gguf, "general.architecture");

    if (arch == "qwen2") {
        return std::make_unique<Qwen2Model>(std::move(gguf), std::move(backend));
    }
    if (arch == "llama") {
        return std::make_unique<LlamaModel>(std::move(gguf), std::move(backend));
    }

    i18n::throw_error(i18n::MessageId::ErrUnsupportedArchitecture, arch);
}

} // namespace smallm
