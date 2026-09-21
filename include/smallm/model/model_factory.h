#pragma once

#include "smallm/core/gguf.h"
#include "smallm/model/common/model.h"

#include <memory>

namespace smallm {

    // picks Qwen2 or Llama from general.architecture in the GGUF metadata
    std::unique_ptr<Model> build_model(GGUFModel gguf);

} // namespace smallm
