#pragma once

#include "smallm/backend/common/backend.h"
#include "smallm/core/gguf.h"
#include "smallm/model/common/model.h"

#include <memory>

namespace smallm {

    // picks Qwen2 or Llama from general.architecture in the GGUF metadata
    std::unique_ptr<Model> build_model(GGUFModel gguf,
                                       std::unique_ptr<Backend> backend);

} // namespace smallm
