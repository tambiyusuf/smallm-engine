cmake_minimum_required(VERSION 3.16)
project(smallm_engine LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# keep the build clean and strict while the codebase is small
add_compile_options(-Wall -Wextra -O2)

# single source for now; more get added as the engine grows
add_executable(smallm
        src/main.cpp

        # core
        src/core/gguf.cpp
        src/core/tensor.cpp
        src/core/ops.cpp

        # config
        src/config/common/config.cpp
        src/config/qwen2_config.cpp

        # model
        src/model/qwen2_model.cpp

        # tokenizer
        src/tokenizer/bpe_tokenizer.cpp

        # runtime
        src/runtime/sampler/greedy_sampler.cpp
        src/runtime/sampler/temperature_sampler.cpp
        src/runtime/sampler/top_k_sampler.cpp
        src/runtime/sampler/top_p_sampler.cpp
        src/runtime/generator.cpp

        # backend
        src/backend/cpu_backend.cpp
)

# public headers will live under include/
target_include_directories(smallm PRIVATE include)