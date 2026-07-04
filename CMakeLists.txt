cmake_minimum_required(VERSION 3.16)
project(smallm_engine LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# keep the build clean and strict while the codebase is small
add_compile_options(-Wall -Wextra -O2)

# single source for now; more get added as the engine grows
add_executable(smallm
        src/main.cpp
        src/gguf.cpp
        src/tensor.cpp
        src/ops.cpp
)

# public headers will live under include/
target_include_directories(smallm PRIVATE include)