# smallm-engine

A from-scratch inference engine for GGUF language models, written in modern C++.

This is not a wrapper around llama.cpp. Every layer — file loading, tensor
handling, operators, tokenizer and the execution loop — is built here directly,
to keep full control over how a model runs on the hardware.

## Goal

Load a GGUF model, run a forward pass on CPU, and generate tokens.
GPU (CUDA) support comes later, once the CPU path is correct.

## Status

Early stage. Current milestone: reading and inspecting GGUF files.

## Build

```bash
cmake -B build -DCMAKE_CXX_COMPILER=clang++
cmake --build build
```

## Progress

- Project skeleton