

<img width="976" height="549" alt="jbgfngj5d4117fwj93z8tz93z8tz93z8" src="https://github.com/user-attachments/assets/15ab3621-2058-4251-a0b7-287417e44642" />




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
- GGUF Loader (header, metadata, tensor table)
- Tensor access and dequantization (F32, Q8_0, Q4_0)
- - Core operators complete: matmul, rmsnorm, softmax, silu, rope
