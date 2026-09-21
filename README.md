

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

CPU inference for Qwen2 and Llama GGUF models, with a small English CLI and
localized error messages (Turkish or English via first-run setup, `SMALLM_LANG`,
or `--lang`).

## Build

```bash
cmake -B build -DCMAKE_CXX_COMPILER=clang++
cmake --build build
```

CUDA backend (optional, requires NVIDIA toolkit):

```bash
cmake -B build -DSMALLM_CUDA=ON
cmake --build build
```

## CLI

```bash
./build/smallm help
./build/smallm inspect models/your-model.gguf
./build/smallm generate models/your-model.gguf --prompt "Hello" --max-new 32
./build/smallm bench models/your-model.gguf --prompt "Hello" --max-new 50
./build/smallm generate models/your-model.gguf --prompt "Hello" --backend cuda
```

See [CONTRIBUTING.md](CONTRIBUTING.md) and [docs/ROADMAP.md](docs/ROADMAP.md).

## Benchmarks

Compare CPU and CUDA decode on the same machine (requires `-DSMALLM_CUDA=ON`):

```bash
smallm bench models/your-model.gguf --max-new 50 --backend cpu
smallm bench models/your-model.gguf --max-new 50 --backend cuda
```

CUDA currently accelerates Q4_0 quantized matmul with weights resident on the
GPU; other ops still run on the CPU until more kernels land.

Language for errors and labels: `SMALLM_LANG=tr`, `--lang tr`, or interactive
setup on first run (saved under `~/.config/smallm/language`).

## Progress

- Project skeleton
- GGUF Loader (header, metadata, tensor table)
- Tensor access and dequantization (F32, Q8_0, Q4_0)
- Core operators complete: matmul, rmsnorm, softmax, silu, rope
- Backend abstraction: Backend interface + CPUBackend (GPU-ready)
- Qwen2 forward pass: end-to-end logits on CPU
- BPE tokenizer: byte-level encode + decode
- Sampler abstraction: Sampler interface + greedy
- Text generation: end-to-end coherent output (Qwen2 complete)
- Sampling strategies: temperature, top-k, top-p (+ greedy)
- Prefix caching: architecture-agnostic KV reuse across shared prefixes
- Performance: quantized matmul (AVX2 + OpenMP), ~9x faster decode
- Llama architecture + SentencePiece tokenizer + Q6_K quantization
- CLI: generate, inspect, bench; model factory (Qwen2 / Llama auto-detect)
- i18n: English CLI, Turkish or English errors (first-run / SMALLM_LANG / --lang)
## Documentation

See [docs/ARCHITECTURE.md](docs/ARCHITECTURE.md) for module responsibilities,
the layered design, and how to extend the engine.