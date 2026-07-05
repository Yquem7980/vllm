# Qwen3 vLLM CUDA Kernel Optimization

## Project goal

This project studies and optimizes CUDA kernels used by Qwen3-0.6B in vLLM v0.17.1 on NVIDIA RTX 4080 SUPER, SM 8.9.

The primary learning goal is CUDA kernel design and performance analysis. Engineering support code may be generated, but core CUDA design decisions must remain understandable to the project owner.

## Repository safety

* Never add, copy, modify, or commit model weights.
* Never commit files matching:

  * `*.safetensors`
  * `*.bin`
  * `*.pt`
  * `*.pth`
  * `*.ckpt`
  * `*.gguf`
* Never commit compiled binaries or build dependencies:

  * `*.so`
  * `*.o`
  * `build/`
  * `.deps/`
* Never commit secrets, tokens, SSH keys, `.env` files, or private absolute paths.
* Do not push, merge, rebase, reset, or delete branches unless explicitly requested.
* Do not modify the Qwen3 model weights or Hugging Face cache.

## CUDA kernel boundaries

* Do not write or replace the core optimized CUDA kernel unless explicitly requested.
* Prefer reviewing the project owner's CUDA implementation over generating a complete replacement.
* Before proposing a kernel change, explain:

  * input and output shapes;
  * tensor layout and strides;
  * thread-to-data mapping;
  * global-memory reads and writes;
  * reduction strategy;
  * synchronization requirements;
  * numerical precision;
  * expected performance effect.
* Do not claim a performance improvement without benchmark data.
* Preserve the native vLLM implementation as a fallback and baseline.
* Do not silently change in-place semantics or tensor layouts.

## Baseline responsibilities

Codex may create and maintain:

* PyTorch correctness references;
* correctness tests;
* operator benchmark scripts;
* CUDA Event timing utilities;
* shape and dtype test matrices;
* CSV or JSON result export;
* build and binding support;
* native-versus-custom A/B dispatch;
* documentation and reproducibility commands.

Correctness tests and performance benchmarks must remain separate.

## Initial operator scope

The first operator is fused Residual Add + RMSNorm.

Expected semantics:

1. `residual = input + residual`
2. `input = RMSNorm(residual, weight, epsilon)`

Both tensors are updated in place when matching the native vLLM operator interface.

Initial supported target:

* GPU: NVIDIA RTX 4080 SUPER
* Compute capability: SM 8.9
* Model: Qwen3-0.6B
* Hidden size: 1024
* Primary dtype: BF16
* Accumulation dtype: FP32

Unsupported shapes or dtypes must fall back to the native vLLM operator.

## Change discipline

* Make one focused change at a time.
* Before editing more than three files, present a concise plan.
* Do not modify existing CUDA source during the initial baseline task.
* Run relevant tests after changes.
* Report exactly which files were changed and which commands were run.
