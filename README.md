# Simple Transformer

Educational, from-scratch PyTorch implementations of major LLM architectures. Each file is a self-contained, readable implementation — meant to be read top-to-bottom as a learning resource.

Inspired by [Sebastian Raschka's LLM Architecture Gallery](https://sebastianraschka.com/llm-architecture-gallery/) and [karpathy/nanochat](https://github.com/karpathy/nanochat).

## Architectures Implemented

| Architecture | Key Innovation | File | Gallery Reference |
|---|---|---|---|
| **GPT-2** | Learned position embeddings, MHA, ReLU² | `architectures/gpt2.py` | GPT-2 XL 1.5B |
| **Llama** | RoPE + GQA + SwiGLU | `architectures/llama.py` | Llama 3 8B |
| **Qwen 2.5** | Selective bias, tied embeddings, NTK-aware RoPE, LogN scaling | `architectures/qwen.py` | Qwen 2.5 7B |
| **MoE** | Top-k expert routing + shared expert | `architectures/moe.py` | DeepSeek V3 |
| **Sliding Window** | Local/global attention + QK-Norm | `architectures/sliding_window.py` | Gemma 3 27B |

Reading order: GPT-2 → Llama → Qwen → MoE → Sliding Window. Each successive architecture adds one or two ideas on top of the previous one.

### What each architecture teaches

**GPT-2** (the baseline):
- Causal self-attention and multi-head attention
- Learned absolute positional embeddings
- Pre-norm with RMSNorm, residual connections
- ReLU² activation in MLP

**Llama** (the modern standard — 4 upgrades over GPT-2):
1. **RoPE** — rotary positional embeddings (relative position via rotation)
2. **GQA** — grouped-query attention (fewer KV heads → smaller KV cache)
3. **SwiGLU** — gated MLP with SiLU activation
4. No learned position embedding table

**Qwen 2.5** (refined Llama — 4 innovations):
1. **Selective bias** — QKV projections have bias, everything else doesn't
2. **Tied embeddings** — input and output share the same weight matrix
3. **NTK-aware RoPE** — extends context beyond training length without retraining
4. **LogN attention scaling** — prevents attention collapse in long sequences

**MoE** (sparse scaling):
- Top-k expert routing (each token selects k of N expert MLPs)
- Load-balancing auxiliary loss (prevents expert collapse)
- Shared always-on expert (DeepSeek V3 pattern)
- Dense prefix layers + MoE layers (hybrid architecture)
- Total params >> active params per token

**Sliding Window** (efficient long context):
- Local sliding-window attention (O(n·w) instead of O(n²))
- Alternating local/global attention layers
- QK-Norm for training stability
- Effective receptive field grows with depth

## Interactive Notebooks

Each architecture has a teaching-first Jupyter notebook that builds the model from scratch:

| Notebook | Description |
|---|---|
| [`notebook/simple_transformer.ipynb`](notebook/simple_transformer.ipynb) | Original standalone GPT walkthrough |
| [`notebook/gpt2.ipynb`](notebook/gpt2.ipynb) | GPT-2 architecture walkthrough |
| [`notebook/llama.ipynb`](notebook/llama.ipynb) | Llama with RoPE, GQA, SwiGLU |
| [`notebook/qwen.ipynb`](notebook/qwen.ipynb) | Qwen 2.5 with selective bias and tied embeddings |
| [`notebook/moe.ipynb`](notebook/moe.ipynb) | Mixture of Experts with routing |
| [`notebook/sliding_window.ipynb`](notebook/sliding_window.ipynb) | Sliding window with local/global attention |

All notebooks are self-contained and run in Google Colab with no setup.

## Project Layout

```text
architectures/
  gpt2.py                    GPT-2 baseline
  llama.py                   Llama-style (RoPE + GQA + SwiGLU)
  qwen.py                    Qwen 2.5 (selective bias, tied embeddings, NTK RoPE, LogN)
  moe.py                     Mixture-of-Experts
  sliding_window.py          Sliding window attention (local/global, QK-Norm)
  common.py                  Shared reference components
notebook/
  simple_transformer.ipynb   Original standalone GPT walkthrough
  gpt2.ipynb                 GPT-2 teaching notebook
  llama.ipynb                Llama teaching notebook
  qwen.ipynb                 Qwen 2.5 teaching notebook
  moe.ipynb                  MoE teaching notebook
  sliding_window.ipynb       Sliding window teaching notebook
tests/
  test_architectures.py      Tests across all architectures
examples/
  compare_architectures.py   Side-by-side comparison script

model.py                     Original standalone GPT (preserved)
data.py                      Tiny Shakespeare dataset + byte tokenizer
train.py                     Training loop with LR scheduling
generate.py                  Text generation from checkpoints
```

## Setup

```bash
uv sync               # install dependencies
uv sync --extra dev   # include pytest for testing
```

## Quick Start

Run self-tests for each architecture:

```bash
uv run python -m architectures.gpt2
uv run python -m architectures.llama
uv run python -m architectures.qwen
uv run python -m architectures.moe
uv run python -m architectures.sliding_window
```

Run the full test suite:

```bash
uv run python -m pytest tests/ -v
```

Compare all architectures side by side:

```bash
uv run python examples/compare_architectures.py
```

Train and generate with the original model:

```bash
uv run python train.py --sanity-check
uv run python train.py
uv run python generate.py --prompt "ROMEO:"
```

## Design Principles

1. **Clarity over performance** — these are learning implementations, not production kernels
2. **Self-contained files** — each architecture file can be understood without reading others
3. **Inline documentation** — comments explain the *why*, not just the *what*
4. **Small defaults** — models are tiny (64-dim, 4 layers) so they run on CPU in seconds
5. **Real architecture patterns** — simplified but faithful to the actual papers

## Support

If this project helps you, you can support it here:

[Buy Me a Coffee](https://buymeacoffee.com/ogamircs)
