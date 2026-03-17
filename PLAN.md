# Architecture Gallery — Roadmap

Phased plan for implementing educational LLM architectures from
[Sebastian Raschka's LLM Architecture Gallery](https://sebastianraschka.com/llm-architecture-gallery/).

## Phase 1: Core Trio (DONE)

The foundational set that covers 80% of the design space.

| Architecture | Status | Key concepts |
|---|---|---|
| GPT-2 baseline | Done | MHA, learned pos emb, ReLU², pre-norm |
| Llama-style | Done | RoPE, GQA, SwiGLU |
| MoE (DeepSeek-lite) | Done | Top-k routing, load-balancing loss, shared expert, dense prefix |

Infrastructure:
- [x] Self-contained architecture files with inline docs
- [x] Test suite (shapes, loss, gradients, architecture properties, overfitting)
- [x] Comparison example script
- [x] README with reading order

## Phase 2: Attention Variants (DONE)

| Architecture | Status | Key concepts |
|---|---|---|
| Sliding-window attention | Done | Local attention window + periodic global layers, QK-Norm |
| Qwen 2.5 | Done | Selective bias, tied embeddings, NTK-aware RoPE, LogN scaling |

Infrastructure:
- [x] Teaching notebooks for all architectures (GPT-2, Llama, Qwen, MoE, Sliding Window)
- [x] All notebooks tested and passing in Google Colab
- [x] Updated test suite covering all 5 architectures
- [x] Updated comparison script with all 5 architectures

## Phase 3: Advanced Attention

| Architecture | Concept | Gallery references |
|---|---|---|
| Multi-head Latent Attention (MLA) | Compress KV into low-rank latent space | DeepSeek V3, Kimi K2 |

## Phase 4: Hybrid Architectures

Models that mix transformer attention with alternative sequence-mixing layers.

| Architecture | Concept | Gallery references |
|---|---|---|
| Mamba-2 / SSM hybrid | State-space model layers mixed with sparse attention | Nemotron Nano |
| Gated DeltaNet hybrid | Linear attention variant mixed with softmax attention | Qwen3 Next, Qwen3.5 |
| Lightning Attention | Linear-complexity attention via cumulative sum | Ling 2.5 |

## Phase 5: Training Techniques

Orthogonal to architecture — training innovations that apply across models.

| Technique | Concept | Gallery references |
|---|---|---|
| Multi-token prediction (MTP) | Predict multiple future tokens per position | Xiaomi MiMo, Step 3.5 Flash |
| Parallel attention + MLP | Run attn and MLP in parallel, not serial | Tiny Aya |
| Post-norm (inside residual) | Alternative norm placement | OLMo 2, OLMo 3 |

## Phase 6: Full Training Comparison

Train all implemented architectures on Tiny Shakespeare with identical
hyperparameters and produce a comparison report:
- Loss curves
- Parameter efficiency
- Generation quality
- Convergence speed

## Architecture Coverage Map

From the gallery's 43 architectures, here's what our implementations cover:

| Pattern | Gallery examples | Our coverage |
|---|---|---|
| Dense decoder (learned pos) | GPT-2 XL | **gpt2.py** |
| Dense decoder (RoPE + GQA + SwiGLU) | Llama 3, Mistral | **llama.py** |
| Dense decoder (selective bias, tied emb) | Qwen 2.5 | **qwen.py** |
| Sparse MoE | DeepSeek V3, Llama 4, Qwen3 235B | **moe.py** |
| Sliding window + global | Gemma 3, OLMo 3 | **sliding_window.py** |
| MLA (latent attention) | DeepSeek V3, Kimi K2 | Phase 3 |
| SSM hybrid (Mamba) | Nemotron Nano/Super | Phase 4 |
| Linear attention hybrid | Qwen3 Next, Ling 2.5 | Phase 4 |
| Parallel attn+MLP | Tiny Aya | Phase 5 |
| Multi-token prediction | Xiaomi MiMo, Step 3.5 | Phase 5 |
