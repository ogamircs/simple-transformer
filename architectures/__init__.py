# Architecture Gallery
#
# Each module implements a different LLM architecture from scratch,
# following the same educational style as the base GPT model.
#
# Inspired by Sebastian Raschka's LLM Architecture Gallery:
# https://sebastianraschka.com/llm-architecture-gallery/
#
# Reading order:
#   1. gpt2.py           — GPT-2 baseline (learned pos emb, MHA, ReLU²)
#   2. llama.py          — Llama-style (RoPE, GQA, SwiGLU)
#   3. qwen.py           — Qwen 2.5 (selective bias, tied embeddings, NTK-aware RoPE, LogN scaling)
#   4. moe.py            — Mixture-of-Experts (top-k routing, shared expert)
#   5. sliding_window.py — Sliding window attention (local/global, QK-Norm)

from architectures.gpt2 import GPT2, GPT2Config
from architectures.llama import Llama, LlamaConfig
from architectures.qwen import Qwen, QwenConfig
from architectures.moe import MoEModel, MoEConfig
from architectures.sliding_window import SlidingWindowModel, SlidingWindowConfig
