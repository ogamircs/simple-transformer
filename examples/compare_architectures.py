"""
Compare all architectures side by side.

Shows parameter counts, forward pass speed, and training behavior
on a small synthetic dataset. Run with:

    uv run python examples/compare_architectures.py
"""

import sys
import os
import time

# Allow running from examples/ or project root
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import torch

from architectures.gpt2 import GPT2, GPT2Config
from architectures.llama import Llama, LlamaConfig
from architectures.qwen import Qwen, QwenConfig
from architectures.moe import MoEModel, MoEConfig
from architectures.sliding_window import SlidingWindowModel, SlidingWindowConfig


def count_params(model):
    return sum(p.numel() for p in model.parameters())


def benchmark_forward(model, idx, n_iters=50):
    """Time forward pass (no grad)."""
    model.eval()
    # Warmup
    with torch.no_grad():
        for _ in range(5):
            model(idx)
    # Timed
    t0 = time.perf_counter()
    with torch.no_grad():
        for _ in range(n_iters):
            model(idx)
    elapsed = (time.perf_counter() - t0) / n_iters
    model.train()
    return elapsed


def train_steps(model, n_steps=200, batch_size=4, seq_len=64, vocab_size=256, lr=1e-3):
    """Train for a few steps and return loss curve."""
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr)
    losses = []

    for step in range(n_steps):
        idx = torch.randint(0, vocab_size, (batch_size, seq_len))
        targets = torch.randint(0, vocab_size, (batch_size, seq_len))
        _, loss = model(idx, targets)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        if step % 50 == 0:
            losses.append((step, loss.item()))

    return losses


def build_models(n_embd, n_layer, seq_len, vocab_size):
    """Build all models with the shared comparison hyperparameters."""
    return {
        "GPT-2": GPT2(GPT2Config(
            vocab_size=vocab_size, n_embd=n_embd, n_head=4,
            n_layer=n_layer, seq_len=seq_len,
        )),
        "Llama": Llama(LlamaConfig(
            vocab_size=vocab_size, n_embd=n_embd, n_head=4, n_kv_head=2,
            n_layer=n_layer, seq_len=seq_len,
        )),
        "Qwen": Qwen(QwenConfig(
            vocab_size=vocab_size, n_embd=n_embd, n_head=4, n_kv_head=2,
            n_layer=n_layer, seq_len=seq_len,
        )),
        "MoE": MoEModel(MoEConfig(
            vocab_size=vocab_size, n_embd=n_embd, n_head=4, n_kv_head=2,
            n_layer=n_layer, n_dense_layers=1, seq_len=seq_len,
            n_experts=8, n_experts_active=2, has_shared_expert=True,
        )),
        "SlidWin": SlidingWindowModel(SlidingWindowConfig(
            vocab_size=vocab_size, n_embd=n_embd, n_head=4, n_kv_head=2,
            n_layer=n_layer, seq_len=seq_len,
            window_size=64, global_every=3, use_qk_norm=True,
        )),
    }


def main():
    print("=" * 70)
    print("Architecture Comparison")
    print("=" * 70)

    # Same embedding size for fair comparison
    n_embd = 64
    n_layer = 4
    seq_len = 128
    vocab_size = 256

    models = build_models(n_embd, n_layer, seq_len, vocab_size)

    # --- Parameter counts ---
    print("\n1. Parameter Counts")
    print("-" * 50)
    for name, model in models.items():
        params = count_params(model)
        print(f"  {name:8s}: {params:>10,} params")

    # --- Forward pass speed ---
    print("\n2. Forward Pass Speed (batch=4, seq=64)")
    print("-" * 50)
    idx = torch.randint(0, vocab_size, (4, 64))
    for name, model in models.items():
        elapsed = benchmark_forward(model, idx)
        print(f"  {name:8s}: {elapsed*1000:>8.2f} ms/forward")

    # --- Training behavior ---
    print("\n3. Training on Random Data (200 steps)")
    print("-" * 50)
    for name, model in models.items():
        losses = train_steps(model, n_steps=200, vocab_size=vocab_size)
        loss_str = " -> ".join(f"{loss:.3f}" for _, loss in losses)
        print(f"  {name:8s}: {loss_str}")

    # --- Architecture summary ---
    print("\n4. Architecture Differences")
    print("-" * 50)
    print("  Feature           GPT-2      Llama      Qwen       MoE        SlidWin")
    print("  ───────           ─────      ─────      ────       ───        ───────")
    print("  Position          Learned    RoPE       RoPE+NTK   RoPE       RoPE")
    print("  Attention         MHA        GQA        GQA+LogN   GQA        GQA+QKNorm")
    print("  MLP               ReLU²      SwiGLU     SwiGLU     SwiGLU     SwiGLU")
    print("  Bias              None       None       QKV only   None       None")
    print("  Embeddings        Untied     Untied     Tied       Untied     Untied")
    print("  Experts           1          1          1          8 (2 act)  1")
    print("  Attn window       Full       Full       Full       Full       Local+Global")

    print("\n" + "=" * 70)
    print("All comparisons complete!")


if __name__ == "__main__":
    main()
