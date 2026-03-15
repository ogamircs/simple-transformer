"""
Simple Transformer - Phase 1: Core Building Blocks

This file builds a transformer from scratch, one component at a time.
Read it top-to-bottom. Each class is a self-contained building block.

Architecture follows karpathy/nanochat:
  - Pre-norm (normalize before each sublayer)
  - RMSNorm (simpler than LayerNorm)
  - ReLU² activation (instead of GELU)
  - No bias in linear layers
"""

import math
from dataclasses import dataclass

import torch
import torch.nn as nn
import torch.nn.functional as F


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

@dataclass
class GPTConfig:
    vocab_size: int = 256       # character-level: 256 possible byte values
    n_embd: int = 64            # embedding dimension (small for CPU training)
    n_head: int = 4             # number of attention heads
    n_layer: int = 4            # number of transformer blocks
    seq_len: int = 256          # maximum sequence length
    dropout: float = 0.0        # dropout rate (0 = no dropout, good for small models)


# ---------------------------------------------------------------------------
# RMSNorm
# ---------------------------------------------------------------------------
# LayerNorm normalizes by subtracting mean and dividing by std.
# RMSNorm is simpler: it only divides by the root-mean-square (no mean subtraction).
# This works just as well in practice and is slightly faster.
#
# Formula: RMSNorm(x) = x / RMS(x) * gamma
#   where RMS(x) = sqrt(mean(x^2) + eps)
#   and gamma is a learnable per-feature scale

class RMSNorm(nn.Module):
    def __init__(self, dim: int, eps: float = 1e-6):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim))  # learnable scale (gamma)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x shape: (batch, seq_len, dim)
        # Compute RMS across the last dimension (the embedding dimension)
        rms = torch.sqrt(torch.mean(x ** 2, dim=-1, keepdim=True) + self.eps)
        # Normalize and apply learnable scale
        return (x / rms) * self.weight


# ---------------------------------------------------------------------------
# Causal Self-Attention
# ---------------------------------------------------------------------------
# This is the core of the transformer. Here's the intuition:
#
# Each token asks: "What should I pay attention to?"
# - Q (query): "What am I looking for?"
# - K (key):   "What do I contain?"
# - V (value): "What information do I provide?"
#
# Attention = softmax(Q @ K^T / sqrt(d_k)) @ V
#
# The scaling by sqrt(d_k) prevents the dot products from growing too large.
# Without it, softmax would produce near-one-hot distributions (vanishing gradients).
#
# The causal mask ensures each position can only attend to previous positions
# (and itself). This is what makes it autoregressive — the model can't cheat
# by looking at future tokens.
#
# Multi-head: we split Q, K, V into multiple "heads" that attend independently,
# then concatenate the results. This lets the model attend to different aspects
# of the input simultaneously (e.g., one head for syntax, another for semantics).

class CausalSelfAttention(nn.Module):
    def __init__(self, config: GPTConfig):
        super().__init__()
        assert config.n_embd % config.n_head == 0, "n_embd must be divisible by n_head"

        self.n_head = config.n_head
        self.head_dim = config.n_embd // config.n_head  # dimension per head

        # Separate projections for Q, K, V (no bias, following nanochat)
        self.W_q = nn.Linear(config.n_embd, config.n_embd, bias=False)
        self.W_k = nn.Linear(config.n_embd, config.n_embd, bias=False)
        self.W_v = nn.Linear(config.n_embd, config.n_embd, bias=False)

        # Output projection: combines multi-head outputs back to n_embd
        self.W_o = nn.Linear(config.n_embd, config.n_embd, bias=False)

        self.dropout = nn.Dropout(config.dropout)

        # Precompute the causal mask
        # This is a lower-triangular matrix of ones: position i can attend to positions 0..i
        # We register it as a buffer (not a parameter) so it moves to the right device
        # but isn't updated by the optimizer
        causal_mask = torch.tril(torch.ones(config.seq_len, config.seq_len))
        self.register_buffer("causal_mask", causal_mask.view(1, 1, config.seq_len, config.seq_len))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, T, C = x.shape  # batch, sequence length, embedding dim

        # Step 1: Project input into Q, K, V
        q = self.W_q(x)  # (B, T, C)
        k = self.W_k(x)  # (B, T, C)
        v = self.W_v(x)  # (B, T, C)

        # Step 2: Reshape into multiple heads
        # (B, T, C) -> (B, T, n_head, head_dim) -> (B, n_head, T, head_dim)
        q = q.view(B, T, self.n_head, self.head_dim).transpose(1, 2)
        k = k.view(B, T, self.n_head, self.head_dim).transpose(1, 2)
        v = v.view(B, T, self.n_head, self.head_dim).transpose(1, 2)

        # Step 3: Compute attention scores
        # Q @ K^T gives us a (T, T) matrix of "how much should position i attend to j?"
        # We scale by sqrt(head_dim) to keep the variance stable
        attn = (q @ k.transpose(-2, -1)) / math.sqrt(self.head_dim)
        # Shape: (B, n_head, T, T)

        # Step 4: Apply causal mask
        # Set future positions to -inf so softmax gives them 0 weight
        attn = attn.masked_fill(self.causal_mask[:, :, :T, :T] == 0, float("-inf"))

        # Step 5: Softmax to get attention weights (they sum to 1 for each query position)
        attn = F.softmax(attn, dim=-1)
        attn = self.dropout(attn)

        # Step 6: Weighted sum of values
        # Each position's output is a weighted combination of all (visible) value vectors
        out = attn @ v  # (B, n_head, T, head_dim)

        # Step 7: Concatenate heads and project back to n_embd
        # (B, n_head, T, head_dim) -> (B, T, n_head, head_dim) -> (B, T, C)
        out = out.transpose(1, 2).contiguous().view(B, T, C)
        out = self.W_o(out)

        return out


# ---------------------------------------------------------------------------
# MLP (Feed-Forward Network)
# ---------------------------------------------------------------------------
# After attention figures out "what to look at", the MLP does the "thinking".
# It's a simple two-layer network with a nonlinearity in between.
#
# The expansion factor of 4x is standard — the hidden layer is 4 times wider
# than the embedding dimension. This gives the network more capacity to
# transform the representations.
#
# ReLU² (ReLU squared): nanochat uses relu(x)^2 instead of GELU.
# - ReLU: max(0, x) — simple but creates dead neurons
# - ReLU²: max(0, x)^2 — smoother gradient near zero, still sparse activations
# - GELU: Gaussian Error Linear Unit — common in GPT-2/3, but more complex

class MLP(nn.Module):
    def __init__(self, config: GPTConfig):
        super().__init__()
        # Up-project: n_embd -> 4 * n_embd (expand)
        self.c_fc = nn.Linear(config.n_embd, 4 * config.n_embd, bias=False)
        # Down-project: 4 * n_embd -> n_embd (compress back)
        self.c_proj = nn.Linear(4 * config.n_embd, config.n_embd, bias=False)
        self.dropout = nn.Dropout(config.dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.c_fc(x)               # (B, T, 4*C)
        x = F.relu(x).square()         # ReLU² activation
        x = self.c_proj(x)             # (B, T, C)
        x = self.dropout(x)
        return x


# ---------------------------------------------------------------------------
# Tests — run with: python model.py
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    config = GPTConfig()
    print(f"Config: {config}")
    print(f"Head dimension: {config.n_embd // config.n_head}")
    print()

    # Test RMSNorm
    norm = RMSNorm(config.n_embd)
    x = torch.randn(2, 16, config.n_embd)
    out = norm(x)
    assert out.shape == x.shape, f"RMSNorm shape mismatch: {out.shape} != {x.shape}"
    # After normalization, RMS should be approximately 1.0
    rms = torch.sqrt(torch.mean(out ** 2, dim=-1))
    print(f"RMSNorm: input shape {x.shape} -> output shape {out.shape}")
    print(f"  RMS after norm (should be ~1.0): {rms.mean().item():.4f}")
    print(f"  Parameters: {sum(p.numel() for p in norm.parameters()):,}")
    print()

    # Test CausalSelfAttention
    attn = CausalSelfAttention(config)
    x = torch.randn(2, 16, config.n_embd)
    out = attn(x)
    assert out.shape == x.shape, f"Attention shape mismatch: {out.shape} != {x.shape}"
    print(f"CausalSelfAttention: input shape {x.shape} -> output shape {out.shape}")
    print(f"  Parameters: {sum(p.numel() for p in attn.parameters()):,}")

    # Verify causal mask: manually check that attention weights are lower-triangular
    with torch.no_grad():
        q = attn.W_q(x).view(2, 16, config.n_head, config.n_embd // config.n_head).transpose(1, 2)
        k = attn.W_k(x).view(2, 16, config.n_head, config.n_embd // config.n_head).transpose(1, 2)
        scores = (q @ k.transpose(-2, -1)) / math.sqrt(config.n_embd // config.n_head)
        scores = scores.masked_fill(attn.causal_mask[:, :, :16, :16] == 0, float("-inf"))
        weights = F.softmax(scores, dim=-1)
        # Check: position 0 should only attend to itself (weight[0,0] = 1.0)
        first_pos_weights = weights[0, 0, 0, :]  # first batch, first head, first position
        assert first_pos_weights[0].item() > 0.99, "Position 0 should attend only to itself"
        # Check: no future attention (upper triangle should be 0)
        upper_sum = weights[0, 0].triu(diagonal=1).sum().item()
        assert upper_sum < 1e-6, f"Future attention leak detected: {upper_sum}"
        print(f"  Causal mask verified: no future attention leakage")
    print()

    # Test MLP
    mlp = MLP(config)
    x = torch.randn(2, 16, config.n_embd)
    out = mlp(x)
    assert out.shape == x.shape, f"MLP shape mismatch: {out.shape} != {x.shape}"
    print(f"MLP: input shape {x.shape} -> output shape {out.shape}")
    print(f"  Parameters: {sum(p.numel() for p in mlp.parameters()):,}")
    print()

    # Summary
    total = sum(
        sum(p.numel() for p in module.parameters())
        for module in [norm, attn, mlp]
    )
    print(f"Total parameters (building blocks only): {total:,}")
    print()
    print("All Phase 1 tests passed!")
