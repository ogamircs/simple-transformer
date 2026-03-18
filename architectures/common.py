"""
Shared building blocks used across architectures.

This module contains components that appear in multiple architectures,
factored out to avoid duplication and to highlight what's shared vs unique.

Components:
  - RMSNorm: root-mean-square normalization (simpler than LayerNorm)
  - RoPE: rotary positional embeddings (encode position via rotation)
  - SwiGLU: gated feed-forward with SiLU activation
  - weight_init: standard weight initialization
"""

import math

import torch
import torch.nn as nn
import torch.nn.functional as F


# ---------------------------------------------------------------------------
# RMSNorm
# ---------------------------------------------------------------------------
# LayerNorm: normalize by subtracting mean, dividing by std, then scale+shift.
# RMSNorm: just divide by root-mean-square, then scale. No mean subtraction.
#
# Why simpler is fine: the mean subtraction in LayerNorm is mostly redundant
# when the model has bias-free linear layers (which modern models use).
# RMSNorm is ~10-15% faster than LayerNorm with no quality loss.

class RMSNorm(nn.Module):
    def __init__(self, dim: int, eps: float = 1e-6):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        rms = torch.sqrt(torch.mean(x ** 2, dim=-1, keepdim=True) + self.eps)
        return (x / rms) * self.weight


# ---------------------------------------------------------------------------
# Rotary Positional Embeddings (RoPE)
# ---------------------------------------------------------------------------
# Instead of adding a learned position vector, RoPE encodes position by
# rotating the query and key vectors. Adjacent dimensions are paired and
# rotated by an angle that depends on position and frequency.
#
# Intuition: think of each pair of dimensions as a 2D point. RoPE rotates
# that point by an angle = position * frequency. Different dimension pairs
# use different frequencies (like a clock with hands at different speeds).
#
# Key advantage over learned positional embeddings:
#   - Naturally generalizes to longer sequences (no max position limit)
#   - The dot product between rotated Q and K depends only on their
#     relative position, not absolute position
#
# Math:
#   For dimension pair (d, d+1) at position p:
#     theta_d = 1 / (base^(2d/dim))    ← frequency for this dimension pair
#     angle = p * theta_d              ← rotation angle
#     [q_d, q_{d+1}] is rotated by this angle
#
# The base (default 10000) controls the wavelength range:
#   - Low-index dimensions: high frequency (captures local patterns)
#   - High-index dimensions: low frequency (captures global patterns)

def precompute_rope_frequencies(dim: int, max_seq_len: int, base: float = 10000.0) -> torch.Tensor:
    """
    Precompute the complex exponentials for RoPE.

    Returns:
        freqs_cis: shape (max_seq_len, dim//2) — complex-valued rotation factors
    """
    # Frequencies for each dimension pair: theta_d = 1 / (base^(2d/dim))
    freqs = 1.0 / (base ** (torch.arange(0, dim, 2).float() / dim))
    # Outer product with positions: angle = position * frequency
    t = torch.arange(max_seq_len)
    angles = torch.outer(t, freqs)  # (max_seq_len, dim//2)
    # Convert to complex exponentials: e^(i*angle) = cos(angle) + i*sin(angle)
    freqs_cis = torch.polar(torch.ones_like(angles), angles)
    return freqs_cis


def apply_rope(x: torch.Tensor, freqs_cis: torch.Tensor) -> torch.Tensor:
    """
    Apply rotary embeddings to a tensor.

    Args:
        x: shape (batch, n_heads, seq_len, head_dim)
        freqs_cis: shape (seq_len, head_dim//2) — from precompute_rope_frequencies

    Returns:
        Rotated tensor, same shape as x
    """
    # Reshape x into pairs of dimensions and view as complex numbers
    # (B, H, T, D) -> (B, H, T, D//2, 2) -> complex (B, H, T, D//2)
    x_complex = torch.view_as_complex(x.float().reshape(*x.shape[:-1], -1, 2))
    # Multiply by rotation factors (broadcasting over batch and heads)
    freqs_cis = freqs_cis.unsqueeze(0).unsqueeze(0)  # (1, 1, T, D//2)
    x_rotated = x_complex * freqs_cis
    # Convert back to real pairs and flatten
    x_out = torch.view_as_real(x_rotated).reshape(*x.shape)
    return x_out.type_as(x)


# ---------------------------------------------------------------------------
# SwiGLU Feed-Forward Network
# ---------------------------------------------------------------------------
# Standard MLP: up-project -> activation -> down-project
# SwiGLU MLP:   gate-project -> SiLU(gate) * up-project -> down-project
#
# The "gated" part: instead of one up-projection, we have two:
#   - One goes through SiLU (the "gate" — decides what to let through)
#   - One is the "value" (the actual information)
#   - They're multiplied element-wise: gate controls how much value passes
#
# SiLU(x) = x * sigmoid(x), also called "swish". It's smooth like GELU
# but has a slight negative region that helps with gradient flow.
#
# Why 8/3 * dim? The standard MLP uses 4*dim hidden size. With SwiGLU
# we have two projections instead of one, so we use 2/3 of the width
# to keep the parameter count similar: 2 * (8/3 * dim) ≈ 2 * (4 * dim) / 1.5
#
# The multiple_of rounding ensures the hidden size is GPU-friendly
# (divisible by 64 or 256 for tensor core efficiency).

class SwiGLU(nn.Module):
    def __init__(self, dim: int, hidden_dim: int | None = None, multiple_of: int = 64, dropout: float = 0.0):
        super().__init__()
        if hidden_dim is None:
            hidden_dim = int(2 * (4 * dim) / 3)
        # Round up to nearest multiple for GPU efficiency
        hidden_dim = multiple_of * ((hidden_dim + multiple_of - 1) // multiple_of)

        self.w_gate = nn.Linear(dim, hidden_dim, bias=False)  # gate projection
        self.w_up = nn.Linear(dim, hidden_dim, bias=False)    # value projection
        self.w_down = nn.Linear(hidden_dim, dim, bias=False)  # down projection
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Gate: SiLU activation decides what information to let through
        gate = F.silu(self.w_gate(x))
        # Up: the actual information
        up = self.w_up(x)
        # Element-wise multiply: gate controls information flow
        x = gate * up
        # Project back down to model dimension
        x = self.w_down(x)
        return self.dropout(x)


# ---------------------------------------------------------------------------
# Weight initialization
# ---------------------------------------------------------------------------

def init_weights(module: nn.Module, std: float = 0.02):
    """Standard weight initialization for transformer models."""
    if isinstance(module, nn.Linear):
        nn.init.normal_(module.weight, mean=0.0, std=std)
        if module.bias is not None:
            nn.init.zeros_(module.bias)
    elif isinstance(module, nn.Embedding):
        nn.init.normal_(module.weight, mean=0.0, std=std)
