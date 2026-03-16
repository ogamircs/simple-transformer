"""
Simple Transformer — a GPT built from scratch for learning.

Read top-to-bottom. Each class is a self-contained building block.

Architecture follows karpathy/nanochat:
  - Pre-norm (normalize before each sublayer)
  - RMSNorm (simpler than LayerNorm)
  - ReLU² activation (instead of GELU)
  - No bias in linear layers
  - Untied embedding / lm_head weights
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
# Transformer Block
# ---------------------------------------------------------------------------
# A single transformer block applies attention then MLP, each with a
# residual connection and pre-normalization.
#
# Pre-norm means we normalize BEFORE each sublayer (not after).
# This is more stable for training than post-norm (original "Attention Is
# All You Need" used post-norm, but modern models all use pre-norm).
#
# The residual connection (x = x + sublayer(norm(x))) gives gradients a
# "highway" to flow through — without it, deep networks can't train.

class Block(nn.Module):
    def __init__(self, config: GPTConfig):
        super().__init__()
        self.ln_1 = RMSNorm(config.n_embd)       # norm before attention
        self.attn = CausalSelfAttention(config)
        self.ln_2 = RMSNorm(config.n_embd)       # norm before MLP
        self.mlp = MLP(config)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Pre-norm residual pattern:
        #   x = x + attn(norm(x))   — "what should I look at?"
        #   x = x + mlp(norm(x))    — "what should I think about it?"
        x = x + self.attn(self.ln_1(x))
        x = x + self.mlp(self.ln_2(x))
        return x


# ---------------------------------------------------------------------------
# GPT Model
# ---------------------------------------------------------------------------
# The full model assembles everything:
#   1. Token embedding: integer token ID -> vector
#   2. Positional embedding: position index -> vector (so the model knows order)
#   3. Stack of transformer blocks
#   4. Final normalization
#   5. Language model head: vector -> logits over vocabulary
#
# The forward pass converts a sequence of token IDs into a probability
# distribution over the next token at each position.
#
# Weight initialization matters! We use:
#   - Normal distribution (std=0.02) for most weights
#   - Scaled initialization for output projections (std=0.02 / sqrt(2 * n_layer))
#     This prevents the residual stream from growing with depth.

class GPT(nn.Module):
    def __init__(self, config: GPTConfig):
        super().__init__()
        self.config = config

        # Token embedding: maps each token ID to a learned vector
        self.wte = nn.Embedding(config.vocab_size, config.n_embd)

        # Positional embedding: maps each position (0, 1, 2, ...) to a learned vector
        # Without this, the model has no idea about token order (attention is permutation-invariant)
        self.wpe = nn.Embedding(config.seq_len, config.n_embd)

        # Stack of transformer blocks — this is where the "thinking" happens
        self.blocks = nn.ModuleList([Block(config) for _ in range(config.n_layer)])

        # Final normalization before projecting to vocabulary
        self.ln_f = RMSNorm(config.n_embd)

        # Language model head: project from embedding space to vocabulary size
        # Untied from wte (separate weights), following nanochat
        self.lm_head = nn.Linear(config.n_embd, config.vocab_size, bias=False)

        # Initialize weights
        self.apply(self._init_weights)
        # Scale residual projections by 1/sqrt(2*n_layer) to keep variance stable
        for name, p in self.named_parameters():
            if name.endswith("W_o.weight") or name.endswith("c_proj.weight"):
                nn.init.normal_(p, mean=0.0, std=0.02 / math.sqrt(2 * config.n_layer))

    def _init_weights(self, module):
        """Initialize weights with small random values."""
        if isinstance(module, nn.Linear):
            nn.init.normal_(module.weight, mean=0.0, std=0.02)
        elif isinstance(module, nn.Embedding):
            nn.init.normal_(module.weight, mean=0.0, std=0.02)

    def forward(self, idx, targets=None):
        """
        Args:
            idx: token indices, shape (batch, seq_len)
            targets: target token indices for loss computation, shape (batch, seq_len)

        Returns:
            logits: shape (batch, seq_len, vocab_size)
            loss: cross-entropy loss (only if targets provided)
        """
        B, T = idx.shape
        assert T <= self.config.seq_len, f"Sequence length {T} exceeds max {self.config.seq_len}"

        # Step 1: Token + positional embeddings
        # Each token gets a vector = (what it is) + (where it is)
        pos = torch.arange(T, device=idx.device)  # [0, 1, 2, ..., T-1]
        x = self.wte(idx) + self.wpe(pos)          # (B, T, n_embd)

        # Step 2: Pass through all transformer blocks
        for block in self.blocks:
            x = block(x)

        # Step 3: Final norm + project to vocabulary
        x = self.ln_f(x)
        logits = self.lm_head(x)  # (B, T, vocab_size)

        # Step 4: Compute loss if targets provided
        loss = None
        if targets is not None:
            # Cross-entropy loss: how wrong are our predictions?
            # Reshape for F.cross_entropy: (B*T, vocab_size) vs (B*T,)
            loss = F.cross_entropy(logits.view(-1, logits.size(-1)), targets.view(-1))

        return logits, loss

    def count_parameters(self):
        """Print a breakdown of parameter counts."""
        total = sum(p.numel() for p in self.parameters())
        print(f"Total parameters: {total:,}")
        print(f"  Token embeddings (wte):    {self.wte.weight.numel():,}")
        print(f"  Position embeddings (wpe): {self.wpe.weight.numel():,}")
        for i, block in enumerate(self.blocks):
            block_params = sum(p.numel() for p in block.parameters())
            print(f"  Block {i}: {block_params:,}")
        print(f"  Final norm:                {sum(p.numel() for p in self.ln_f.parameters()):,}")
        print(f"  LM head:                   {self.lm_head.weight.numel():,}")
        return total


# ---------------------------------------------------------------------------
# Tests — run with: uv run python model.py
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    config = GPTConfig()
    model = GPT(config)
    print(f"Config: {config}\n")
    model.count_parameters()

    # Test forward pass with random token IDs
    idx = torch.randint(0, config.vocab_size, (2, 32))  # batch=2, seq=32
    logits, loss = model(idx)
    assert logits.shape == (2, 32, config.vocab_size), f"Logit shape wrong: {logits.shape}"
    print(f"\nForward pass: idx {idx.shape} -> logits {logits.shape}")

    # Test with targets — loss should be ~ln(vocab_size) for random init
    targets = torch.randint(0, config.vocab_size, (2, 32))
    logits, loss = model(idx, targets)
    expected_loss = math.log(config.vocab_size)
    print(f"Loss: {loss.item():.4f} (expected ~{expected_loss:.4f} for random init)")
    assert abs(loss.item() - expected_loss) < 1.0, "Loss too far from expected"

    # Test gradient flow
    loss.backward()
    grad_norms = {name: p.grad.norm().item() for name, p in model.named_parameters() if p.grad is not None}
    assert len(grad_norms) > 0, "No gradients computed"
    print(f"Gradient flow: all {len(grad_norms)} parameters have gradients")

    print("\nAll tests passed!")
