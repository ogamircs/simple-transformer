"""
GPT-2 — the classic decoder-only transformer baseline.

This is where it all started for modern LLMs. Understanding GPT-2 first
makes every subsequent architecture a clear set of diffs. Each later model
in this gallery changes one or two things from this baseline.

Key features (our version vs original GPT-2):
  - Learned absolute positional embeddings (same as original)
  - Multi-Head Attention — all heads have independent Q, K, V (same)
  - Pre-norm with RMSNorm (original used post-norm LayerNorm)
  - ReLU² activation (original used GELU)
  - No bias in linear layers (original had bias)
  - Untied embedding / lm_head weights

What changes in Llama (next file):
  - Learned pos embeddings → RoPE (rotary positional embeddings)
  - MHA → GQA (grouped query attention, fewer KV heads)
  - ReLU² MLP → SwiGLU (gated feed-forward)

Reference: Radford et al., "Language Models are Unsupervised Multitask
Learners" (2019). Gallery entry: GPT-2 XL 1.5B.
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
class GPT2Config:
    vocab_size: int = 256       # byte-level tokenization
    n_embd: int = 64            # embedding dimension (small for CPU training)
    n_head: int = 4             # number of attention heads
    n_layer: int = 4            # number of transformer blocks
    seq_len: int = 256          # maximum sequence length
    dropout: float = 0.0        # dropout rate


# ---------------------------------------------------------------------------
# RMSNorm
# ---------------------------------------------------------------------------
# Simpler than LayerNorm: just divide by root-mean-square, no mean subtraction.
# Works just as well when linear layers have no bias.

class RMSNorm(nn.Module):
    def __init__(self, dim: int, eps: float = 1e-6):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        rms = torch.sqrt(torch.mean(x ** 2, dim=-1, keepdim=True) + self.eps)
        return (x / rms) * self.weight


# ---------------------------------------------------------------------------
# Multi-Head Attention (MHA)
# ---------------------------------------------------------------------------
# Every head has its own Q, K, V projections of size head_dim.
# Total attention params: 4 * n_embd² (three projections + output)
#
# Compare with GQA in llama.py: Q has n_head projections, but K and V
# have fewer (n_kv_head). This reduces KV-cache memory at inference.

class MultiHeadAttention(nn.Module):
    def __init__(self, config: GPT2Config):
        super().__init__()
        assert config.n_embd % config.n_head == 0
        self.n_head = config.n_head
        self.head_dim = config.n_embd // config.n_head

        self.W_q = nn.Linear(config.n_embd, config.n_embd, bias=False)
        self.W_k = nn.Linear(config.n_embd, config.n_embd, bias=False)
        self.W_v = nn.Linear(config.n_embd, config.n_embd, bias=False)
        self.W_o = nn.Linear(config.n_embd, config.n_embd, bias=False)
        self.dropout = nn.Dropout(config.dropout)

        causal_mask = torch.tril(torch.ones(config.seq_len, config.seq_len))
        self.register_buffer("causal_mask", causal_mask.view(1, 1, config.seq_len, config.seq_len))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, T, C = x.shape
        q = self.W_q(x).view(B, T, self.n_head, self.head_dim).transpose(1, 2)
        k = self.W_k(x).view(B, T, self.n_head, self.head_dim).transpose(1, 2)
        v = self.W_v(x).view(B, T, self.n_head, self.head_dim).transpose(1, 2)

        attn = (q @ k.transpose(-2, -1)) / math.sqrt(self.head_dim)
        attn = attn.masked_fill(self.causal_mask[:, :, :T, :T] == 0, float("-inf"))
        attn = F.softmax(attn, dim=-1)
        attn = self.dropout(attn)

        out = attn @ v
        out = out.transpose(1, 2).contiguous().view(B, T, C)
        return self.W_o(out)


# ---------------------------------------------------------------------------
# MLP with ReLU²
# ---------------------------------------------------------------------------
# The standard feed-forward network: expand 4x, activate, compress back.
# ReLU²(x) = max(0, x)² — sparser than GELU, smoother gradient than ReLU.
#
# Compare with SwiGLU in llama.py: SwiGLU uses two up-projections
# (gate + value) with a gating mechanism instead of a single activation.

class MLP(nn.Module):
    def __init__(self, config: GPT2Config):
        super().__init__()
        self.c_fc = nn.Linear(config.n_embd, 4 * config.n_embd, bias=False)
        self.c_proj = nn.Linear(4 * config.n_embd, config.n_embd, bias=False)
        self.dropout = nn.Dropout(config.dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.c_fc(x)
        x = F.relu(x).square()     # ReLU² activation
        x = self.c_proj(x)
        return self.dropout(x)


# ---------------------------------------------------------------------------
# Transformer Block
# ---------------------------------------------------------------------------

class Block(nn.Module):
    def __init__(self, config: GPT2Config):
        super().__init__()
        self.ln_1 = RMSNorm(config.n_embd)
        self.attn = MultiHeadAttention(config)
        self.ln_2 = RMSNorm(config.n_embd)
        self.mlp = MLP(config)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x + self.attn(self.ln_1(x))
        x = x + self.mlp(self.ln_2(x))
        return x


# ---------------------------------------------------------------------------
# GPT-2 Model
# ---------------------------------------------------------------------------

class GPT2(nn.Module):
    def __init__(self, config: GPT2Config):
        super().__init__()
        self.config = config

        # Token + position embeddings (position is learned, not rotary)
        self.wte = nn.Embedding(config.vocab_size, config.n_embd)
        self.wpe = nn.Embedding(config.seq_len, config.n_embd)

        self.blocks = nn.ModuleList([Block(config) for _ in range(config.n_layer)])
        self.ln_f = RMSNorm(config.n_embd)
        self.lm_head = nn.Linear(config.n_embd, config.vocab_size, bias=False)

        # Initialize weights
        self.apply(self._init_weights)
        for name, p in self.named_parameters():
            if name.endswith("W_o.weight") or name.endswith("c_proj.weight"):
                nn.init.normal_(p, mean=0.0, std=0.02 / math.sqrt(2 * config.n_layer))

    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            nn.init.normal_(module.weight, mean=0.0, std=0.02)
        elif isinstance(module, nn.Embedding):
            nn.init.normal_(module.weight, mean=0.0, std=0.02)

    def forward(self, idx, targets=None):
        B, T = idx.shape
        assert T <= self.config.seq_len, f"Sequence length {T} exceeds max {self.config.seq_len}"

        pos = torch.arange(T, device=idx.device)
        x = self.wte(idx) + self.wpe(pos)

        for block in self.blocks:
            x = block(x)

        x = self.ln_f(x)
        logits = self.lm_head(x)

        loss = None
        if targets is not None:
            loss = F.cross_entropy(logits.view(-1, logits.size(-1)), targets.view(-1))
        return logits, loss

    def count_parameters(self):
        total = sum(p.numel() for p in self.parameters())
        print(f"Total parameters: {total:,}")
        print(f"  Token embeddings (wte):    {self.wte.weight.numel():,}")
        print(f"  Position embeddings (wpe): {self.wpe.weight.numel():,}")
        for i, block in enumerate(self.blocks):
            block_params = sum(p.numel() for p in block.parameters())
            print(f"  Block {i}: {block_params:,}")
        print(f"  Final norm:  {sum(p.numel() for p in self.ln_f.parameters()):,}")
        print(f"  LM head:     {self.lm_head.weight.numel():,}")
        return total


# ---------------------------------------------------------------------------
# Tests — run with: uv run python -m architectures.gpt2
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    config = GPT2Config()
    model = GPT2(config)
    print(f"GPT-2 Config: {config}\n")
    model.count_parameters()

    # Test forward pass
    idx = torch.randint(0, config.vocab_size, (2, 32))
    logits, loss = model(idx)
    assert logits.shape == (2, 32, config.vocab_size), f"Logit shape wrong: {logits.shape}"
    print(f"\nForward pass: idx {idx.shape} -> logits {logits.shape}")

    # Test with targets
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

    # Verify it has position embeddings (unlike Llama which uses RoPE)
    assert hasattr(model, "wpe"), "GPT-2 should have learned positional embeddings"
    print(f"\nPosition embeddings: learned (seq_len={config.seq_len})")

    print("\nAll GPT-2 tests passed!")
