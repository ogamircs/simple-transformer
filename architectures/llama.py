"""
Llama 3 — the modern dense transformer baseline.

Three key innovations over GPT-2:

1. RoPE (Rotary Position Embeddings)
   Instead of learned position vectors added to token embeddings, RoPE encodes
   position by ROTATING query and key vectors. The angle of rotation depends
   on the position, so the dot product Q·K naturally encodes relative distance.
   This means the model generalizes better to unseen sequence lengths.

2. GQA (Grouped Query Attention)
   Standard multi-head attention uses separate Q, K, V for each head.
   GQA shares K and V across groups of query heads. For example, 8 query heads
   might share 2 KV heads (group size = 4). This cuts KV-cache memory during
   inference while keeping most of the quality.

3. SwiGLU (Swish-Gated Linear Unit)
   Replaces the simple ReLU/GELU MLP with a gated architecture:
     SwiGLU(x) = Swish(W_gate · x) * (W_up · x)
   The gating mechanism lets the network learn what to pass through,
   which consistently outperforms standard activations.

Reference: Llama 3 8B (2024-04-18), 8B parameters
  - GQA with RoPE
  - Pre-norm with RMSNorm
  - SwiGLU activation in MLP
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
class LlamaConfig:
    vocab_size: int = 256       # byte-level tokenization (same as base model)
    n_embd: int = 64            # embedding dimension
    n_head: int = 4             # number of query heads
    n_kv_head: int = 2          # number of key/value heads (GQA: n_kv_head < n_head)
    n_layer: int = 4            # number of transformer blocks
    seq_len: int = 256          # maximum sequence length
    dropout: float = 0.0        # dropout rate
    rope_theta: float = 10000.0 # RoPE base frequency


# ---------------------------------------------------------------------------
# RMSNorm (same as base model)
# ---------------------------------------------------------------------------

class RMSNorm(nn.Module):
    def __init__(self, dim: int, eps: float = 1e-6):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        rms = torch.sqrt(torch.mean(x ** 2, dim=-1, keepdim=True) + self.eps)
        return (x / rms) * self.weight


# ---------------------------------------------------------------------------
# RoPE (Rotary Position Embeddings)
# ---------------------------------------------------------------------------
# The key insight: instead of ADDING position information, we ROTATE vectors.
#
# Each pair of dimensions (2i, 2i+1) in the head gets rotated by an angle
# that depends on the position. Deeper dimensions rotate more slowly
# (lower frequency), creating a multi-scale position encoding.
#
# Why rotation works:
#   When we compute Q·K, the dot product of two rotated vectors depends on
#   the DIFFERENCE in their rotation angles, which equals the relative
#   distance between positions. This means attention naturally becomes
#   relative-position-aware without needing explicit position embeddings.
#
# The frequency formula: theta_i = base^(-2i/d)
#   - Dimension 0,1: rotates fast (high frequency, local patterns)
#   - Dimension d-2,d-1: rotates slow (low frequency, global patterns)

def validate_rope_head_dim(head_dim: int) -> None:
    """RoPE rotates pairs of features, so each head needs an even width."""
    if head_dim % 2 != 0:
        raise ValueError(f"RoPE requires an even head_dim, got {head_dim}")


def precompute_rope_freqs(head_dim: int, seq_len: int, theta: float = 10000.0):
    """Precompute the complex exponentials for RoPE.

    Returns cos and sin tensors of shape (seq_len, head_dim // 2).
    """
    validate_rope_head_dim(head_dim)
    # Frequencies for each dimension pair: theta^(-2i/d) for i in [0, d/2)
    freqs = 1.0 / (theta ** (torch.arange(0, head_dim, 2).float() / head_dim))
    # Outer product with positions gives the rotation angles
    # positions: [0, 1, 2, ..., seq_len-1]
    # freqs: [theta^0, theta^(-2/d), theta^(-4/d), ...]
    t = torch.arange(seq_len).float()
    angles = torch.outer(t, freqs)  # (seq_len, head_dim // 2)
    return torch.cos(angles), torch.sin(angles)


def apply_rope(x: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor) -> torch.Tensor:
    """Apply rotary embeddings to a tensor.

    Args:
        x: shape (batch, n_head, seq_len, head_dim)
        cos, sin: shape (seq_len, head_dim // 2), precomputed frequencies

    The rotation is applied to pairs of dimensions:
        [x0, x1] -> [x0 * cos - x1 * sin, x0 * sin + x1 * cos]

    This is equivalent to multiplying by a 2D rotation matrix for each pair.
    """
    T = x.shape[2]
    # Split head_dim into pairs: (..., head_dim) -> (..., head_dim//2, 2)
    x_pairs = x.float().reshape(*x.shape[:-1], -1, 2)
    x0 = x_pairs[..., 0]  # even dimensions
    x1 = x_pairs[..., 1]  # odd dimensions

    # Slice cos/sin to match sequence length and reshape for broadcasting
    cos_t = cos[:T].unsqueeze(0).unsqueeze(0)  # (1, 1, T, head_dim//2)
    sin_t = sin[:T].unsqueeze(0).unsqueeze(0)  # (1, 1, T, head_dim//2)

    # Apply 2D rotation to each pair
    out0 = x0 * cos_t - x1 * sin_t
    out1 = x0 * sin_t + x1 * cos_t

    # Interleave back: stack pairs and flatten
    out = torch.stack([out0, out1], dim=-1).flatten(-2)
    return out.type_as(x)


# ---------------------------------------------------------------------------
# Grouped Query Attention (GQA)
# ---------------------------------------------------------------------------
# Standard MHA: each head has its own Q, K, V projections
# GQA: Q heads are grouped, and each group shares one K,V pair
#
# Example with n_head=8, n_kv_head=2:
#   Group 0: Q heads 0,1,2,3 share K0,V0
#   Group 1: Q heads 4,5,6,7 share K1,V1
#
# Why? During inference, the KV-cache stores K,V for all past tokens.
# With GQA, you only store n_kv_head sets instead of n_head sets,
# cutting memory by n_head/n_kv_head (4x in the example above).
#
# Special cases:
#   n_kv_head == n_head: standard MHA (no sharing)
#   n_kv_head == 1: MQA (multi-query attention, maximum sharing)

class GroupedQueryAttention(nn.Module):
    def __init__(self, config: LlamaConfig, cos: torch.Tensor, sin: torch.Tensor):
        super().__init__()
        assert config.n_embd % config.n_head == 0
        assert config.n_head % config.n_kv_head == 0

        self.n_head = config.n_head
        self.n_kv_head = config.n_kv_head
        self.n_groups = config.n_head // config.n_kv_head  # queries per KV group
        self.head_dim = config.n_embd // config.n_head
        validate_rope_head_dim(self.head_dim)

        # Q has n_head projections, K and V have n_kv_head (fewer!)
        self.W_q = nn.Linear(config.n_embd, config.n_head * self.head_dim, bias=False)
        self.W_k = nn.Linear(config.n_embd, config.n_kv_head * self.head_dim, bias=False)
        self.W_v = nn.Linear(config.n_embd, config.n_kv_head * self.head_dim, bias=False)
        self.W_o = nn.Linear(config.n_embd, config.n_embd, bias=False)

        self.dropout = nn.Dropout(config.dropout)

        # RoPE frequencies (registered as buffers so they move to the right device)
        self.register_buffer("rope_cos", cos)
        self.register_buffer("rope_sin", sin)

        # Causal mask
        causal_mask = torch.tril(torch.ones(config.seq_len, config.seq_len))
        self.register_buffer("causal_mask", causal_mask.view(1, 1, config.seq_len, config.seq_len))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, T, C = x.shape

        # Project to Q, K, V
        q = self.W_q(x).view(B, T, self.n_head, self.head_dim).transpose(1, 2)
        k = self.W_k(x).view(B, T, self.n_kv_head, self.head_dim).transpose(1, 2)
        v = self.W_v(x).view(B, T, self.n_kv_head, self.head_dim).transpose(1, 2)
        # q: (B, n_head, T, head_dim)
        # k, v: (B, n_kv_head, T, head_dim)

        # Apply RoPE to Q and K (not V — values don't need position info)
        q = apply_rope(q, self.rope_cos, self.rope_sin)
        k = apply_rope(k, self.rope_cos, self.rope_sin)

        # Expand K,V to match Q heads by repeating each KV head n_groups times
        # (B, n_kv_head, T, head_dim) -> (B, n_head, T, head_dim)
        if self.n_groups > 1:
            k = k.repeat_interleave(self.n_groups, dim=1)
            v = v.repeat_interleave(self.n_groups, dim=1)

        # Standard attention from here
        attn = (q @ k.transpose(-2, -1)) / math.sqrt(self.head_dim)
        attn = attn.masked_fill(self.causal_mask[:, :, :T, :T] == 0, float("-inf"))
        attn = F.softmax(attn, dim=-1)
        attn = self.dropout(attn)

        out = attn @ v
        out = out.transpose(1, 2).contiguous().view(B, T, C)
        out = self.W_o(out)
        return out


# ---------------------------------------------------------------------------
# SwiGLU MLP
# ---------------------------------------------------------------------------
# The standard MLP does: up_project -> activation -> down_project
# SwiGLU adds a gating mechanism:
#   SwiGLU(x) = Swish(W_gate · x) ⊙ (W_up · x)
#
# Swish(x) = x * sigmoid(x) — a smooth, self-gated activation
#
# The gate learns WHAT information to let through, while the up-projection
# learns WHAT information to compute. This separation consistently beats
# using a single activation like ReLU or GELU.
#
# Note: because we have two up-projections (gate + up) instead of one,
# we use a hidden_dim of (4 * n_embd * 2/3) to keep parameter count similar.
# In practice, Llama rounds this to the nearest multiple of 256.

class SwiGLU(nn.Module):
    def __init__(self, config: LlamaConfig):
        super().__init__()
        # Hidden dim adjusted to keep param count ~same as 4x expansion
        # 4 * n_embd * 2/3 ≈ 2.67 * n_embd (but we have 2 up-projections)
        hidden_dim = int(4 * config.n_embd * 2 / 3)
        # Round to nearest multiple of 8 for hardware efficiency
        hidden_dim = ((hidden_dim + 7) // 8) * 8

        self.w_gate = nn.Linear(config.n_embd, hidden_dim, bias=False)  # gate
        self.w_up = nn.Linear(config.n_embd, hidden_dim, bias=False)    # up-projection
        self.w_down = nn.Linear(hidden_dim, config.n_embd, bias=False)  # down-projection
        self.dropout = nn.Dropout(config.dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Gate: learns what to let through
        gate = F.silu(self.w_gate(x))  # Swish = SiLU = x * sigmoid(x)
        # Up: learns what to compute
        up = self.w_up(x)
        # Element-wise product: gated information
        x = gate * up
        # Down-project back to model dimension
        x = self.w_down(x)
        x = self.dropout(x)
        return x


# ---------------------------------------------------------------------------
# Transformer Block (Llama style)
# ---------------------------------------------------------------------------

class LlamaBlock(nn.Module):
    def __init__(self, config: LlamaConfig, cos: torch.Tensor, sin: torch.Tensor):
        super().__init__()
        self.ln_1 = RMSNorm(config.n_embd)
        self.attn = GroupedQueryAttention(config, cos, sin)
        self.ln_2 = RMSNorm(config.n_embd)
        self.mlp = SwiGLU(config)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x + self.attn(self.ln_1(x))
        x = x + self.mlp(self.ln_2(x))
        return x


# ---------------------------------------------------------------------------
# Llama Model
# ---------------------------------------------------------------------------

class Llama(nn.Module):
    def __init__(self, config: LlamaConfig):
        super().__init__()
        self.config = config

        # Token embedding only — no positional embedding! (RoPE handles position)
        self.wte = nn.Embedding(config.vocab_size, config.n_embd)

        # Precompute RoPE frequencies (shared across all layers)
        head_dim = config.n_embd // config.n_head
        cos, sin = precompute_rope_freqs(head_dim, config.seq_len, config.rope_theta)

        # Stack of transformer blocks
        self.blocks = nn.ModuleList([
            LlamaBlock(config, cos, sin) for _ in range(config.n_layer)
        ])

        self.ln_f = RMSNorm(config.n_embd)
        self.lm_head = nn.Linear(config.n_embd, config.vocab_size, bias=False)

        # Initialize weights
        self.apply(self._init_weights)
        for name, p in self.named_parameters():
            if name.endswith("W_o.weight") or name.endswith("w_down.weight"):
                nn.init.normal_(p, mean=0.0, std=0.02 / math.sqrt(2 * config.n_layer))

    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            nn.init.normal_(module.weight, mean=0.0, std=0.02)
        elif isinstance(module, nn.Embedding):
            nn.init.normal_(module.weight, mean=0.0, std=0.02)

    def forward(self, idx, targets=None):
        B, T = idx.shape
        assert T <= self.config.seq_len, f"Sequence length {T} exceeds max {self.config.seq_len}"

        # Only token embeddings — RoPE adds position info inside attention
        x = self.wte(idx)  # (B, T, n_embd)

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
        print(f"  Token embeddings (wte): {self.wte.weight.numel():,}")
        for i, block in enumerate(self.blocks):
            block_params = sum(p.numel() for p in block.parameters())
            print(f"  Block {i}: {block_params:,}")
        print(f"  Final norm: {sum(p.numel() for p in self.ln_f.parameters()):,}")
        print(f"  LM head:   {self.lm_head.weight.numel():,}")
        return total


# ---------------------------------------------------------------------------
# Tests — run with: uv run python -m architectures.llama
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    config = LlamaConfig()
    model = Llama(config)
    print(f"Llama Config: {config}\n")
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

    # Verify GQA: K,V should have fewer parameters than Q
    block = model.blocks[0].attn
    q_params = block.W_q.weight.numel()
    k_params = block.W_k.weight.numel()
    print(f"\nGQA check: Q params={q_params:,}, K params={k_params:,} (K should be {config.n_kv_head}/{config.n_head} of Q)")
    assert k_params == q_params * config.n_kv_head // config.n_head

    # Verify no positional embedding exists
    assert not hasattr(model, "wpe"), "Llama should not have learned positional embeddings (uses RoPE)"

    print("\nAll Llama tests passed!")
