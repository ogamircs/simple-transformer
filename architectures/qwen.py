"""
Qwen 2.5 — a refined Llama variant with smarter scaling and parameter sharing.

Qwen builds on Llama's foundation (RoPE, GQA, SwiGLU, RMSNorm) but adds four
innovations that improve long-context handling and parameter efficiency:

1. Selective Bias (QKV bias=True, all other linears bias=False)
   Most transformers go all-bias or no-bias. Qwen adds bias ONLY to the Q, K, V
   projections, which gives the attention mechanism more freedom to shift its
   operating point (especially helpful for context extrapolation), while keeping
   the rest of the network clean and regularized.

2. Tied Embeddings (wte.weight == lm_head.weight)
   The input embedding and output projection share the same weight matrix.
   This creates a symmetric encode/decode path: the same vectors that map
   tokens into embedding space also map hidden states back to token logits.
   Cuts parameter count and acts as a regularizer.

3. NTK-aware RoPE Interpolation
   Standard RoPE breaks when sequence length exceeds training length.
   NTK-aware interpolation scales the base frequency dynamically:
     base_scaled = base * ((alpha * seq_len / max_train_len - (alpha-1))
                          ** (dim / (dim - 2)))
   This stretches the slower-rotating (lower-frequency) dimensions more than
   fast ones, preserving local resolution while extending global reach.
   The model can handle longer contexts without any fine-tuning.

4. LogN Attention Scaling
   As sequences get longer, attention entropy increases and the distribution
   flattens. LogN scaling compensates by multiplying attention scores by
   log(seq_len) / log(train_len). At training length the factor is 1.0;
   for longer sequences it grows, keeping attention scores sharp.

Reference: Qwen 2.5 (2024-09), 7B parameters
  - GQA with RoPE + NTK-aware interpolation
  - Pre-norm with RMSNorm
  - SwiGLU activation in MLP (no bias)
  - Selective bias: QKV projections only
  - Tied input/output embeddings
  - LogN attention scaling
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
class QwenConfig:
    vocab_size: int = 256       # byte-level tokenization (same as base model)
    n_embd: int = 64            # embedding dimension
    n_head: int = 4             # number of query heads
    n_kv_head: int = 2          # number of key/value heads (GQA)
    n_layer: int = 4            # number of transformer blocks
    seq_len: int = 256          # maximum sequence length
    dropout: float = 0.0        # dropout rate
    rope_theta: float = 10000.0 # RoPE base frequency
    tie_embeddings: bool = True # share wte and lm_head weights
    max_train_len: int = 256    # training length (for LogN scaling reference)
    ntk_alpha: float = 1.0      # extra RoPE extrapolation factor for long contexts


# ---------------------------------------------------------------------------
# RMSNorm (same as Llama)
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
# RoPE (Rotary Position Embeddings) — same as Llama
# ---------------------------------------------------------------------------

def precompute_rope_freqs(head_dim: int, seq_len: int, theta: float = 10000.0):
    """Precompute the complex exponentials for RoPE.

    Returns cos and sin tensors of shape (seq_len, head_dim // 2).
    """
    freqs = 1.0 / (theta ** (torch.arange(0, head_dim, 2).float() / head_dim))
    t = torch.arange(seq_len).float()
    angles = torch.outer(t, freqs)  # (seq_len, head_dim // 2)
    return torch.cos(angles), torch.sin(angles)


def apply_rope(x: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor) -> torch.Tensor:
    """Apply rotary embeddings to a tensor.

    Args:
        x: shape (batch, n_head, seq_len, head_dim)
        cos, sin: shape (seq_len, head_dim // 2), precomputed frequencies
    """
    T = x.shape[2]
    x_pairs = x.float().reshape(*x.shape[:-1], -1, 2)
    x0 = x_pairs[..., 0]  # even dimensions
    x1 = x_pairs[..., 1]  # odd dimensions

    cos_t = cos[:T].unsqueeze(0).unsqueeze(0)  # (1, 1, T, head_dim//2)
    sin_t = sin[:T].unsqueeze(0).unsqueeze(0)  # (1, 1, T, head_dim//2)

    out0 = x0 * cos_t - x1 * sin_t
    out1 = x0 * sin_t + x1 * cos_t

    out = torch.stack([out0, out1], dim=-1).flatten(-2)
    return out.type_as(x)


# ---------------------------------------------------------------------------
# NTK-aware RoPE Interpolation
# ---------------------------------------------------------------------------
# Standard RoPE uses a fixed base frequency (e.g. 10000). When the sequence
# length exceeds the training length, the rotation angles go beyond what the
# model has seen, and quality degrades.
#
# NTK-aware interpolation scales the base frequency so that:
#   - High-frequency dimensions (local patterns) are barely changed
#   - Low-frequency dimensions (global patterns) are stretched more
#
# This preserves fine-grained local attention while extending global reach.
# The key formula:
#   alpha = scale_factor (typically > 1 for extrapolation)
#   base_scaled = base * ((alpha * seq_len / max_train_len - (alpha-1))
#                        ** (dim / (dim - 2)))

def ntk_aware_rope_freqs(head_dim: int, seq_len: int, max_train_len: int,
                         theta: float = 10000.0, alpha: float = 1.0):
    """Compute RoPE frequencies with NTK-aware interpolation.

    When seq_len <= max_train_len, this is equivalent to standard RoPE.
    When seq_len > max_train_len, the base is scaled up to preserve
    local resolution while extending global position coverage.
    """
    if seq_len > max_train_len:
        if head_dim <= 2:
            # The NTK exponent is undefined at head_dim == 2; tiny configs
            # fall back to standard RoPE instead of crashing.
            return precompute_rope_freqs(head_dim, seq_len, theta)
        # Scale the base frequency using the NTK-aware formula
        scale = alpha * seq_len / max_train_len - (alpha - 1)
        theta = theta * (scale ** (head_dim / (head_dim - 2)))
    return precompute_rope_freqs(head_dim, seq_len, theta)


def logn_attention_scale(seq_len: int, max_train_len: int) -> float:
    """Scale attention only when decoding past the training context."""
    if seq_len <= 1 or max_train_len <= 1:
        return 1.0
    return max(1.0, math.log(seq_len) / math.log(max_train_len))


# ---------------------------------------------------------------------------
# Qwen Attention (GQA + Selective Bias + LogN Scaling)
# ---------------------------------------------------------------------------
# Three differences from Llama's GroupedQueryAttention:
#
# 1. Selective bias: W_q, W_k, W_v have bias=True; W_o has bias=False
#    This gives the attention projections more flexibility to shift their
#    operating point, improving context extrapolation.
#
# 2. LogN attention scaling: after computing Q·K^T / sqrt(d), multiply by
#    log(T) / log(train_len). This compensates for attention entropy
#    increasing with sequence length. At training length it's 1.0.
#
# 3. NTK-aware RoPE is applied via the precomputed frequencies passed in.

class QwenAttention(nn.Module):
    def __init__(self, config: QwenConfig, cos: torch.Tensor, sin: torch.Tensor):
        super().__init__()
        assert config.n_embd % config.n_head == 0
        assert config.n_head % config.n_kv_head == 0

        self.n_head = config.n_head
        self.n_kv_head = config.n_kv_head
        self.n_groups = config.n_head // config.n_kv_head
        self.head_dim = config.n_embd // config.n_head
        self.max_train_len = config.max_train_len
        self.rope_theta = config.rope_theta
        self.ntk_alpha = config.ntk_alpha

        # SELECTIVE BIAS: QKV have bias=True, output has bias=False
        self.W_q = nn.Linear(config.n_embd, config.n_head * self.head_dim, bias=True)
        self.W_k = nn.Linear(config.n_embd, config.n_kv_head * self.head_dim, bias=True)
        self.W_v = nn.Linear(config.n_embd, config.n_kv_head * self.head_dim, bias=True)
        self.W_o = nn.Linear(config.n_embd, config.n_embd, bias=False)

        self.dropout = nn.Dropout(config.dropout)

        # RoPE frequencies (registered as buffers so they move to the right device)
        self.register_buffer("rope_cos", cos)
        self.register_buffer("rope_sin", sin)

        # Causal mask
        causal_mask = torch.tril(torch.ones(config.seq_len, config.seq_len))
        self.register_buffer("causal_mask", causal_mask.view(1, 1, config.seq_len, config.seq_len))

    def _rope_tables(self, seq_len: int, device: torch.device) -> tuple[torch.Tensor, torch.Tensor]:
        if seq_len <= self.max_train_len:
            return self.rope_cos[:seq_len], self.rope_sin[:seq_len]

        cos, sin = ntk_aware_rope_freqs(
            self.head_dim,
            seq_len,
            self.max_train_len,
            self.rope_theta,
            self.ntk_alpha,
        )
        return cos.to(device), sin.to(device)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, T, C = x.shape

        # Project to Q, K, V (with bias!)
        q = self.W_q(x).view(B, T, self.n_head, self.head_dim).transpose(1, 2)
        k = self.W_k(x).view(B, T, self.n_kv_head, self.head_dim).transpose(1, 2)
        v = self.W_v(x).view(B, T, self.n_kv_head, self.head_dim).transpose(1, 2)

        # Apply RoPE to Q and K
        rope_cos, rope_sin = self._rope_tables(T, x.device)
        q = apply_rope(q, rope_cos, rope_sin)
        k = apply_rope(k, rope_cos, rope_sin)

        # Expand K,V to match Q heads by repeating each KV head n_groups times
        if self.n_groups > 1:
            k = k.repeat_interleave(self.n_groups, dim=1)
            v = v.repeat_interleave(self.n_groups, dim=1)

        # Standard scaled dot-product attention
        attn = (q @ k.transpose(-2, -1)) / math.sqrt(self.head_dim)

        # LOGN SCALING: compensate for attention entropy increasing with length
        # At training length -> factor is 1.0. Longer -> factor > 1.0.
        attn = attn * logn_attention_scale(T, self.max_train_len)

        attn = attn.masked_fill(self.causal_mask[:, :, :T, :T] == 0, float("-inf"))
        attn = F.softmax(attn, dim=-1)
        attn = self.dropout(attn)

        out = attn @ v
        out = out.transpose(1, 2).contiguous().view(B, T, C)
        out = self.W_o(out)
        return out


# ---------------------------------------------------------------------------
# SwiGLU MLP (same as Llama — no bias)
# ---------------------------------------------------------------------------

class QwenSwiGLU(nn.Module):
    def __init__(self, config: QwenConfig):
        super().__init__()
        hidden_dim = int(4 * config.n_embd * 2 / 3)
        hidden_dim = ((hidden_dim + 7) // 8) * 8

        self.w_gate = nn.Linear(config.n_embd, hidden_dim, bias=False)
        self.w_up = nn.Linear(config.n_embd, hidden_dim, bias=False)
        self.w_down = nn.Linear(hidden_dim, config.n_embd, bias=False)
        self.dropout = nn.Dropout(config.dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        gate = F.silu(self.w_gate(x))
        up = self.w_up(x)
        x = gate * up
        x = self.w_down(x)
        x = self.dropout(x)
        return x


# ---------------------------------------------------------------------------
# Transformer Block (Qwen style — same pattern as Llama)
# ---------------------------------------------------------------------------

class QwenBlock(nn.Module):
    def __init__(self, config: QwenConfig, cos: torch.Tensor, sin: torch.Tensor):
        super().__init__()
        self.ln_1 = RMSNorm(config.n_embd)
        self.attn = QwenAttention(config, cos, sin)
        self.ln_2 = RMSNorm(config.n_embd)
        self.mlp = QwenSwiGLU(config)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x + self.attn(self.ln_1(x))
        x = x + self.mlp(self.ln_2(x))
        return x


# ---------------------------------------------------------------------------
# Qwen Model
# ---------------------------------------------------------------------------

class Qwen(nn.Module):
    def __init__(self, config: QwenConfig):
        super().__init__()
        self.config = config

        # Token embedding only — RoPE handles position
        self.wte = nn.Embedding(config.vocab_size, config.n_embd)

        # Precompute standard RoPE tables. NTK scaling is selected at runtime
        # so short inputs stay equivalent to training-length behavior.
        head_dim = config.n_embd // config.n_head
        cos, sin = precompute_rope_freqs(head_dim, config.seq_len, config.rope_theta)

        # Stack of transformer blocks
        self.blocks = nn.ModuleList([
            QwenBlock(config, cos, sin) for _ in range(config.n_layer)
        ])

        self.ln_f = RMSNorm(config.n_embd)
        self.lm_head = nn.Linear(config.n_embd, config.vocab_size, bias=False)

        # TIED EMBEDDINGS: lm_head shares weights with wte
        # The same vectors that encode tokens into embedding space
        # also decode hidden states back to token logits.
        if config.tie_embeddings:
            self.lm_head.weight = self.wte.weight

        # Initialize weights
        self.apply(self._init_weights)
        for name, p in self.named_parameters():
            if name.endswith("W_o.weight") or name.endswith("w_down.weight"):
                nn.init.normal_(p, mean=0.0, std=0.02 / math.sqrt(2 * config.n_layer))

    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                nn.init.zeros_(module.bias)
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
        # With tied embeddings, unique params differ from total
        total = sum(p.numel() for p in self.parameters())
        print(f"Total parameters: {total:,}")
        if self.config.tie_embeddings:
            print(f"  (embeddings are tied — wte and lm_head share weights)")
        print(f"  Token embeddings (wte): {self.wte.weight.numel():,}")
        for i, block in enumerate(self.blocks):
            block_params = sum(p.numel() for p in block.parameters())
            print(f"  Block {i}: {block_params:,}")
        print(f"  Final norm: {sum(p.numel() for p in self.ln_f.parameters()):,}")
        print(f"  LM head:   {self.lm_head.weight.numel():,}" +
              (" (tied)" if self.config.tie_embeddings else ""))
        return total


# ---------------------------------------------------------------------------
# Tests — run with: uv run python -m architectures.qwen
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    config = QwenConfig()
    model = Qwen(config)
    print(f"Qwen Config: {config}\n")
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

    # Verify selective bias: QKV has bias, W_o does not
    block = model.blocks[0].attn
    assert block.W_q.bias is not None, "W_q should have bias"
    assert block.W_k.bias is not None, "W_k should have bias"
    assert block.W_v.bias is not None, "W_v should have bias"
    assert block.W_o.bias is None, "W_o should NOT have bias"
    print("\nSelective bias check: W_q/W_k/W_v have bias, W_o does not")

    # Verify tied embeddings
    assert model.lm_head.weight is model.wte.weight, "lm_head.weight should be wte.weight (tied)"
    print("Tied embeddings check: lm_head.weight is wte.weight")

    # Verify GQA: K,V should have fewer parameters than Q
    q_params = block.W_q.weight.numel()
    k_params = block.W_k.weight.numel()
    print(f"\nGQA check: Q params={q_params:,}, K params={k_params:,} (K should be {config.n_kv_head}/{config.n_head} of Q)")
    assert k_params == q_params * config.n_kv_head // config.n_head

    print("\nAll Qwen tests passed!")
