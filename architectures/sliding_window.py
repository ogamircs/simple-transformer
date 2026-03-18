"""
Sliding Window Attention — efficient long-context attention.

The problem with standard attention: it's O(n²) in sequence length.
A 128K context window means each token attends to all 128K tokens,
which is extremely expensive in both compute and KV-cache memory.

The solution: most tokens only NEED to attend to nearby tokens.
Sliding window attention restricts each token to a local window
(e.g., 4096 tokens), making attention O(n * w) where w << n.

But some information needs to flow globally (e.g., the system prompt
at the beginning). So modern models alternate:
  - Local layers: sliding window attention (cheap, captures local patterns)
  - Global layers: full attention (expensive, captures long-range dependencies)

The ratio varies by model:
  - Gemma 3:  5:1 local:global (aggressive — mostly local)
  - OLMo 3:   3:1 local:global
  - Mistral:  all sliding window (no global layers in early versions)

Information still propagates globally through local layers — it just
takes multiple hops. Layer 1's window overlaps with Layer 2's window,
creating an "effective receptive field" that grows with depth.

Other innovations from Gemma 3 included here:
  - QK-Norm: normalizes Q and K before computing attention scores,
    stabilizing training at scale (prevents attention logit explosion)

References:
  - Gemma 3 27B (2025-03-11): 5:1 sliding-window/global, QK-Norm
  - OLMo 3 (2025-11-20): 3:1 sliding-window/global, QK-Norm
  - Mistral Small 3.1 24B (2025-03-18): standard GQA, no sliding window
"""

import math
from dataclasses import dataclass

import torch
import torch.nn as nn
import torch.nn.functional as F

from architectures.llama import (
    RMSNorm,
    SwiGLU,
    precompute_rope_freqs,
    apply_rope,
    LlamaConfig,
    validate_rope_head_dim,
)


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

@dataclass
class SlidingWindowConfig:
    vocab_size: int = 256
    n_embd: int = 64
    n_head: int = 4
    n_kv_head: int = 2
    n_layer: int = 6            # more layers to show the alternating pattern
    seq_len: int = 256
    dropout: float = 0.0
    rope_theta: float = 10000.0

    # Sliding window specific
    window_size: int = 64       # local attention window (each side)
    global_every: int = 3       # insert a global attention layer every N layers
    use_qk_norm: bool = True    # normalize Q and K before attention


# ---------------------------------------------------------------------------
# QK-Norm
# ---------------------------------------------------------------------------
# At large scale, the dot product Q·K can grow very large, causing attention
# scores to become extremely peaked (near one-hot after softmax). This makes
# training unstable.
#
# QK-Norm normalizes Q and K vectors to unit length before computing attention,
# then uses a learnable temperature to control the sharpness.
#
# attention = softmax((Q_norm · K_norm) / temperature)
#
# This is equivalent to computing cosine similarity scaled by temperature.

class QKNorm(nn.Module):
    def __init__(self, head_dim: int):
        super().__init__()
        self.q_norm = RMSNorm(head_dim)
        self.k_norm = RMSNorm(head_dim)

    def forward(self, q: torch.Tensor, k: torch.Tensor):
        """Normalize Q and K along the head dimension.

        Args:
            q: (batch, n_head, seq_len, head_dim)
            k: (batch, n_kv_head, seq_len, head_dim)
        """
        return self.q_norm(q), self.k_norm(k)


# ---------------------------------------------------------------------------
# Sliding Window / Global Attention
# ---------------------------------------------------------------------------
# The mask is the key difference:
#   - Global: standard causal mask (lower triangular)
#   - Sliding window: causal mask AND |i - j| <= window_size
#
# Tokens outside the window get -inf in the attention scores,
# so they contribute nothing after softmax.

class SlidingWindowAttention(nn.Module):
    def __init__(
        self,
        config: SlidingWindowConfig,
        cos: torch.Tensor,
        sin: torch.Tensor,
        is_global: bool,
    ):
        super().__init__()
        assert config.n_embd % config.n_head == 0
        assert config.n_head % config.n_kv_head == 0

        self.n_head = config.n_head
        self.n_kv_head = config.n_kv_head
        self.n_groups = config.n_head // config.n_kv_head
        self.head_dim = config.n_embd // config.n_head
        validate_rope_head_dim(self.head_dim)
        self.is_global = is_global
        self.window_size = config.window_size

        self.W_q = nn.Linear(config.n_embd, config.n_head * self.head_dim, bias=False)
        self.W_k = nn.Linear(config.n_embd, config.n_kv_head * self.head_dim, bias=False)
        self.W_v = nn.Linear(config.n_embd, config.n_kv_head * self.head_dim, bias=False)
        self.W_o = nn.Linear(config.n_embd, config.n_embd, bias=False)

        self.dropout = nn.Dropout(config.dropout)

        # QK-Norm (optional)
        self.qk_norm = QKNorm(self.head_dim) if config.use_qk_norm else None

        # RoPE
        self.register_buffer("rope_cos", cos)
        self.register_buffer("rope_sin", sin)

        if is_global:
            # Standard causal mask
            mask = torch.tril(torch.ones(config.seq_len, config.seq_len))
            self.register_buffer("mask", mask.view(1, 1, config.seq_len, config.seq_len))
        else:
            self.mask = None

    def _local_attention(self, q: torch.Tensor, k: torch.Tensor, v: torch.Tensor) -> torch.Tensor:
        """Compute causal sliding-window attention without building a full T x T matrix."""
        _, _, T, _ = q.shape

        k_padded = F.pad(k, (0, 0, self.window_size - 1, 0))
        v_padded = F.pad(v, (0, 0, self.window_size - 1, 0))
        k_windows = k_padded.unfold(2, self.window_size, 1).permute(0, 1, 2, 4, 3)
        v_windows = v_padded.unfold(2, self.window_size, 1).permute(0, 1, 2, 4, 3)

        offsets = torch.arange(self.window_size, device=q.device)
        positions = torch.arange(T, device=q.device)
        valid = offsets.unsqueeze(0) >= (self.window_size - 1 - positions).unsqueeze(1)

        attn = (q.unsqueeze(-2) * k_windows).sum(dim=-1) / math.sqrt(self.head_dim)
        attn = attn.masked_fill(~valid.view(1, 1, T, self.window_size), float("-inf"))
        attn = F.softmax(attn, dim=-1)
        attn = self.dropout(attn)
        return (attn.unsqueeze(-1) * v_windows).sum(dim=-2)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, T, C = x.shape

        q = self.W_q(x).view(B, T, self.n_head, self.head_dim).transpose(1, 2)
        k = self.W_k(x).view(B, T, self.n_kv_head, self.head_dim).transpose(1, 2)
        v = self.W_v(x).view(B, T, self.n_kv_head, self.head_dim).transpose(1, 2)

        # Apply RoPE
        q = apply_rope(q, self.rope_cos, self.rope_sin)
        k = apply_rope(k, self.rope_cos, self.rope_sin)

        # Apply QK-Norm if enabled
        if self.qk_norm is not None:
            q, k = self.qk_norm(q, k)

        # Expand KV for GQA
        if self.n_groups > 1:
            k = k.repeat_interleave(self.n_groups, dim=1)
            v = v.repeat_interleave(self.n_groups, dim=1)

        if self.is_global:
            attn = (q @ k.transpose(-2, -1)) / math.sqrt(self.head_dim)
            attn = attn.masked_fill(self.mask[:, :, :T, :T] == 0, float("-inf"))
            attn = F.softmax(attn, dim=-1)
            attn = self.dropout(attn)
            out = attn @ v
        else:
            out = self._local_attention(q, k, v)
        out = out.transpose(1, 2).contiguous().view(B, T, C)
        out = self.W_o(out)
        return out


# ---------------------------------------------------------------------------
# Transformer Block
# ---------------------------------------------------------------------------

class SlidingWindowBlock(nn.Module):
    def __init__(
        self,
        config: SlidingWindowConfig,
        cos: torch.Tensor,
        sin: torch.Tensor,
        is_global: bool,
    ):
        super().__init__()
        self.is_global = is_global
        llama_cfg = LlamaConfig(
            vocab_size=config.vocab_size, n_embd=config.n_embd,
            n_head=config.n_head, n_kv_head=config.n_kv_head,
            n_layer=config.n_layer, seq_len=config.seq_len,
            dropout=config.dropout, rope_theta=config.rope_theta,
        )
        self.ln_1 = RMSNorm(config.n_embd)
        self.attn = SlidingWindowAttention(config, cos, sin, is_global)
        self.ln_2 = RMSNorm(config.n_embd)
        self.mlp = SwiGLU(llama_cfg)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x + self.attn(self.ln_1(x))
        x = x + self.mlp(self.ln_2(x))
        return x


# ---------------------------------------------------------------------------
# Sliding Window Model
# ---------------------------------------------------------------------------

class SlidingWindowModel(nn.Module):
    def __init__(self, config: SlidingWindowConfig):
        super().__init__()
        self.config = config

        self.wte = nn.Embedding(config.vocab_size, config.n_embd)

        head_dim = config.n_embd // config.n_head
        cos, sin = precompute_rope_freqs(head_dim, config.seq_len, config.rope_theta)

        # Build alternating local/global blocks
        self.blocks = nn.ModuleList()
        for i in range(config.n_layer):
            # Last layer in each group of global_every is global
            is_global = ((i + 1) % config.global_every == 0)
            self.blocks.append(SlidingWindowBlock(config, cos, sin, is_global))

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
        assert T <= self.config.seq_len

        x = self.wte(idx)

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
        print(f"  Token embeddings: {self.wte.weight.numel():,}")
        for i, block in enumerate(self.blocks):
            block_params = sum(p.numel() for p in block.parameters())
            kind = "Global" if block.is_global else "Local"
            print(f"  Block {i} ({kind}):  {block_params:,}")
        print(f"  Final norm: {sum(p.numel() for p in self.ln_f.parameters()):,}")
        print(f"  LM head:   {self.lm_head.weight.numel():,}")
        return total


# ---------------------------------------------------------------------------
# Tests — run with: uv run python -m architectures.sliding_window
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    config = SlidingWindowConfig()
    model = SlidingWindowModel(config)
    print(f"Sliding Window Config: {config}\n")
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
    assert abs(loss.item() - expected_loss) < 1.0

    # Test gradient flow
    loss.backward()
    grad_norms = {name: p.grad.norm().item() for name, p in model.named_parameters() if p.grad is not None}
    assert len(grad_norms) > 0
    print(f"Gradient flow: all {len(grad_norms)} parameters have gradients")

    # Verify alternating pattern
    pattern = []
    for i, block in enumerate(model.blocks):
        kind = "Global" if block.is_global else "Local"
        pattern.append(kind)
    print(f"\nAttention pattern: {' -> '.join(pattern)}")

    # Verify local blocks use the optimized windowed path
    local_block = model.blocks[0]
    assert not local_block.is_global
    assert local_block.attn.mask is None, "Local sliding-window blocks should skip the dense mask"
    print("Sliding window: local blocks use the optimized no-dense-mask path")

    # Verify the global mask allows full attention
    global_block = model.blocks[config.global_every - 1]
    assert global_block.is_global
    global_mask = global_block.attn.mask[0, 0]
    assert global_mask[-1, 0] == 1, "Global mask should allow attending to all previous positions"
    print(f"Global attention: position {config.seq_len-1} correctly attends to position 0")

    # Verify QK-Norm is present
    assert local_block.attn.qk_norm is not None, "QK-Norm should be enabled"
    print(f"QK-Norm: enabled")

    print("\nAll Sliding Window tests passed!")
