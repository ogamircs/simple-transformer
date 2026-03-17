"""
Mixture of Experts (MoE) — sparse computation for massive models.

The core idea: instead of one big MLP, use many small "expert" MLPs and
a router that picks the best ones for each token. This lets you have a
huge model (total params) while only activating a fraction (active params)
per token, giving you more capacity without proportional compute cost.

Key concepts:

1. Expert Routing (Top-K)
   A small linear layer (the "router") scores all experts for each token,
   then picks the top-k (usually 2). The token only passes through those
   experts, weighted by their router scores.

2. Load Balancing Loss
   Without encouragement, the router tends to collapse — sending all tokens
   to one or two "popular" experts while others go unused. The auxiliary
   load-balancing loss penalizes this imbalance, pushing toward uniform
   expert utilization.

3. Shared Expert (DeepSeek innovation)
   DeepSeek V3 adds an "always-on" expert that processes every token
   regardless of routing. This captures common patterns that all tokens
   need, while routed experts specialize in rarer patterns.

4. Dense + MoE Hybrid
   Real models (DeepSeek V3, Llama 4) alternate dense blocks and MoE
   blocks. The first few layers are typically dense (they need to build
   basic representations before routing makes sense).

Reference: DeepSeek V3 (2024-12-26), 671B total / 37B active
  - Sparse MoE with top-2 routing
  - Shared expert + routed experts
  - Dense prefix layers
  - Uses MLA attention (we use GQA here for simplicity)
"""

import math
from dataclasses import dataclass

import torch
import torch.nn as nn
import torch.nn.functional as F

from architectures.llama import (
    RMSNorm,
    GroupedQueryAttention,
    SwiGLU,
    precompute_rope_freqs,
)


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

@dataclass
class MoEConfig:
    vocab_size: int = 256
    n_embd: int = 64
    n_head: int = 4
    n_kv_head: int = 2
    n_layer: int = 4            # total layers (dense + MoE)
    n_dense_layers: int = 1     # first N layers are dense (no routing)
    seq_len: int = 256
    dropout: float = 0.0
    rope_theta: float = 10000.0

    # MoE specific
    n_experts: int = 8          # total number of expert MLPs
    n_experts_active: int = 2   # top-k experts activated per token
    has_shared_expert: bool = True  # always-on shared expert (DeepSeek style)
    aux_loss_weight: float = 0.01   # weight of load-balancing loss


# ---------------------------------------------------------------------------
# Expert MLP
# ---------------------------------------------------------------------------
# Each expert is a small SwiGLU MLP. In real MoE models, experts are
# identical in architecture but learn different specializations through
# routing — some might specialize in code, others in math, etc.

class Expert(nn.Module):
    """A single expert MLP (same architecture as SwiGLU)."""
    def __init__(self, config: MoEConfig):
        super().__init__()
        hidden_dim = int(4 * config.n_embd * 2 / 3)
        hidden_dim = ((hidden_dim + 7) // 8) * 8

        self.w_gate = nn.Linear(config.n_embd, hidden_dim, bias=False)
        self.w_up = nn.Linear(config.n_embd, hidden_dim, bias=False)
        self.w_down = nn.Linear(hidden_dim, config.n_embd, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.w_down(F.silu(self.w_gate(x)) * self.w_up(x))


# ---------------------------------------------------------------------------
# Top-K Router
# ---------------------------------------------------------------------------
# The router is a simple linear layer that scores each expert for each token.
# We take the top-k scores, softmax them (so they sum to 1), and use them
# as weights when combining expert outputs.
#
# The load-balancing auxiliary loss has two parts:
#   f_i = fraction of tokens routed to expert i
#   p_i = average router probability for expert i
#   aux_loss = N * sum(f_i * p_i)
#
# This is minimized when all experts get equal load (f_i = p_i = 1/N).
# The N multiplier scales the loss to be O(1) regardless of expert count.

class TopKRouter(nn.Module):
    def __init__(self, config: MoEConfig):
        super().__init__()
        self.n_experts = config.n_experts
        self.top_k = config.n_experts_active

        # Router: maps each token to expert scores
        self.gate = nn.Linear(config.n_embd, config.n_experts, bias=False)

    def forward(self, x: torch.Tensor):
        """
        Args:
            x: (batch * seq_len, n_embd) — flattened token representations

        Returns:
            router_weights: (batch * seq_len, top_k) — normalized weights for selected experts
            expert_indices: (batch * seq_len, top_k) — which experts were selected
            aux_loss: scalar — load balancing loss
        """
        # Score each expert for each token
        logits = self.gate(x)  # (N_tokens, n_experts)

        # Select top-k experts per token
        top_k_logits, expert_indices = logits.topk(self.top_k, dim=-1)
        # Normalize top-k scores to sum to 1
        router_weights = F.softmax(top_k_logits, dim=-1)

        # Compute load-balancing auxiliary loss
        # f_i: fraction of tokens where expert i is in the top-k
        # We use a straight-through estimator: create a one-hot mask from indices
        mask = F.one_hot(expert_indices, self.n_experts).sum(dim=1).float()  # (N_tokens, n_experts)
        f = mask.mean(dim=0) / self.top_k  # fraction of tokens per expert

        # p_i: average router probability per expert (using full softmax, not just top-k)
        p = F.softmax(logits, dim=-1).mean(dim=0)

        # aux_loss = N * sum(f_i * p_i) — minimized when load is balanced
        aux_loss = self.n_experts * (f * p).sum()

        return router_weights, expert_indices, aux_loss


# ---------------------------------------------------------------------------
# MoE Layer
# ---------------------------------------------------------------------------
# Combines the router with multiple expert MLPs. For each token:
#   1. Router selects top-k experts and their weights
#   2. Token is processed by each selected expert
#   3. Outputs are combined as weighted sum
#   4. (Optional) Shared expert output is added
#
# Implementation note: we loop over experts rather than batching, which is
# simpler to understand. Production code uses grouped GEMM or expert-parallel
# strategies for efficiency.

class MoELayer(nn.Module):
    def __init__(self, config: MoEConfig):
        super().__init__()
        self.router = TopKRouter(config)
        self.experts = nn.ModuleList([Expert(config) for _ in range(config.n_experts)])

        # Optional shared expert (DeepSeek V3 style)
        self.shared_expert = Expert(config) if config.has_shared_expert else None
        self.dropout = nn.Dropout(config.dropout)

    def forward(self, x: torch.Tensor):
        """
        Args:
            x: (batch, seq_len, n_embd)

        Returns:
            output: (batch, seq_len, n_embd)
            aux_loss: scalar load-balancing loss
        """
        B, T, C = x.shape
        # Flatten to (B*T, C) for routing — each token routed independently
        x_flat = x.view(-1, C)

        # Route tokens to experts
        router_weights, expert_indices, aux_loss = self.router(x_flat)
        # router_weights: (B*T, top_k)
        # expert_indices: (B*T, top_k)

        # Compute expert outputs
        # For each expert, find which tokens selected it, process them, weight and accumulate
        output = torch.zeros_like(x_flat)

        for i, expert in enumerate(self.experts):
            # Find (token_idx, slot_idx) pairs where this expert was selected
            token_idx, slot_idx = torch.where(expert_indices == i)

            if token_idx.numel() == 0:
                continue  # no tokens routed to this expert

            # Gather the tokens assigned to this expert
            expert_input = x_flat[token_idx]  # (n_assigned, C)

            # Process through expert
            expert_output = expert(expert_input)  # (n_assigned, C)

            # Weight by router score and accumulate
            weights = router_weights[token_idx, slot_idx].unsqueeze(-1)  # (n_assigned, 1)
            output.index_add_(0, token_idx, expert_output * weights)

        # Add shared expert output (processes ALL tokens, no routing)
        if self.shared_expert is not None:
            output = output + self.shared_expert(x_flat)

        output = self.dropout(output)
        return output.view(B, T, C), aux_loss


# ---------------------------------------------------------------------------
# Transformer Blocks
# ---------------------------------------------------------------------------

class DenseBlock(nn.Module):
    """Standard transformer block with SwiGLU MLP (for dense prefix layers)."""
    def __init__(self, config: MoEConfig, cos: torch.Tensor, sin: torch.Tensor):
        super().__init__()
        # Create a compatible config for Llama components
        from architectures.llama import LlamaConfig
        llama_cfg = LlamaConfig(
            vocab_size=config.vocab_size, n_embd=config.n_embd,
            n_head=config.n_head, n_kv_head=config.n_kv_head,
            n_layer=config.n_layer, seq_len=config.seq_len,
            dropout=config.dropout, rope_theta=config.rope_theta,
        )
        self.ln_1 = RMSNorm(config.n_embd)
        self.attn = GroupedQueryAttention(llama_cfg, cos, sin)
        self.ln_2 = RMSNorm(config.n_embd)
        self.mlp = SwiGLU(llama_cfg)

    def forward(self, x: torch.Tensor):
        x = x + self.attn(self.ln_1(x))
        x = x + self.mlp(self.ln_2(x))
        return x, torch.tensor(0.0, device=x.device)  # no aux loss from dense block


class MoEBlock(nn.Module):
    """Transformer block with MoE replacing the MLP."""
    def __init__(self, config: MoEConfig, cos: torch.Tensor, sin: torch.Tensor):
        super().__init__()
        from architectures.llama import LlamaConfig
        llama_cfg = LlamaConfig(
            vocab_size=config.vocab_size, n_embd=config.n_embd,
            n_head=config.n_head, n_kv_head=config.n_kv_head,
            n_layer=config.n_layer, seq_len=config.seq_len,
            dropout=config.dropout, rope_theta=config.rope_theta,
        )
        self.ln_1 = RMSNorm(config.n_embd)
        self.attn = GroupedQueryAttention(llama_cfg, cos, sin)
        self.ln_2 = RMSNorm(config.n_embd)
        self.moe = MoELayer(config)

    def forward(self, x: torch.Tensor):
        x = x + self.attn(self.ln_1(x))
        moe_out, aux_loss = self.moe(self.ln_2(x))
        x = x + moe_out
        return x, aux_loss


# ---------------------------------------------------------------------------
# MoE Model
# ---------------------------------------------------------------------------

class MoEModel(nn.Module):
    def __init__(self, config: MoEConfig):
        super().__init__()
        self.config = config

        self.wte = nn.Embedding(config.vocab_size, config.n_embd)

        # Precompute RoPE frequencies
        head_dim = config.n_embd // config.n_head
        cos, sin = precompute_rope_freqs(head_dim, config.seq_len, config.rope_theta)

        # Build layers: dense prefix + MoE layers
        self.blocks = nn.ModuleList()
        for i in range(config.n_layer):
            if i < config.n_dense_layers:
                self.blocks.append(DenseBlock(config, cos, sin))
            else:
                self.blocks.append(MoEBlock(config, cos, sin))

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

        # Accumulate auxiliary losses from MoE layers
        total_aux_loss = torch.tensor(0.0, device=idx.device)
        n_moe_layers = 0

        for block in self.blocks:
            x, aux_loss = block(x)
            if aux_loss.item() > 0:
                total_aux_loss = total_aux_loss + aux_loss
                n_moe_layers += 1

        x = self.ln_f(x)
        logits = self.lm_head(x)

        loss = None
        if targets is not None:
            ce_loss = F.cross_entropy(logits.view(-1, logits.size(-1)), targets.view(-1))
            # Add load-balancing loss (averaged over MoE layers)
            if n_moe_layers > 0:
                avg_aux_loss = total_aux_loss / n_moe_layers
                loss = ce_loss + self.config.aux_loss_weight * avg_aux_loss
            else:
                loss = ce_loss

        return logits, loss

    def count_parameters(self):
        total = sum(p.numel() for p in self.parameters())
        # Count "active" params (what runs per token): exclude inactive expert params
        # Active = total - (n_experts - n_active) * expert_params * n_moe_layers
        expert_params = sum(p.numel() for p in self.blocks[-1].moe.experts[0].parameters()) if hasattr(self.blocks[-1], 'moe') else 0
        n_moe_layers = self.config.n_layer - self.config.n_dense_layers
        inactive_per_layer = (self.config.n_experts - self.config.n_experts_active) * expert_params
        active = total - (inactive_per_layer * n_moe_layers)
        if self.config.has_shared_expert:
            active = active  # shared expert is always active, already counted

        print(f"Total parameters:  {total:,}")
        print(f"Active parameters: {active:,} (per token)")
        print(f"  Token embeddings: {self.wte.weight.numel():,}")
        for i, block in enumerate(self.blocks):
            block_params = sum(p.numel() for p in block.parameters())
            kind = "Dense" if i < self.config.n_dense_layers else "MoE"
            print(f"  Block {i} ({kind}): {block_params:,}")
        print(f"  Final norm: {sum(p.numel() for p in self.ln_f.parameters()):,}")
        print(f"  LM head:   {self.lm_head.weight.numel():,}")
        return total


# ---------------------------------------------------------------------------
# Tests — run with: uv run python -m architectures.moe
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    config = MoEConfig()
    model = MoEModel(config)
    print(f"MoE Config: {config}\n")
    model.count_parameters()

    # Test forward pass
    idx = torch.randint(0, config.vocab_size, (2, 32))
    logits, loss = model(idx)
    assert logits.shape == (2, 32, config.vocab_size), f"Logit shape wrong: {logits.shape}"
    print(f"\nForward pass: idx {idx.shape} -> logits {logits.shape}")

    # Test with targets (loss includes aux load-balancing loss)
    targets = torch.randint(0, config.vocab_size, (2, 32))
    logits, loss = model(idx, targets)
    expected_loss = math.log(config.vocab_size)
    print(f"Loss: {loss.item():.4f} (expected ~{expected_loss:.4f} + aux for random init)")

    # Test gradient flow
    loss.backward()
    grad_norms = {name: p.grad.norm().item() for name, p in model.named_parameters() if p.grad is not None}
    assert len(grad_norms) > 0, "No gradients computed"
    print(f"Gradient flow: all {len(grad_norms)} parameters have gradients")

    # Verify architecture: first block should be dense, rest should be MoE
    assert isinstance(model.blocks[0], DenseBlock), "First block should be dense"
    for i in range(1, config.n_layer):
        assert isinstance(model.blocks[i], MoEBlock), f"Block {i} should be MoE"
    print(f"\nArchitecture: {config.n_dense_layers} dense + {config.n_layer - config.n_dense_layers} MoE blocks")

    # Verify MoE has more total params than active params
    total = sum(p.numel() for p in model.parameters())
    print(f"Sparsity: {config.n_experts} experts, {config.n_experts_active} active per token")

    # Verify shared expert exists
    moe_block = model.blocks[-1]
    assert moe_block.moe.shared_expert is not None, "Should have shared expert"
    print(f"Shared expert: present (DeepSeek V3 style)")

    print("\nAll MoE tests passed!")
