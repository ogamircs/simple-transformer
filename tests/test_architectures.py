"""
Tests for all architecture implementations.

Run with: uv run python -m pytest tests/ -v
"""

import math

import torch
import pytest

from architectures.gpt2 import GPT2, GPT2Config
from architectures.llama import Llama, LlamaConfig
from architectures.qwen import (
    Qwen,
    QwenConfig,
    logn_attention_scale,
    precompute_rope_freqs,
)
from architectures.moe import MoEModel, MoEConfig
from architectures.sliding_window import SlidingWindowModel, SlidingWindowConfig


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def gpt2():
    config = GPT2Config(vocab_size=256, n_embd=64, n_head=4, n_layer=2, seq_len=128)
    return GPT2(config), config

@pytest.fixture
def llama():
    config = LlamaConfig(vocab_size=256, n_embd=64, n_head=4, n_kv_head=2, n_layer=2, seq_len=128)
    return Llama(config), config

@pytest.fixture
def qwen():
    config = QwenConfig(vocab_size=256, n_embd=64, n_head=4, n_kv_head=2, n_layer=2, seq_len=128)
    return Qwen(config), config

@pytest.fixture
def moe():
    config = MoEConfig(
        vocab_size=256, n_embd=64, n_head=4, n_kv_head=2,
        n_layer=3, n_dense_layers=1, seq_len=128,
        n_experts=4, n_experts_active=2, has_shared_expert=True,
    )
    return MoEModel(config), config

@pytest.fixture
def sliding_window():
    config = SlidingWindowConfig(
        vocab_size=256, n_embd=64, n_head=4, n_kv_head=2,
        n_layer=6, seq_len=128, window_size=32, global_every=3,
        use_qk_norm=True,
    )
    return SlidingWindowModel(config), config


# ---------------------------------------------------------------------------
# Shape tests
# ---------------------------------------------------------------------------

class TestForwardShapes:
    """Verify output shapes for all architectures."""

    def test_gpt2_shapes(self, gpt2):
        model, config = gpt2
        idx = torch.randint(0, config.vocab_size, (2, 32))
        logits, _ = model(idx)
        assert logits.shape == (2, 32, config.vocab_size)

    def test_llama_shapes(self, llama):
        model, config = llama
        idx = torch.randint(0, config.vocab_size, (2, 32))
        logits, _ = model(idx)
        assert logits.shape == (2, 32, config.vocab_size)

    def test_qwen_shapes(self, qwen):
        model, config = qwen
        idx = torch.randint(0, config.vocab_size, (2, 32))
        logits, _ = model(idx)
        assert logits.shape == (2, 32, config.vocab_size)

    def test_moe_shapes(self, moe):
        model, config = moe
        idx = torch.randint(0, config.vocab_size, (2, 32))
        logits, _ = model(idx)
        assert logits.shape == (2, 32, config.vocab_size)

    def test_sliding_window_shapes(self, sliding_window):
        model, config = sliding_window
        idx = torch.randint(0, config.vocab_size, (2, 32))
        logits, _ = model(idx)
        assert logits.shape == (2, 32, config.vocab_size)

    def test_single_token(self, llama):
        """Models should handle single-token sequences."""
        model, config = llama
        idx = torch.randint(0, config.vocab_size, (1, 1))
        logits, _ = model(idx)
        assert logits.shape == (1, 1, config.vocab_size)

    def test_full_sequence_length(self, gpt2):
        """Models should handle sequences up to max seq_len."""
        model, config = gpt2
        idx = torch.randint(0, config.vocab_size, (1, config.seq_len))
        logits, _ = model(idx)
        assert logits.shape == (1, config.seq_len, config.vocab_size)


# ---------------------------------------------------------------------------
# Loss tests
# ---------------------------------------------------------------------------

class TestLoss:
    """Verify loss computation is reasonable."""

    def test_gpt2_random_init_loss(self, gpt2):
        model, config = gpt2
        idx = torch.randint(0, config.vocab_size, (4, 32))
        targets = torch.randint(0, config.vocab_size, (4, 32))
        _, loss = model(idx, targets)
        expected = math.log(config.vocab_size)
        assert abs(loss.item() - expected) < 1.5, f"Loss {loss.item():.2f} too far from {expected:.2f}"

    def test_llama_random_init_loss(self, llama):
        model, config = llama
        idx = torch.randint(0, config.vocab_size, (4, 32))
        targets = torch.randint(0, config.vocab_size, (4, 32))
        _, loss = model(idx, targets)
        expected = math.log(config.vocab_size)
        assert abs(loss.item() - expected) < 1.5

    def test_qwen_random_init_loss(self, qwen):
        model, config = qwen
        idx = torch.randint(0, config.vocab_size, (4, 32))
        targets = torch.randint(0, config.vocab_size, (4, 32))
        _, loss = model(idx, targets)
        expected = math.log(config.vocab_size)
        assert abs(loss.item() - expected) < 1.5

    def test_sliding_window_random_init_loss(self, sliding_window):
        model, config = sliding_window
        idx = torch.randint(0, config.vocab_size, (4, 32))
        targets = torch.randint(0, config.vocab_size, (4, 32))
        _, loss = model(idx, targets)
        expected = math.log(config.vocab_size)
        assert abs(loss.item() - expected) < 1.5

    def test_moe_loss_includes_aux(self, moe):
        """MoE loss should be slightly higher than CE due to aux loss."""
        model, config = moe
        idx = torch.randint(0, config.vocab_size, (4, 32))
        targets = torch.randint(0, config.vocab_size, (4, 32))
        _, loss = model(idx, targets)
        # Loss should be reasonable (CE + small aux)
        assert loss.item() > 0
        assert loss.item() < 10  # sanity bound

    def test_no_loss_without_targets(self, gpt2):
        model, config = gpt2
        idx = torch.randint(0, config.vocab_size, (2, 32))
        _, loss = model(idx)
        assert loss is None


# ---------------------------------------------------------------------------
# Gradient tests
# ---------------------------------------------------------------------------

class TestGradients:
    """Verify gradient flow through all parameters."""

    def _check_gradients(self, model, config):
        idx = torch.randint(0, config.vocab_size, (2, 16))
        targets = torch.randint(0, config.vocab_size, (2, 16))
        _, loss = model(idx, targets)
        loss.backward()
        params_with_grad = sum(1 for p in model.parameters() if p.grad is not None and p.grad.abs().sum() > 0)
        total_params = sum(1 for p in model.parameters())
        # Allow some router/expert params to have zero grad if no tokens routed there
        assert params_with_grad > total_params * 0.8, \
            f"Only {params_with_grad}/{total_params} params have nonzero gradients"

    def test_gpt2_gradients(self, gpt2):
        self._check_gradients(*gpt2)

    def test_llama_gradients(self, llama):
        self._check_gradients(*llama)

    def test_qwen_gradients(self, qwen):
        self._check_gradients(*qwen)

    def test_moe_gradients(self, moe):
        self._check_gradients(*moe)

    def test_sliding_window_gradients(self, sliding_window):
        self._check_gradients(*sliding_window)


# ---------------------------------------------------------------------------
# Architecture-specific tests
# ---------------------------------------------------------------------------

class TestArchitectureProperties:
    """Verify architectural distinctions between models."""

    def test_gpt2_has_position_embeddings(self, gpt2):
        model, _ = gpt2
        assert hasattr(model, "wpe"), "GPT-2 should have learned position embeddings"

    def test_llama_no_position_embeddings(self, llama):
        model, _ = llama
        assert not hasattr(model, "wpe"), "Llama should use RoPE, not learned positions"

    def test_qwen_no_position_embeddings(self, qwen):
        model, _ = qwen
        assert not hasattr(model, "wpe"), "Qwen should use RoPE, not learned positions"

    def test_qwen_tied_embeddings(self, qwen):
        """Qwen should tie input embedding and output projection weights."""
        model, _ = qwen
        # With tied embeddings, lm_head weight should be the same object as wte weight
        assert model.lm_head.weight is model.wte.weight, \
            "Qwen should have tied embeddings (lm_head.weight is wte.weight)"

    def test_qwen_selective_bias(self, qwen):
        """Qwen QKV should have bias, W_o should not."""
        model, _ = qwen
        attn = model.blocks[0].attn
        assert attn.W_q.bias is not None, "Qwen Q should have bias"
        assert attn.W_k.bias is not None, "Qwen K should have bias"
        assert attn.W_v.bias is not None, "Qwen V should have bias"
        assert attn.W_o.bias is None, "Qwen W_o should NOT have bias"

    def test_gqa_fewer_kv_params(self, llama):
        """GQA should have fewer KV params than Q params."""
        model, config = llama
        attn = model.blocks[0].attn
        q_params = attn.W_q.weight.numel()
        k_params = attn.W_k.weight.numel()
        assert k_params < q_params, "GQA: K should have fewer params than Q"
        assert k_params == q_params * config.n_kv_head // config.n_head

    def test_moe_more_total_than_active(self, moe):
        """MoE should have more total params than dense equivalent."""
        model, config = moe
        total = sum(p.numel() for p in model.parameters())
        dense_equiv = LlamaConfig(
            vocab_size=config.vocab_size, n_embd=config.n_embd,
            n_head=config.n_head, n_kv_head=config.n_kv_head,
            n_layer=config.n_layer, seq_len=config.seq_len,
        )
        dense_model = Llama(dense_equiv)
        dense_total = sum(p.numel() for p in dense_model.parameters())
        assert total > dense_total, "MoE should have more total params than dense"

    def test_moe_dense_prefix(self, moe):
        """MoE should have dense prefix layers."""
        model, config = moe
        from architectures.moe import DenseBlock, MoEBlock
        assert isinstance(model.blocks[0], DenseBlock)
        assert isinstance(model.blocks[1], MoEBlock)

    def test_sliding_window_alternating_pattern(self, sliding_window):
        """Sliding window should alternate local and global blocks."""
        model, config = sliding_window
        global_count = sum(1 for b in model.blocks if b.is_global)
        local_count = sum(1 for b in model.blocks if not b.is_global)
        assert global_count > 0, "Should have at least one global block"
        assert local_count > 0, "Should have at least one local block"
        assert local_count > global_count, "Should have more local than global blocks"

    def test_sliding_window_mask_blocks_distant(self, sliding_window):
        """Local attention mask should block distant tokens."""
        model, config = sliding_window
        local_block = next(b for b in model.blocks if not b.is_global)
        mask = local_block.attn.mask[0, 0]
        if config.seq_len > config.window_size:
            assert mask[-1, 0] == 0, "Sliding window should block distant tokens"

    def test_sliding_window_qk_norm(self, sliding_window):
        """Sliding window should have QK-Norm."""
        model, _ = sliding_window
        block = model.blocks[0]
        assert block.attn.qk_norm is not None, "QK-Norm should be enabled"

    def test_qwen_ntk_rope_scales_beyond_training_length_by_default(self):
        """Extended-context Qwen should not fall back to standard RoPE tables."""
        config = QwenConfig(
            vocab_size=256,
            n_embd=64,
            n_head=4,
            n_kv_head=2,
            n_layer=1,
            seq_len=512,
            max_train_len=256,
        )
        model = Qwen(config)
        head_dim = config.n_embd // config.n_head
        standard_cos, _ = precompute_rope_freqs(head_dim, config.seq_len, config.rope_theta)
        assert not torch.allclose(model.blocks[0].attn.rope_cos, standard_cos)

    def test_qwen_ntk_alpha_changes_rope_tables(self):
        """Changing NTK alpha should change the extrapolated RoPE tables."""
        base = QwenConfig(
            vocab_size=256,
            n_embd=64,
            n_head=4,
            n_kv_head=2,
            n_layer=1,
            seq_len=512,
            max_train_len=256,
            ntk_alpha=1.0,
        )
        scaled = QwenConfig(
            vocab_size=256,
            n_embd=64,
            n_head=4,
            n_kv_head=2,
            n_layer=1,
            seq_len=512,
            max_train_len=256,
            ntk_alpha=2.0,
        )
        base_model = Qwen(base)
        scaled_model = Qwen(scaled)
        assert not torch.allclose(base_model.blocks[0].attn.rope_cos, scaled_model.blocks[0].attn.rope_cos)

    def test_qwen_logn_scaling_clamps_short_context(self):
        """LogN scaling should not shrink attention scores below training-time behavior."""
        assert logn_attention_scale(seq_len=32, max_train_len=256) == 1.0

    def test_qwen_logn_scaling_grows_for_long_context(self):
        """LogN scaling should increase for longer-than-training contexts."""
        assert logn_attention_scale(seq_len=512, max_train_len=256) > 1.0


# ---------------------------------------------------------------------------
# Overfitting test (sanity check)
# ---------------------------------------------------------------------------

class TestOverfit:
    """Verify models can overfit a small batch (proves learning works)."""

    def _overfit_batch(self, model, config, steps=200):
        optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)
        idx = torch.randint(0, config.vocab_size, (2, 16))
        targets = torch.randint(0, config.vocab_size, (2, 16))

        initial_loss = None
        for step in range(steps):
            _, loss = model(idx, targets)
            if initial_loss is None:
                initial_loss = loss.item()
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        final_loss = loss.item()
        assert final_loss < initial_loss * 0.5, \
            f"Model didn't learn: {initial_loss:.4f} -> {final_loss:.4f}"
        return final_loss

    def test_gpt2_overfits(self, gpt2):
        self._overfit_batch(*gpt2)

    def test_llama_overfits(self, llama):
        self._overfit_batch(*llama)

    def test_qwen_overfits(self, qwen):
        self._overfit_batch(*qwen)

    def test_moe_overfits(self, moe):
        self._overfit_batch(*moe, steps=300)

    def test_sliding_window_overfits(self, sliding_window):
        self._overfit_batch(*sliding_window)
