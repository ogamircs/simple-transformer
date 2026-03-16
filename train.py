"""
Training loop for the simple transformer.

Concepts covered:
  - AdamW optimizer (Adam with decoupled weight decay)
  - Learning rate scheduling (linear warmup + cosine decay)
  - Overfitting sanity check (can the model memorize one batch?)
  - Mixed precision training (when GPU available)
  - Checkpoint saving/loading

Usage:
  uv run python train.py                    # train with defaults
  uv run python train.py --sanity-check     # overfit one batch (quick test)
"""

import argparse
import math
import os
import time

import torch

from data import get_batch, prepare_data, decode
from model import GPT, GPTConfig


# ---------------------------------------------------------------------------
# Learning rate schedule
# ---------------------------------------------------------------------------
# Two phases:
#   1. Warmup: linearly increase LR from 0 to max over first N steps.
#      This prevents early instability when weights are random.
#   2. Cosine decay: smoothly decrease LR from max to min.
#      This helps the model settle into a good minimum.
#
# Why not just use a constant LR?
#   - Too high: training diverges (loss explodes)
#   - Too low: training is slow and gets stuck
#   - Schedule: high enough to make progress, low enough to converge

def get_lr(step: int, max_lr: float, min_lr: float, warmup_steps: int, total_steps: int) -> float:
    """Compute learning rate for a given step (warmup + cosine decay)."""
    # Phase 1: Linear warmup
    if step < warmup_steps:
        return max_lr * (step + 1) / warmup_steps

    # Phase 2: Cosine decay
    progress = (step - warmup_steps) / max(1, total_steps - warmup_steps)
    cosine = 0.5 * (1 + math.cos(math.pi * progress))
    return min_lr + (max_lr - min_lr) * cosine


# ---------------------------------------------------------------------------
# Evaluation
# ---------------------------------------------------------------------------

@torch.no_grad()
def estimate_loss(model, train_data, val_data, batch_size, seq_len, device, eval_iters=20):
    """Estimate train and val loss by averaging over several batches."""
    model.eval()
    results = {}
    for name, data in [("train", train_data), ("val", val_data)]:
        losses = []
        for _ in range(eval_iters):
            x, y = get_batch(data, batch_size, seq_len, device)
            _, loss = model(x, y)
            losses.append(loss.item())
        results[name] = sum(losses) / len(losses)
    model.train()
    return results


# ---------------------------------------------------------------------------
# Sanity check: overfit one batch
# ---------------------------------------------------------------------------
# Before doing a full training run, verify the model can memorize a single
# batch. If it can't drive the loss to near zero, something is fundamentally
# wrong (bug in model, data, or training loop).

def sanity_check(config, device):
    """Overfit a single batch — loss should reach ~0."""
    print("=" * 60)
    print("SANITY CHECK: overfitting one batch")
    print("=" * 60)

    model = GPT(config).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=3e-4)

    # Get one fixed batch
    train_data, _ = prepare_data()
    x, y = get_batch(train_data, batch_size=4, seq_len=config.seq_len, device=device)

    for step in range(500):
        _, loss = model(x, y)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        if step % 50 == 0:
            print(f"  step {step:4d} | loss {loss.item():.4f}")

    final_loss = loss.item()
    if final_loss < 0.1:
        print(f"\nPASSED: loss reached {final_loss:.4f} (< 0.1)")
    else:
        print(f"\nFAILED: loss is {final_loss:.4f} (should be < 0.1)")
    print()
    return final_loss < 0.1


# ---------------------------------------------------------------------------
# Training loop
# ---------------------------------------------------------------------------

def train(config, device, max_steps=2000, batch_size=32, max_lr=3e-3, min_lr=3e-4,
          warmup_steps=100, eval_interval=100, save_interval=500, checkpoint_dir="checkpoints"):
    """
    Full training loop with LR scheduling, evaluation, and checkpointing.
    """
    print("=" * 60)
    print("TRAINING")
    print("=" * 60)
    print(f"Device: {device}")
    print(f"Config: n_embd={config.n_embd}, n_head={config.n_head}, n_layer={config.n_layer}")
    print(f"Steps: {max_steps}, Batch size: {batch_size}, Seq len: {config.seq_len}")
    print(f"LR: {max_lr} -> {min_lr} (warmup {warmup_steps} steps)")
    print()

    # Prepare data and model
    train_data, val_data = prepare_data()
    model = GPT(config).to(device)
    total_params = sum(p.numel() for p in model.parameters())
    print(f"Parameters: {total_params:,}")

    # AdamW optimizer
    # Weight decay penalizes large weights (regularization), but we don't
    # apply it to biases or normalization parameters (they don't need it).
    decay_params = [p for n, p in model.named_parameters() if p.dim() >= 2]
    nodecay_params = [p for n, p in model.named_parameters() if p.dim() < 2]
    optimizer = torch.optim.AdamW([
        {"params": decay_params, "weight_decay": 0.1},
        {"params": nodecay_params, "weight_decay": 0.0},
    ], lr=max_lr, betas=(0.9, 0.95))

    print(f"Optimizer groups: {len(decay_params)} decay, {len(nodecay_params)} no-decay")
    print()

    # Use mixed precision on GPU (faster, less memory)
    use_amp = device in ("cuda", "mps")
    scaler = torch.amp.GradScaler(enabled=(device == "cuda"))
    amp_dtype = torch.bfloat16 if device == "cuda" and torch.cuda.is_bf16_supported() else torch.float16

    os.makedirs(checkpoint_dir, exist_ok=True)
    best_val_loss = float("inf")
    t0 = time.time()

    for step in range(max_steps):
        # Update learning rate
        lr = get_lr(step, max_lr, min_lr, warmup_steps, max_steps)
        for param_group in optimizer.param_groups:
            param_group["lr"] = lr

        # Forward + backward
        x, y = get_batch(train_data, batch_size, config.seq_len, device)

        if use_amp:
            with torch.autocast(device_type=device, dtype=amp_dtype):
                _, loss = model(x, y)
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            scaler.step(optimizer)
            scaler.update()
        else:
            _, loss = model(x, y)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()

        optimizer.zero_grad(set_to_none=True)

        # Logging
        if step % 10 == 0:
            elapsed = time.time() - t0
            tokens_per_sec = (step + 1) * batch_size * config.seq_len / elapsed if elapsed > 0 else 0
            print(f"step {step:5d} | loss {loss.item():.4f} | lr {lr:.2e} | {tokens_per_sec:.0f} tok/s")

        # Evaluation
        if step > 0 and step % eval_interval == 0:
            losses = estimate_loss(model, train_data, val_data, batch_size, config.seq_len, device)
            print(f"  >>> eval | train {losses['train']:.4f} | val {losses['val']:.4f}")

            if losses["val"] < best_val_loss:
                best_val_loss = losses["val"]
                save_checkpoint(model, config, optimizer, step, os.path.join(checkpoint_dir, "best.pt"))
                print(f"  >>> new best val loss, saved checkpoint")

        # Periodic save
        if step > 0 and step % save_interval == 0:
            save_checkpoint(model, config, optimizer, step, os.path.join(checkpoint_dir, "latest.pt"))

    # Final save
    save_checkpoint(model, config, optimizer, max_steps, os.path.join(checkpoint_dir, "final.pt"))
    print(f"\nTraining complete. Best val loss: {best_val_loss:.4f}")

    # Generate a sample to see how we're doing
    print("\n--- Sample generation ---")
    model.eval()
    prompt = torch.zeros((1, 1), dtype=torch.long, device=device)  # start with token 0
    with torch.no_grad():
        for _ in range(200):
            # Only use the last seq_len tokens if the sequence is too long
            idx_cond = prompt[:, -config.seq_len:]
            logits, _ = model(idx_cond)
            # Greedy: take the most likely next token
            next_token = logits[:, -1, :].argmax(dim=-1, keepdim=True)
            prompt = torch.cat([prompt, next_token], dim=1)
    text = decode(prompt[0].tolist())
    print(text)


# ---------------------------------------------------------------------------
# Checkpointing
# ---------------------------------------------------------------------------

def save_checkpoint(model, config, optimizer, step, path):
    """Save model, optimizer, and config to a file."""
    torch.save({
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "config": config,
        "step": step,
    }, path)

def load_checkpoint(path, device="cpu"):
    """Load a checkpoint and return (model, config, step)."""
    checkpoint = torch.load(path, map_location=device, weights_only=False)
    config = checkpoint["config"]
    model = GPT(config).to(device)
    model.load_state_dict(checkpoint["model_state_dict"])
    return model, config, checkpoint["step"]


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def get_device():
    """Auto-detect the best available device."""
    if torch.cuda.is_available():
        return "cuda"
    elif torch.backends.mps.is_available():
        return "mps"
    return "cpu"

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train a simple transformer")
    parser.add_argument("--sanity-check", action="store_true", help="Overfit one batch and exit")
    parser.add_argument("--n-embd", type=int, default=64, help="Embedding dimension")
    parser.add_argument("--n-head", type=int, default=4, help="Number of attention heads")
    parser.add_argument("--n-layer", type=int, default=4, help="Number of transformer blocks")
    parser.add_argument("--seq-len", type=int, default=256, help="Sequence length")
    parser.add_argument("--max-steps", type=int, default=2000, help="Training steps")
    parser.add_argument("--batch-size", type=int, default=32, help="Batch size")
    parser.add_argument("--max-lr", type=float, default=3e-3, help="Peak learning rate")
    args = parser.parse_args()

    device = get_device()
    config = GPTConfig(
        n_embd=args.n_embd,
        n_head=args.n_head,
        n_layer=args.n_layer,
        seq_len=args.seq_len,
    )

    if args.sanity_check:
        sanity_check(config, device)
    else:
        train(config, device, max_steps=args.max_steps, batch_size=args.batch_size, max_lr=args.max_lr)
