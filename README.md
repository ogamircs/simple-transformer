# Simple Transformer

A transformer built from scratch in PyTorch, inspired by [karpathy/nanochat](https://github.com/karpathy/nanochat). Designed for learning — read the code top-to-bottom, each component is explained inline.

## Setup

```bash
uv sync
```

## Run

```bash
# Test model components
uv run python model.py

# Test data loading + tokenizer
uv run python data.py

# Sanity check: overfit one batch (should reach loss < 0.1)
uv run python train.py --sanity-check

# Train (default: 2000 steps, ~5 min on GPU)
uv run python train.py

# Train with custom settings
uv run python train.py --n-embd 128 --n-head 8 --n-layer 6 --max-steps 5000

# Generate text from a trained model
uv run python generate.py --prompt "ROMEO:" --temperature 0.8 --top-k 50
uv run python generate.py --temperature 0 --max-tokens 500  # greedy
```

## Project Structure

```
model.py      # GPT model: RMSNorm, CausalSelfAttention, MLP, Block, GPT
data.py       # Tiny Shakespeare download, character-level tokenizer, batching
train.py      # Training loop with LR scheduling, eval, checkpoints
generate.py   # Text generation with temperature + top-k sampling
```
