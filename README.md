# Simple Transformer

A transformer built from scratch in PyTorch, inspired by [karpathy/nanochat](https://github.com/karpathy/nanochat). Designed for learning — read the code top-to-bottom, each component is explained inline.

## Setup

```bash
uv sync
```

## Run

```bash
# Run component tests (Phase 1: RMSNorm, Attention, MLP)
uv run python model.py
```

## Project Structure

```
model.py          # All model components (embeddings, attention, MLP, transformer)
pyproject.toml    # Project config and dependencies
```

More files will be added as we build through the phases:
- `data.py` — dataset download + tokenizer
- `train.py` — training loop
- `generate.py` — text generation / inference
