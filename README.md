# Simple Transformer

`simple-transformer` is a small, readable GPT-style language model built in PyTorch for learning purposes. The code is written to be read top-to-bottom, with inline explanations for each major part of the system: tokenization, self-attention, transformer blocks, training, checkpointing, and autoregressive generation.

The project trains a byte-level model on Tiny Shakespeare and includes both script-based workflows and a notebook version for interactive exploration.

## What Is Included

- A compact GPT implementation with RMSNorm, causal self-attention, ReLU squared MLPs, and pre-norm residual blocks
- A byte-level tokenizer and Tiny Shakespeare data download pipeline
- A training loop with AdamW, warmup + cosine learning rate scheduling, evaluation, and checkpoint saving
- A text generation script with temperature and top-k sampling
- A notebook walkthrough in `notebook/simple_transformer.ipynb`

## Project Layout

```text
model.py                         GPT model implementation and model self-test
data.py                          Dataset download, byte tokenizer, and batching
train.py                         Training loop, evaluation, checkpoint save/load
generate.py                      Text generation from a saved checkpoint
pyproject.toml                   Project metadata and dependencies
notebook/simple_transformer.ipynb  Notebook version of the project
```

## Requirements

- Python 3.12+
- `uv` for dependency management

## Setup

Install dependencies:

```bash
uv sync
```

## Quick Start

Run the built-in checks:

```bash
uv run python model.py
uv run python data.py
```

Run the one-batch sanity check:

```bash
uv run python train.py --sanity-check
```

Train the model with default settings:

```bash
uv run python train.py
```

Train with custom hyperparameters:

```bash
uv run python train.py --n-embd 128 --n-head 8 --n-layer 6 --seq-len 256 --max-steps 5000 --batch-size 32 --max-lr 3e-3
```

Generate text from a checkpoint:

```bash
uv run python generate.py --prompt "ROMEO:" --temperature 0.8 --top-k 50
uv run python generate.py --temperature 0 --max-tokens 500
```

## Workflow

1. `data.py` downloads Tiny Shakespeare and converts text into byte tokens.
2. `model.py` defines the GPT architecture and a small self-test for tensor shapes and gradients.
3. `train.py` prepares batches, trains the model, evaluates periodically, and saves checkpoints.
4. `generate.py` loads a checkpoint and samples continuation text from a prompt.

By default, downloaded data is written to `data/` and checkpoints are written to `checkpoints/`.

## Notebook

If you prefer learning interactively, open [`notebook/simple_transformer.ipynb`](notebook/simple_transformer.ipynb). It mirrors the main scripts in a step-by-step notebook format.

## Inspiration

This project is inspired by [karpathy/nanochat](https://github.com/karpathy/nanochat) and follows a similarly educational, minimal style.

## Support

If this project helps you, you can support it here:

[Buy Me a Coffee](https://buymeacoffee.com/ogamircs)
