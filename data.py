"""
Data loading and tokenization for the simple transformer.

Uses a character-level tokenizer — the simplest possible approach.
Each byte (0-255) is its own token, no subword magic needed.
This lets us focus entirely on the model, not the tokenizer.

Dataset: Tiny Shakespeare (~1MB of Shakespeare plays).
"""

import os
import requests
import torch

# ---------------------------------------------------------------------------
# Character-level tokenizer
# ---------------------------------------------------------------------------
# The simplest tokenizer: each character is a token.
# vocab_size = 256 (one for each byte value).
#
# Fancier tokenizers (BPE, SentencePiece) split text into subword pieces
# like "un" + "believ" + "able". We'll add that later — for now, characters
# are enough to get a working model.

def encode(text: str) -> list[int]:
    """Convert a string to a list of byte values (0-255)."""
    return list(text.encode("utf-8"))

def decode(tokens: list[int]) -> str:
    """Convert a list of byte values back to a string."""
    return bytes(tokens).decode("utf-8", errors="replace")


# ---------------------------------------------------------------------------
# Dataset download
# ---------------------------------------------------------------------------

DATA_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "data")
SHAKESPEARE_URL = "https://raw.githubusercontent.com/karpathy/char-rnn/master/data/tinyshakespeare/input.txt"

def download_shakespeare() -> str:
    """Download tiny Shakespeare and return the text."""
    os.makedirs(DATA_DIR, exist_ok=True)
    filepath = os.path.join(DATA_DIR, "shakespeare.txt")

    if not os.path.exists(filepath):
        print(f"Downloading tiny Shakespeare to {filepath}...")
        response = requests.get(SHAKESPEARE_URL)
        response.raise_for_status()
        with open(filepath, "w") as f:
            f.write(response.text)
        print(f"Downloaded {len(response.text):,} characters.")

    with open(filepath, "r") as f:
        return f.read()


# ---------------------------------------------------------------------------
# Batch construction
# ---------------------------------------------------------------------------
# For language modeling, the input is a sequence of tokens and the target
# is the same sequence shifted right by 1. The model learns to predict
# the next token at each position.
#
# Example (seq_len=4):
#   text:    "Hello World"
#   tokens:  [72, 101, 108, 108, 111, 32, 87, 111, 114, 108, 100]
#   input:   [72, 101, 108, 108]    <- "Hell"
#   target:  [101, 108, 108, 111]   <- "ello"
#
# We pick random starting positions in the text to create batches.

def get_batch(data: torch.Tensor, batch_size: int, seq_len: int, device: str = "cpu"):
    """
    Sample a random batch of (input, target) pairs from the data.

    Args:
        data: 1D tensor of token IDs (the entire dataset)
        batch_size: number of sequences per batch
        seq_len: length of each sequence
        device: "cpu", "cuda", or "mps"

    Returns:
        x: input tokens, shape (batch_size, seq_len)
        y: target tokens (shifted by 1), shape (batch_size, seq_len)
    """
    # Pick random starting positions (need room for seq_len + 1 tokens)
    max_start = len(data) - seq_len - 1
    starts = torch.randint(0, max_start, (batch_size,))

    # Extract sequences
    x = torch.stack([data[i : i + seq_len] for i in starts])
    y = torch.stack([data[i + 1 : i + 1 + seq_len] for i in starts])

    return x.to(device), y.to(device)


# ---------------------------------------------------------------------------
# Prepare dataset with train/val split
# ---------------------------------------------------------------------------

def prepare_data(val_fraction: float = 0.1):
    """
    Download Shakespeare, encode it, and split into train/val.

    Returns:
        train_data: 1D tensor of token IDs (90% of data)
        val_data: 1D tensor of token IDs (10% of data)
    """
    text = download_shakespeare()
    tokens = encode(text)
    data = torch.tensor(tokens, dtype=torch.long)

    # Split: first 90% for training, last 10% for validation
    split_idx = int(len(data) * (1 - val_fraction))
    train_data = data[:split_idx]
    val_data = data[split_idx:]

    return train_data, val_data


# ---------------------------------------------------------------------------
# Test — run with: uv run python data.py
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    # Test tokenizer
    text = "Hello, World!"
    tokens = encode(text)
    decoded = decode(tokens)
    assert decoded == text, f"Round-trip failed: {decoded!r} != {text!r}"
    print(f"Tokenizer: '{text}' -> {tokens} -> '{decoded}'")
    print(f"Vocab size: 256 (one per byte)")
    print()

    # Test data loading
    train_data, val_data = prepare_data()
    print(f"Dataset: {len(train_data) + len(val_data):,} total tokens")
    print(f"  Train: {len(train_data):,} tokens")
    print(f"  Val:   {len(val_data):,} tokens")
    print()

    # Test batch construction
    x, y = get_batch(train_data, batch_size=4, seq_len=32)
    print(f"Batch: x={x.shape}, y={y.shape}")
    print(f"  First input:  '{decode(x[0].tolist())}'")
    print(f"  First target: '{decode(y[0].tolist())}'")
    # Verify target is input shifted by 1
    assert torch.equal(x[0, 1:], y[0, :-1]), "Target should be input shifted right by 1"
    print(f"  Shift-by-1 verified: y is x shifted right")
    print()
    print("All data tests passed!")
