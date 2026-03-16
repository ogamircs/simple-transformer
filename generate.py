"""
Text generation with the trained transformer.

Concepts covered:
  - Autoregressive generation (feed output back as input)
  - Temperature scaling (controls randomness)
  - Top-k sampling (restricts candidate pool)

Usage:
  uv run python generate.py --prompt "To be or not"
  uv run python generate.py --prompt "ROMEO:" --temperature 0.8 --top-k 50
  uv run python generate.py --temperature 0 --max-tokens 500   # greedy, deterministic
"""

import argparse

import torch

from data import encode, decode
from train import load_checkpoint, get_device


# ---------------------------------------------------------------------------
# Generation
# ---------------------------------------------------------------------------
# Autoregressive generation: we predict one token at a time, append it
# to the sequence, and repeat.
#
# Temperature controls how "creative" vs "conservative" the model is:
#   - temp=0: greedy (always pick the most likely token) — deterministic
#   - temp=0.5: fairly conservative, mostly coherent
#   - temp=1.0: sample from the model's distribution as-is
#   - temp=2.0: very random, often incoherent
#
# Temperature works by dividing logits before softmax:
#   probs = softmax(logits / temperature)
# Lower temp → sharper distribution → less randomness
# Higher temp → flatter distribution → more randomness
#
# Top-k restricts sampling to only the k most likely tokens.
# This prevents the model from picking extremely unlikely tokens
# that would derail the generation.

@torch.no_grad()
def generate(model, prompt_tokens: list[int], max_new_tokens: int = 200,
             temperature: float = 0.8, top_k: int = 50, device: str = "cpu") -> list[int]:
    """
    Generate tokens autoregressively.

    Args:
        model: trained GPT model
        prompt_tokens: list of token IDs to start from
        max_new_tokens: how many tokens to generate
        temperature: sampling temperature (0 = greedy)
        top_k: restrict to top-k most likely tokens (0 = no restriction)
        device: "cpu", "cuda", or "mps"

    Returns:
        List of all token IDs (prompt + generated)
    """
    model.eval()
    seq_len = model.config.seq_len

    # Start with prompt tokens
    tokens = torch.tensor([prompt_tokens], dtype=torch.long, device=device)

    for _ in range(max_new_tokens):
        # Crop to last seq_len tokens if the sequence is too long
        idx = tokens[:, -seq_len:]

        # Forward pass — only care about the last position's predictions
        logits, _ = model(idx)
        logits = logits[:, -1, :]  # (1, vocab_size)

        if temperature == 0:
            # Greedy decoding: always pick the most likely token
            next_token = logits.argmax(dim=-1, keepdim=True)
        else:
            # Scale logits by temperature
            logits = logits / temperature

            # Top-k: zero out everything except the k highest logits
            if top_k > 0:
                top_values, _ = torch.topk(logits, min(top_k, logits.size(-1)))
                # Set everything below the k-th highest value to -inf
                logits[logits < top_values[:, -1:]] = float("-inf")

            # Convert to probabilities and sample
            probs = torch.softmax(logits, dim=-1)
            next_token = torch.multinomial(probs, num_samples=1)

        # Append to sequence
        tokens = torch.cat([tokens, next_token], dim=1)

    return tokens[0].tolist()


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Generate text with a trained transformer")
    parser.add_argument("--checkpoint", type=str, default="checkpoints/best.pt",
                        help="Path to model checkpoint")
    parser.add_argument("--prompt", type=str, default="",
                        help="Text prompt to start generation")
    parser.add_argument("--max-tokens", type=int, default=500,
                        help="Number of tokens to generate")
    parser.add_argument("--temperature", type=float, default=0.8,
                        help="Sampling temperature (0=greedy, 1.0=model distribution)")
    parser.add_argument("--top-k", type=int, default=50,
                        help="Top-k sampling (0=no restriction)")
    args = parser.parse_args()

    device = get_device()

    # Load trained model
    print(f"Loading checkpoint from {args.checkpoint}...")
    model, config, step = load_checkpoint(args.checkpoint, device)
    print(f"Model loaded (step {step}, {sum(p.numel() for p in model.parameters()):,} params)")
    print(f"Config: n_embd={config.n_embd}, n_head={config.n_head}, n_layer={config.n_layer}")
    print(f"Generating with temperature={args.temperature}, top_k={args.top_k}")
    print("-" * 60)

    # Encode prompt (empty prompt = start from scratch)
    prompt_tokens = encode(args.prompt) if args.prompt else [0]

    # Generate
    output_tokens = generate(
        model,
        prompt_tokens,
        max_new_tokens=args.max_tokens,
        temperature=args.temperature,
        top_k=args.top_k,
        device=device,
    )

    # Decode and print
    text = decode(output_tokens)
    print(text)
