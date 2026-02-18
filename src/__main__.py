from __future__ import annotations
from utils import serialize_vocab, serialize_merges

import argparse
import json
from pathlib import Path

from src.bpe_train import bpe_train


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Train BPE tokenizer and serialize vocab/merges."
    )
    parser.add_argument(
        "--input-path",
        default="data/TinyStoriesV2-GPT4-train.txt",
        help="Training corpus path.",
    )
    parser.add_argument(
        "--vocab-size",
        type=int,
        default=10000,
        help="Final vocab size including bytes, merges and special tokens.",
    )
    parser.add_argument(
        "--special-tokens",
        nargs="*",
        default=["<|endoftext|>"],
        help="Special tokens to append to vocab.",
    )
    parser.add_argument(
        "--vocab-out",
        default="data/tinystories_vocab.json",
        help="Path to save serialized vocab JSON.",
    )
    parser.add_argument(
        "--merges-out",
        default="data/tinystories_merges.txt",
        help="Path to save serialized merges text.",
    )
    parser.add_argument(
        "--no-parallel",
        action="store_true",
        help="Disable parallel pre-tokenization.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    vocab, merges = bpe_train(
        input_path=args.input_path,
        vocab_size=args.vocab_size,
        special_tokens=args.special_tokens,
        parallel=not args.no_parallel,
    )

    vocab_out = Path(args.vocab_out)
    merges_out = Path(args.merges_out)
    serialize_vocab(vocab, vocab_out)
    serialize_merges(merges, merges_out)


if __name__ == "__main__":
    main()
