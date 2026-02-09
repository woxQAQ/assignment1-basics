import cProfile
import os
import regex as re
from collections import Counter

PAT = re.compile(
    r"""'s|'t|'re|'ve|'m|'ll|'d| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+"""
)


def pre_tokenize(
    text_block: str, special_tokens: list[str] | None = None
) -> Counter[tuple[bytes, ...]]:
    c: Counter[tuple[bytes, ...]] = Counter()

    # Split by special tokens if provided
    if special_tokens:
        # Longest-first alternation avoids partial matches for overlapping specials.
        escaped_special_tokens = sorted(
            (re.escape(t) for t in special_tokens), key=len, reverse=True
        )
        pattern = "|".join(escaped_special_tokens)
        parts = [block for block in re.split(pattern, text_block) if block]
    else:
        parts = [text_block]

    # Pre-tokenize each part separately
    for part in parts:
        for token in PAT.finditer(part):
            token_bytes = token.group().encode("utf-8")
            c[tuple(bytes([b]) for b in token_bytes)] += 1
    return c
