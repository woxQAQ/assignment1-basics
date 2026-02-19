from functools import lru_cache
import regex as re
from collections import Counter

PAT = re.compile(
    r"""'s|'t|'re|'ve|'m|'ll|'d| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+"""
)

# Pre-compute cache of single-byte tuples for performance
# This avoids creating 300M+ temporary bytes([b]) objects
CACHE = tuple(bytes([i]) for i in range(256))


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
    str_counter = Counter()
    for part in parts:
        str_counter.update(PAT.findall(part))
    for token_str, count in str_counter.items():
        c[_str_to_tuple(token_str)] = count
    return c


@lru_cache(maxsize=100000)
def _str_to_tuple(b: str) -> tuple[bytes, ...]:
    return tuple(CACHE[i] for i in b.encode("utf-8"))
