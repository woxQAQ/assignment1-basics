import io
import time
from collections import Counter
from io import TextIOWrapper
from typing import BinaryIO
import regex as re
import multiprocessing
from functools import partial

from cs336_basics.pretokenization_example import find_chunk_boundaries


class BPE:
    def __init__(self) -> None:
        self.pat = re.compile(r"""'(?:[sdmt]|ll|ve|re)| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+""")


PAT = re.compile(r"""'s|'t|'re|'ve|'m|'ll|'d| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+""")


def pre_tokenize(text_block: str, special_tokens: str | None = None) -> Counter[tuple[bytes, ...]]:
    c: Counter[tuple[bytes, ...]] = Counter()

    # Split by special tokens if provided
    if special_tokens:
        # Escape special tokens and join with |
        pattern = "|".join([re.escape(special_tokens)])
        parts = [block for block in re.split(pattern, text_block) if not (block.isspace() or not block)]
    else:
        parts = [text_block]

    # Pre-tokenize each part separately
    for part in parts:
        for token in PAT.finditer(part):
            c[tuple(ch.encode("utf-8") for ch in token.group())] += 1
    return c


def bpe_train(
    f: BinaryIO, merge_times: int = 6, special_tokens: str | None = None, parallel: bool = True
) -> list[bytes]:
    # step 0: load vocabulary
    num_processes = 4
    boundaries = find_chunk_boundaries(f, num_processes, b"<|endoftext|>")

    # step 1: pre-tokenization (parallelized)
    chunks = []
    for start, end in zip(boundaries[:-1], boundaries[1:]):
        f.seek(start)
        chunk = f.read(end - start).decode("utf-8", errors="ignore")
        chunks.append(chunk)
    if not parallel:
        results = [pre_tokenize(chunk, special_tokens=special_tokens) for chunk in chunks]
    else:
        with multiprocessing.Pool(processes=num_processes) as pool:
            pre_tokenize_with_special = partial(pre_tokenize, special_tokens=special_tokens)
            results: list[Counter[tuple[bytes, ...]]] = pool.map(pre_tokenize_with_special, chunks)
    results = [pre_tokenize(chunks[0], special_tokens=special_tokens)]
    word_freq = sum(results, Counter())
    voc = [b"<|endoftext|>"] + [bytes([i]) for i in range(256)]

    # step2: merge
    start = time.time()
    # step2.1: init merge stats
    bp_counts, word_to_bytes = init_bpe_merge_stats(word_freq)
    for it in range(0, merge_times):
        # step2.2: find max byte pair
        max_count = max(bp_counts.values())
        max_byte_pair = max([byte_pair for byte_pair, count in bp_counts.items() if count == max_count])
        first, second = max_byte_pair
        voc.append(first + second)
        # 2.3 merge back to word freq
        result = Counter()
        # iterate over the word freq and found the max bp and merge them
        del bp_counts[max_byte_pair]
        for word, freq in word_freq.items():
            bytes_seq = word_to_bytes[word]
            i = 0
            while i < len(bytes_seq) - 1:
                if (bytes_seq[i], bytes_seq[i + 1]) == max_byte_pair:
                    left = bytes_seq[i - 1] if i - 1 >= 0 else None
                    right = bytes_seq[i + 2] if i + 2 < len(bytes_seq) else None
                    if left is not None:
                        bp_counts[(left, first)] -= freq
                        if bp_counts[(left, first)] == 0:
                            del bp_counts[(left, first)]
                    if right is not None:
                        bp_counts[(second, right)] -= freq
                        if bp_counts[(second, right)] == 0:
                            del bp_counts[(second, right)]
                    bytes_seq[i : i + 2] = [first + second]
                    if left is not None:
                        bp_counts[(left, first + second)] += freq
                    if right is not None:
                        bp_counts[(first + second, right)] += freq
                else:
                    i += 1
        end = time.time()
    print(f"Time taken for merge {(end - start) * 1000:.2f} milliseconds")
    return voc


def init_bpe_merge_stats(
    word_freq: Counter[tuple[bytes, ...]],
) -> tuple[Counter[tuple[bytes, ...]], dict[tuple[bytes, ...], list[bytes]]]:
    pair_count_map: Counter[tuple[bytes, ...]] = Counter()
    word_to_bytes: dict[tuple[bytes, ...], list[bytes]] = {}
    for word, count in word_freq.items():
        prev_b = b""
        word_to_bytes[word] = list(word)
        for b in word:
            if prev_b != b"":
                pair_count_map[prev_b, b] = pair_count_map.get((prev_b, b), 0) + count
            prev_b = b
    return pair_count_map, word_to_bytes


if __name__ == "__main__":
    text = """low low low low low
lower lower widest widest widest
newest newest newest newest newest newest
"""
    # voc = bpe_train(io.BytesIO(text.encode()), special_tokens="<|endoftext|>")
    # print(voc)
    with open("tests/fixtures/tinystories_sample.txt", "rb") as f:
        voc = bpe_train(f, 100, "<|endoftext|>")
        print(voc)
