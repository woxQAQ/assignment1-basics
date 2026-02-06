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


def bpe_train(f: BinaryIO, merge_times: int = 6, special_tokens: str | None = None) -> list[bytes]:
    # step 0: load vocabulary
    num_processes = 4
    boundaries = find_chunk_boundaries(f, num_processes, b"<|endoftext|>")

    # step 1: pre-tokenization (parallelized)
    chunks = []
    for start, end in zip(boundaries[:-1], boundaries[1:]):
        f.seek(start)
        chunk = f.read(end - start).decode("utf-8", errors="ignore")
        chunks.append(chunk)

    with multiprocessing.Pool(processes=num_processes) as pool:
        pre_tokenize_with_special = partial(pre_tokenize, special_tokens=special_tokens)
        results = pool.map(pre_tokenize_with_special, chunks)
    c = sum(results, Counter())
    voc = [b"<|endoftext|>"] + [bytes([i]) for i in range(256)]
    # step2: merge
    for i in range(0, merge_times):
        temp: dict[tuple[bytes, ...], int] = {}
        merge_rount = 6
        for token, count in c.items():
            prev_b = b""
            for b in token:
                if prev_b != b"":
                    temp[prev_b, b] = temp.get((prev_b, b), 0) + count
                prev_b = b
        max_count = max(temp.values())
        max_byte_pair = max([key for key, value in temp.items() if value == max_count])
        first, second = max_byte_pair
        # print((first+second).decode())
        voc.append(first + second)
        # print(first,second)

        result = Counter()

        for token, count in c.items():
            prev = b""
            token_list = list(token)
            for i, b in enumerate(token_list):
                # (a, b, c, d)
                #     i
                # => (ab, c, d)
                if prev == first and b == second:
                    token_list[i - 1] += b
                    del token_list[i]
                prev = b
            result[tuple(token_list)] += count
        c = result
    return voc


if __name__ == "__main__":
    text = """low low low low low
lower lower widest widest widest
newest newest newest newest newest newest
"""
    with open("tests/fixtures/tinystories_sample.txt", "rb") as f:
        voc = bpe_train(f, 100, "<|endoftext|>")
        print(voc)
