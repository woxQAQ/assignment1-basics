import time
import os
from collections import Counter
from typing import BinaryIO
import regex as re
import multiprocessing
from functools import partial

PAT = re.compile(r"""'s|'t|'re|'ve|'m|'ll|'d| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+""")

def pre_tokenize(text_block: str, special_tokens: list[str] | None = None) -> Counter[tuple[bytes, ...]]:
    c: Counter[tuple[bytes, ...]] = Counter()

    # Split by special tokens if provided
    if special_tokens:
        # Longest-first alternation avoids partial matches for overlapping specials.
        escaped_special_tokens = sorted((re.escape(t) for t in special_tokens), key=len, reverse=True)
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


def bpe_train(
    input_path: str, vocab_size: int, special_tokens: list[str] | None = None, parallel: bool = True
) -> tuple[dict[int, bytes], list[tuple[bytes, bytes]]]:
    """
    Train a BPE tokenizer.

    Args:
        input_path: Path to the training data text file.
        vocab_size: Maximum vocab size (including initial bytes, merges, and special tokens).
        special_tokens: List of special tokens to add to vocab (don't affect BPE training).
        parallel: Whether to use parallel processing for pre-tokenization.

    Returns:
        vocab: Mapping from token ID to bytes.
        merges: List of merge operations in order.
    """
    if special_tokens is None:
        special_tokens = []

    # step 0: load vocabulary
    num_processes = 4
    with open(input_path, "rb") as f:
        split_special_token = special_tokens[0].encode("utf-8") if special_tokens else b""
        boundaries = find_chunk_boundaries(f, num_processes, split_special_token)

        # step 1: pre-tokenization (parallelized)
        chunks = []
        for start, end in zip(boundaries[:-1], boundaries[1:]):
            f.seek(start)
            chunk = f.read(end - start).decode("utf-8", errors="ignore")
            chunks.append(chunk)
        tokenize_chunk = partial(pre_tokenize, special_tokens=special_tokens)
        if not parallel:
            results = [tokenize_chunk(chunk) for chunk in chunks]
        else:
            with multiprocessing.Pool(processes=num_processes) as pool:
                # Preserve special-token boundaries in each worker.
                results = pool.map(tokenize_chunk, chunks)
        word_freq = sum(results, Counter())

    # Initialize vocab with 256 bytes
    voc: list[bytes] = [bytes([i]) for i in range(256)]
    merges: list[tuple[bytes, bytes]] = []

    # Calculate number of merges needed
    # vocab_size includes: 256 initial bytes + merges + special tokens
    num_merges = vocab_size - 256 - len(special_tokens)

    # step2: merge
    start = time.time()
    # step2.1: init merge stats
    bp_counts, word_to_bytes = init_bpe_merge_stats(word_freq)
    for it in range(0, num_merges):
        # step2.2: find max byte pair
        max_count = max(bp_counts.values())
        max_byte_pair = max([byte_pair for byte_pair, count in bp_counts.items() if count == max_count])
        first, second = max_byte_pair
        voc.append(first + second)
        merges.append((first, second))
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

    # Add special tokens to vocab
    for token in special_tokens:
        voc.append(token.encode("utf-8"))

    # Convert list to dict with int IDs
    vocab = {i: token for i, token in enumerate(voc)}

    return vocab, merges


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


def find_chunk_boundaries(
    file: BinaryIO,
    desired_num_chunks: int,
    split_special_token: bytes,
) -> list[int]:
    """
    Chunk the file into parts that can be counted independently.
    May return fewer chunks if the boundaries end up overlapping.
    """
    assert isinstance(split_special_token, bytes), "Must represent special token as a bytestring"

    # Get total file size in bytes
    file.seek(0, os.SEEK_END)
    file_size = file.tell()
    file.seek(0)

    chunk_size = file_size // desired_num_chunks

    # Initial guesses for chunk boundary locations, uniformly spaced
    # Chunks start on previous index, don't include last index
    chunk_boundaries = [i * chunk_size for i in range(desired_num_chunks + 1)]
    chunk_boundaries[-1] = file_size

    # Fall back to uniform chunking when there is no split token.
    if split_special_token == b"":
        return chunk_boundaries

    mini_chunk_size = 4096  # Read ahead by 4k bytes at a time

    for bi in range(1, len(chunk_boundaries) - 1):
        initial_position = chunk_boundaries[bi]
        file.seek(initial_position)  # Start at boundary guess
        while True:
            mini_chunk = file.read(mini_chunk_size)  # Read a mini chunk

            # If EOF, this boundary should be at the end of the file
            if mini_chunk == b"":
                chunk_boundaries[bi] = file_size
                break

            # Find the special token in the mini chunk
            found_at = mini_chunk.find(split_special_token)
            if found_at != -1:
                chunk_boundaries[bi] = initial_position + found_at
                break
            initial_position += mini_chunk_size

    # Make sure all boundaries are unique, but might be fewer than desired_num_chunks
    return sorted(set(chunk_boundaries))


if __name__ == "__main__":
    text = """low low low low low
lower lower widest widest widest
newest newest newest newest newest newest
"""
    # Test the new signature
    vocab, merges = bpe_train("tests/fixtures/tinystories_sample.txt", 300, ["<eos>"])
    print(f"Vocab size: {len(vocab)}")
    print(f"Number of merges: {len(merges)}")
    print(f"First few merges: {merges[:5]}")
