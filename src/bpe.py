from src.pre_tokenizer import pre_tokenize
import time
import os
from collections import Counter
from typing import BinaryIO
import regex as re
import multiprocessing
from functools import partial
import cProfile

# Global variable for profile queue, initialized via pool initializer
_profile_queue: "multiprocessing.Queue | None" = None


def _init_worker(queue: "multiprocessing.Queue | None") -> None:
    """Initialize worker process with shared queue for profile data."""
    global _profile_queue
    _profile_queue = queue


def _tokenize_chunk_with_index(
    index_chunk: tuple[int, str],
    special_tokens: list[str] | None,
) -> Counter[tuple[bytes, ...]]:
    """
    Worker function that processes a chunk with its index.
    Only index 0 will be profiled if profile_queue is provided.

    Args:
        index_chunk: Tuple of (chunk_index, chunk_data)
        special_tokens: List of special tokens

    Returns:
        Counter of token frequencies
    """
    global _profile_queue
    index, chunk = index_chunk

    # Only profile the first chunk (index 0)
    if index == 0 and _profile_queue is not None:
        profiler = cProfile.Profile()
        profiler.enable()
        result = pre_tokenize(chunk, special_tokens)
        profiler.disable()

        # Dump to a temporary file, then notify main process
        temp_prof_file = f".profile_worker_{index}_{os.getpid()}.prof"
        profiler.dump_stats(temp_prof_file)
        _profile_queue.put((index, temp_prof_file))

        return result
    else:
        return pre_tokenize(chunk, special_tokens)


def pre_tokenize_file(
    input_path: str,
    special_tokens: list[str] | None = None,
    parallel: bool = True,
    num_processes: int = 8,
    profile_workers: bool = False,
) -> Counter[tuple[bytes, ...]]:
    """
    Pre-tokenize a text file and return word frequencies.

    Args:
        input_path: Path to the training data text file.
        special_tokens: List of special tokens to preserve during tokenization.
        parallel: Whether to use parallel processing.
        num_processes: Number of processes for parallel processing.
        profile_workers: Whether to profile the first worker (saves to pre_tokenize_worker_0.prof).

    Returns:
        Counter mapping token tuples to their frequencies.
    """
    if special_tokens is None:
        special_tokens = []

    with open(input_path, "rb") as f:
        split_special_token = (
            special_tokens[0].encode("utf-8") if special_tokens else b""
        )
        boundaries = find_chunk_boundaries(f, num_processes, split_special_token)

        # Read and split into chunks
        chunks = []
        pre_tokenize_start = time.time()
        for start, end in zip(boundaries[:-1], boundaries[1:]):
            f.seek(start)
            chunk = f.read(end - start).decode("utf-8", errors="ignore")
            chunks.append(chunk)

        # Enumerate chunks to track index for profiling
        indexed_chunks = list(enumerate(chunks))

        if not parallel:
            if profile_workers:
                profiler = cProfile.Profile()
                profiler.enable()
                results = [pre_tokenize(chunk, special_tokens) for chunk in chunks]
                profiler.disable()
                profiler.dump_stats("pre_tokenize_main.prof")
                print(f"Profile saved to pre_tokenize_main.prof")
            else:
                results = [pre_tokenize(chunk, special_tokens) for chunk in chunks]
        else:
            if profile_workers:
                # Create a queue for collecting profile file paths from workers
                profile_queue: "multiprocessing.Queue" = multiprocessing.Queue()

                # Use Pool with initializer to share queue with workers
                with multiprocessing.Pool(
                    processes=num_processes,
                    initializer=_init_worker,
                    initargs=(profile_queue,),
                ) as pool:
                    worker_func = partial(
                        _tokenize_chunk_with_index,
                        special_tokens=special_tokens,
                    )
                    results = pool.map(worker_func, indexed_chunks)

                # After all workers complete, retrieve profile file path from queue
                try:
                    worker_index, temp_prof_file = profile_queue.get(timeout=5.0)
                    # Move the temp file to the final destination
                    if os.path.exists(temp_prof_file):
                        import shutil

                        shutil.move(temp_prof_file, "pre_tokenize_worker_0.prof")
                        print(f"Profile saved to pre_tokenize_worker_0.prof")
                except Exception as e:
                    print(f"Warning: Could not retrieve profile data: {e}")
            else:
                # No profiling - simple parallel map
                worker_func = partial(
                    _tokenize_chunk_with_index,
                    special_tokens=special_tokens,
                )
                with multiprocessing.Pool(
                    processes=num_processes,
                    initializer=_init_worker,
                    initargs=(None,),
                ) as pool:
                    results = pool.map(worker_func, indexed_chunks)

        word_freq = sum(results, Counter())
        pre_tokenize_end = time.time()
        print(
            f"Time taken for pre_token {(pre_tokenize_end - pre_tokenize_start) * 1000:.2f} milliseconds"
        )

    return word_freq


def bpe_train(
    input_path: str,
    vocab_size: int,
    special_tokens: list[str] | None = None,
    parallel: bool = True,
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

    # step 1: pre-tokenization (parallelized)
    word_freq = pre_tokenize_file(input_path, special_tokens, parallel)

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
        if not bp_counts:
            break
        max_count = max(bp_counts.values())
        max_byte_pair = max(
            [
                byte_pair
                for byte_pair, count in bp_counts.items()
                if count == max_count
            ]
        )
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
            seq_len = len(bytes_seq)
            while i < seq_len - 1:
                if (bytes_seq[i], bytes_seq[i + 1]) == max_byte_pair:
                    left = bytes_seq[i - 1] if i - 1 >= 0 else None
                    right = bytes_seq[i + 2] if i + 2 < seq_len else None
                    if left is not None:
                        bp_counts[(left, first)] -= freq
                        if bp_counts[(left, first)] == 0:
                            del bp_counts[(left, first)]
                    if right is not None:
                        bp_counts[(second, right)] -= freq
                        if bp_counts[(second, right)] == 0:
                            del bp_counts[(second, right)]
                    bytes_seq[i : i + 2] = [first + second]
                    seq_len -= 1
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
                pair_count_map[prev_b, b] = (
                    pair_count_map.get((prev_b, b), 0) + count
                )
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
    assert isinstance(split_special_token, bytes), (
        "Must represent special token as a bytestring"
    )

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
