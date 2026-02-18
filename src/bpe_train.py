import heapq
from src.pre_tokenizer import pre_tokenize
import time
import os
from collections import Counter, defaultdict
from typing import BinaryIO
import regex as re
import multiprocessing
from functools import partial, total_ordering
import cProfile

# Global variable for profile queue, initialized via pool initializer
_profile_queue: "multiprocessing.Queue | None" = None


def _init_worker(queue: "multiprocessing.Queue | None") -> None:
    """Initialize worker process with shared queue for profile data."""
    global _profile_queue
    _profile_queue = queue


def _read_chunk(input_path: str, start: int, end: int) -> str:
    """Read and decode a file slice [start, end)."""
    with open(input_path, "rb") as f:
        f.seek(start)
        return f.read(end - start).decode("utf-8", errors="ignore")


def _tokenize_range_with_index(
    index_range: tuple[int, int, int],
    input_path: str,
    special_tokens: list[str] | None,
) -> Counter[tuple[bytes, ...]]:
    """
    Worker function that processes a byte-range with its index.
    Only index 0 will be profiled if profile_queue is provided.

    Args:
        index_range: Tuple of (chunk_index, start, end)
        input_path: Training data path
        special_tokens: List of special tokens

    Returns:
        Counter of token frequencies
    """
    global _profile_queue
    index, start, end = index_range
    chunk = _read_chunk(input_path, start, end)

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
        boundaries = find_chunk_boundaries(
            f, num_processes, split_special_token
        )

    indexed_ranges = [
        (idx, start, end)
        for idx, (start, end) in enumerate(zip(boundaries[:-1], boundaries[1:]))
    ]

    pre_tokenize_start = time.time()
    word_freq = Counter()
    if not parallel:
        if profile_workers:
            profiler = cProfile.Profile()
            profiler.enable()
            for _, start, end in indexed_ranges:
                word_freq.update(
                    pre_tokenize(
                        _read_chunk(input_path, start, end), special_tokens
                    )
                )
            profiler.disable()
            profiler.dump_stats("pre_tokenize_main.prof")
            print(f"Profile saved to pre_tokenize_main.prof")
        else:
            for _, start, end in indexed_ranges:
                word_freq.update(
                    pre_tokenize(
                        _read_chunk(input_path, start, end), special_tokens
                    )
                )
    else:
        worker_func = partial(
            _tokenize_range_with_index,
            input_path=input_path,
            special_tokens=special_tokens,
        )
        if profile_workers:
            # Create a queue for collecting profile file paths from workers
            profile_queue: "multiprocessing.Queue" = multiprocessing.Queue()

            # Use Pool with initializer to share queue with workers
            with multiprocessing.Pool(
                processes=num_processes,
                initializer=_init_worker,
                initargs=(profile_queue,),
            ) as pool:
                for result in pool.imap_unordered(
                    worker_func, indexed_ranges, chunksize=1
                ):
                    word_freq.update(result)

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
            # No profiling - consume results as workers finish
            with multiprocessing.Pool(
                processes=num_processes,
                initializer=_init_worker,
                initargs=(None,),
            ) as pool:
                for result in pool.imap_unordered(
                    worker_func, indexed_ranges, chunksize=1
                ):
                    word_freq.update(result)
    pre_tokenize_end = time.time()
    print(
        f"Time taken for pre_token {(pre_tokenize_end - pre_tokenize_start) * 1000:.2f} milliseconds"
    )

    return word_freq


@total_ordering
class RevPair:
    __slots__ = ("pair",)

    def __init__(self, pair):
        self.pair = pair

    def __lt__(self, other):
        return self.pair > other.pair

    def __eq__(self, other):
        return self.pair == other.pair


def push_pair(heap, bp_counts, pair):
    cnt = bp_counts.get(pair)
    if cnt and cnt > 0:
        heapq.heappush(heap, (-cnt, RevPair(pair), pair))


def pop_best_pair(heap, bp_counts):
    while heap:
        neg_cnt, _, pair = heapq.heappop(heap)
        cnt = -neg_cnt
        if bp_counts.get(pair) == cnt:
            return pair, cnt
    return None, 0


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

    # 1: pre-tokenization (parallelized)
    pre_tokenize_results = pre_tokenize_file(
        input_path, special_tokens, parallel
    )

    # Initialize vocab with 256 bytes
    voc: list[bytes] = [bytes([i]) for i in range(256)]
    merges: list[tuple[bytes, bytes]] = []

    # Calculate number of merges needed
    # vocab_size includes: 256 initial bytes + merges + special tokens
    merge_times = vocab_size - 256 - len(special_tokens)

    start = time.time()
    # 2.1: init merge stats
    #
    # bp_counts are the initial byte pair counts map, we use to
    # maintain the byte pair status during whole merge procedure
    #
    # bp_to_words are inverted-index that map byte pairs to words id
    # so that we can only search the affected word after we get max byte pair
    #
    # words_bp_seq are the list of the [real time] byte pair sequence
    # of words, we can get sequence by id quickly.
    #
    # word_freq_list are the immutable list of the frequence of words,
    # we can get freq by id quickly
    bp_counts, bp_to_words, words_bp_sequence, word_freq_list = (
        init_bpe_merge_stats(pre_tokenize_results)
    )
    heap = []
    for bp in bp_counts:
        push_pair(heap, bp_counts, bp)
    end = start
    for it in range(0, merge_times):
        # 2.2: find max byte pair
        max_bp, max_bp_count = pop_best_pair(heap, bp_counts)
        if max_bp is None:
            break
        if max_bp_count <= 0:
            break
        first, second = max_bp
        merged = first + second
        # 2.3 merge back to word freq
        # Only words containing max_byte_pair can change.
        affected = bp_to_words.get(max_bp)
        if not affected:
            bp_counts.pop(max_bp, None)
            bp_to_words.pop(max_bp, None)
            continue

        merged_any = False
        for wid in list(affected):
            old_seq = words_bp_sequence[wid]
            freq = word_freq_list[wid]

            # 2.3.1 get the old word byte pair sequence
            old_bp_seq = Counter(zip(old_seq, old_seq[1:]))

            i = 0
            seq_len = len(old_seq)

            # new_seq are the merged version of byte_seq
            new_seq: list[bytes] = []
            # flag to indicate there is merged happended
            local_merged = False
            # 2.3.2 find all target(merged) pair in the bp sequence
            while i < seq_len:
                if (
                    i + 1 < seq_len
                    and old_seq[i] == first
                    and old_seq[i + 1] == second
                ):
                    new_seq.append(merged)
                    i += 2
                    local_merged = True
                else:
                    new_seq.append(old_seq[i])
                    i += 1
            # no merge happens, fast path
            if not local_merged:
                # If the inverted index entry is stale, remove it.
                stale = bp_to_words.get(max_bp)
                if stale is not None:
                    stale.discard(wid)
                    if not stale:
                        bp_to_words.pop(max_bp, None)
                continue

            merged_any = True
            # new_pairs is the merged version of old_bp_seq
            # after merge
            new_bp_seq = Counter(zip(new_seq, new_seq[1:]))

            # update the bp counter
            for pair, count in old_bp_seq.items():
                new_count = bp_counts.get(pair, 0) - count * freq
                if new_count > 0:
                    bp_counts[pair] = new_count
                    push_pair(heap, bp_counts, pair)
                else:
                    bp_counts.pop(pair, None)

            for pair, count in new_bp_seq.items():
                bp_counts[pair] = bp_counts.get(pair, 0) + count * freq
                push_pair(heap, bp_counts, pair)

            # update inverted indexes
            old_bp_set = set(old_bp_seq)
            new_bp_set = set(new_bp_seq)
            for pair in old_bp_set - new_bp_set:
                words = bp_to_words.get(pair)
                if words is not None:
                    words.discard(wid)
                    if not words:
                        bp_to_words.pop(pair, None)
            for pair in new_bp_set - old_bp_set:
                bp_to_words[pair].add(wid)

            # update word
            words_bp_sequence[wid] = new_seq

        if not merged_any:
            bp_counts.pop(max_bp, None)
            bp_to_words.pop(max_bp, None)
            continue

        voc.append(merged)
        merges.append((first, second))
        if len(heap) > 4 * max(len(bp_counts), 1):
            heap = []
            for bp in bp_counts:
                push_pair(heap, bp_counts, bp)
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
) -> tuple[
    Counter[tuple[bytes, ...]],
    dict[tuple[bytes, ...], set[int]],
    list[list[bytes]],
    list[int],
]:
    bp_counts: Counter[tuple[bytes, ...]] = Counter()
    bp_to_words: dict[tuple[bytes, ...], set[int]] = defaultdict(set)
    words_bp_sequence: list[list[bytes]] = []
    word_freq_list: list[int] = []
    for wid, (word, freq) in enumerate(word_freq.items()):
        bp_sequences = list(word)
        words_bp_sequence.append(bp_sequences)
        word_freq_list.append(freq)
        for i in range(len(bp_sequences) - 1):
            p = (bp_sequences[i], bp_sequences[i + 1])
            bp_counts[p] += freq
            bp_to_words[p].add(wid)
    return bp_counts, bp_to_words, words_bp_sequence, word_freq_list


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
