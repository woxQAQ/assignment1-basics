from src.pre_tokenizer import PAT
from collections.abc import Iterable, Iterator
import regex as re
import json

_BYTE_CACHE = tuple(bytes([i]) for i in range(256))


class Tokenizer:
    __slots__ = ["token_ids", "merge_rank", "vocab", "special_tokens"]

    def __init__(
        self,
        vocab: dict[int, bytes],
        merges: list[tuple[bytes, bytes]],
        special_tokens: list[str] | None = None,
    ):
        self.special_tokens = special_tokens
        self.token_ids: dict[bytes, int] = {}
        self.merge_rank: dict[tuple[bytes, bytes], int] = {}
        self.vocab = vocab
        for id, token in vocab.items():
            self.token_ids[token] = id
        for i, merge in enumerate(merges):
            self.merge_rank[merge] = i

    @classmethod
    def from_files(
        cls,
        vocab_filepath: str,
        merges_filepath: str,
        special_tokens: list[str] | None = None,
    ) -> "Tokenizer":
        try:
            from src.utils import gpt2_bytes_to_unicode
        except ImportError:
            from utils import gpt2_bytes_to_unicode

        byte_decoder = {v: k for k, v in gpt2_bytes_to_unicode().items()}

        with open(vocab_filepath, encoding="utf-8") as vf:
            serialized_vocab: dict[str, int] = json.load(vf)

        vocab: dict[int, bytes] = {
            token_id: bytes(byte_decoder[ch] for ch in token_str)
            for token_str, token_id in serialized_vocab.items()
        }

        merges: list[tuple[bytes, bytes]] = []
        with open(merges_filepath, encoding="utf-8") as mf:
            for line in mf:
                cleaned = line.rstrip("\n")
                parts = cleaned.split(" ")
                if len(parts) != 2:
                    continue
                left_str, right_str = parts
                merges.append(
                    (
                        bytes(byte_decoder[ch] for ch in left_str),
                        bytes(byte_decoder[ch] for ch in right_str),
                    )
                )

        if special_tokens:
            existing_tokens = set(vocab.values())
            next_id = max(vocab.keys(), default=-1) + 1
            for token in special_tokens:
                token_bytes = token.encode("utf-8")
                if token_bytes not in existing_tokens:
                    vocab[next_id] = token_bytes
                    existing_tokens.add(token_bytes)
                    next_id += 1

        return cls(vocab, merges, special_tokens=special_tokens)

    def _pre_tokenize(self, text: str) -> list[str]:
        """Split text into pre-tokens while preserving configured special tokens."""
        return [token for token, _, _ in self._pre_tokenize_with_spans(text)]

    def _pre_tokenize_with_spans(
        self, text: str
    ) -> list[tuple[str, int, int]]:
        """Split text into pre-tokens and return (token, start, end) spans."""
        if not text:
            return []

        if not self.special_tokens:
            return [
                (match.group(0), match.start(), match.end())
                for match in PAT.finditer(text)
            ]

        # Longest-first alternation avoids partial matches for overlapping specials.
        escaped_special_tokens = sorted(
            (re.escape(token) for token in self.special_tokens),
            key=len,
            reverse=True,
        )
        special_pattern = re.compile("|".join(escaped_special_tokens))
        result: list[tuple[str, int, int]] = []
        cursor = 0

        for special_match in special_pattern.finditer(text):
            if cursor < special_match.start():
                segment = text[cursor : special_match.start()]
                for token_match in PAT.finditer(segment):
                    result.append(
                        (
                            token_match.group(0),
                            cursor + token_match.start(),
                            cursor + token_match.end(),
                        )
                    )
            result.append(
                (
                    special_match.group(0),
                    special_match.start(),
                    special_match.end(),
                )
            )
            cursor = special_match.end()

        if cursor < len(text):
            segment = text[cursor:]
            for token_match in PAT.finditer(segment):
                result.append(
                    (
                        token_match.group(0),
                        cursor + token_match.start(),
                        cursor + token_match.end(),
                    )
                )

        return result

    def _merge_pretoken(self, token: str) -> list[bytes]:
        """
        Apply BPE merges inside a single pre-token and return merged byte symbols.
        """
        symbols = [_BYTE_CACHE[b] for b in token.encode("utf-8")]
        if len(symbols) < 2:
            return symbols

        while len(symbols) > 1:
            best_pair: tuple[bytes, bytes] | None = None
            best_rank: int | None = None

            for i in range(len(symbols) - 1):
                pair = (symbols[i], symbols[i + 1])
                rank = self.merge_rank.get(pair)
                if rank is None:
                    continue
                if best_rank is None or rank < best_rank:
                    best_rank = rank
                    best_pair = pair

            if best_pair is None:
                break

            left, right = best_pair
            merged: list[bytes] = []
            i = 0
            while i < len(symbols):
                if (
                    i < len(symbols) - 1
                    and symbols[i] == left
                    and symbols[i + 1] == right
                ):
                    merged.append(left + right)
                    i += 2
                else:
                    merged.append(symbols[i])
                    i += 1
            symbols = merged

        return symbols

    def encode(self, text: str) -> list[int]:
        ids: list[int] = []
        special_tokens = set(self.special_tokens or [])

        for token in self._pre_tokenize(text):
            ids.extend(self._encode_token(token, special_tokens))

        return ids

    def _encode_token(self, token: str, special_tokens: set[str]) -> list[int]:
        if token in special_tokens:
            return [self.token_ids[token.encode("utf-8")]]
        merged_symbols = self._merge_pretoken(token)
        return [self.token_ids[symbol] for symbol in merged_symbols]

    def encode_iterable(self, iterable: Iterable[str]) -> Iterator[int]:
        special_tokens = set(self.special_tokens or [])
        max_special_len = max((len(t) for t in special_tokens), default=0)
        pending = ""

        for chunk in iterable:
            if not chunk:
                continue
            pending += chunk

            token_spans = self._pre_tokenize_with_spans(pending)
            if not token_spans:
                continue

            # Keep a protected suffix so partial special-token prefixes can
            # complete across chunk boundaries.
            protect_start = (
                max(0, len(pending) - (max_special_len - 1))
                if max_special_len > 1
                else len(pending)
            )

            first_unemitted_idx = 0
            # Emit all tokens that are both:
            # 1) not the final token in the current pending text, and
            # 2) entirely before the protected suffix.
            for idx, (token, _, end) in enumerate(token_spans[:-1]):
                if end > protect_start:
                    break
                yield from self._encode_token(token, special_tokens)
                first_unemitted_idx = idx + 1

            pending = pending[token_spans[first_unemitted_idx][1] :]

        if pending:
            for token in self._pre_tokenize(pending):
                yield from self._encode_token(token, special_tokens)

    def decode(self, ids: list[int]) -> str:
        token_bytes = bytearray()
        for token_id in ids:
            token = self.vocab.get(token_id)
            if token is None:
                raise ValueError(f"Unknown token id: {token_id}")
            token_bytes.extend(token)
        return bytes(token_bytes).decode("utf-8", errors="replace")


if __name__ == "__main__":
    # tokenizer = Tokenizer.from_files(
    #     "data/tinystories_vocab.json", "data/tinystories_merges.txt"
    # )
    mock_voc = {
        0: b" ",
        1: b"a",
        2: b"c",
        3: b"e",
        4: b"h",
        5: b"t",
        6: b"th",
        7: b" c",
        8: b" a",
        9: b"the",
        10: b" at",
    }
    merge = [
        (b"t", b"h"),
        (b" ", b"c"),
        (b" ", b"a"),
        (b"th", b"e"),
        (b" a", b"t"),
    ]
    tokenizer = Tokenizer(mock_voc, merge)
    res = tokenizer.encode("the cat ate")
    print(res)
