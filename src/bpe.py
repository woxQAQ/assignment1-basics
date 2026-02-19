from pre_tokenizer import PAT
from collections.abc import Iterable, Iterator
import regex as re

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
        pass

    def _pre_tokenize(self, text: str) -> list[str]:
        """Split text into pre-tokens while preserving configured special tokens."""
        if not text:
            return []

        if not self.special_tokens:
            return PAT.findall(text)

        # Longest-first alternation avoids partial matches for overlapping specials.
        escaped_special_tokens = sorted(
            (re.escape(token) for token in self.special_tokens),
            key=len,
            reverse=True,
        )
        pattern = "|".join(escaped_special_tokens)
        # Capture group keeps matched special tokens in the split output.
        parts = re.split(f"({pattern})", text)

        specials = set(self.special_tokens)
        result: list[str] = []
        for part in parts:
            if not part:
                continue
            if part in specials:
                result.append(part)
            else:
                result.extend(PAT.findall(part))
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
            if token in special_tokens:
                ids.append(self.token_ids[token.encode("utf-8")])
                continue

            merged_symbols = self._merge_pretoken(token)
            ids.extend(self.token_ids[symbol] for symbol in merged_symbols)

        return ids

    def encode_iterable(self, iterable: Iterable[str]) -> Iterator[int]:
        pass

    def decode(self, ids: list[int]) -> str:
        pass


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
