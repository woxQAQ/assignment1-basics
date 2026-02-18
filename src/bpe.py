from pre_tokenizer import PAT
from collections.abc import Iterable, Iterator
import regex as re



class Tokenizer:
    __slots__ = ["token_ids", "merge_rank", "special_tokens"]

    def __init__(
        self,
        vocab: dict[int, bytes],
        merges: list[tuple[bytes, bytes]],
        special_tokens: list[str] | None = None,
    ):
        self.special_tokens = special_tokens
        self.token_ids: dict[bytes,int] = {}
        self.merge_rank: dict[tuple[bytes,bytes],int] = {}
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

    def _pre_tokenize(self, text: str) -> list[tuple[bytes, bytes]]:
        if self.special_tokens is not None:
            escaped_special_token = sorted(
                (re.escape(t) for t in self.special_tokens),
                key=len,
                reverse=True,
            )
            pattern = "|".join(escaped_special_token)
            parts = [block for block in re.split(pattern, text)]
        else:
            parts = [text]
        result = []
        for part in parts:
            result.append(PAT.findall(part))
        return result

    def encode(self, text: str) -> list[int]:
        pretoken_result = self._pre_tokenize(text)


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
        (b" ", "a"),
        (b"th", b"e"),
        (b" a", b"t"),
    ]
    tokenizer = Tokenizer(mock_voc, merge)
    tokenizer.encode("the cat ate")
