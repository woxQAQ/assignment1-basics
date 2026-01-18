from collections import Counter
from typing_extensions import Text, Tuple
import regex as re


class BPE:
    def __init__(self) -> None:
        self.pat = re.compile(
            r"""'(?:[sdmt]|ll|ve|re)| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+"""
        )
PAT = re.compile(
    r"""'s|'t|'re|'ve|'m|'ll|'d| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+"""
)
def pre_tokenize(text_block:str) -> Counter[tuple[bytes,...]]:
    c: Counter[tuple[bytes, ...]] = Counter()
    for token in PAT.finditer(text_block):
        c[tuple(ch.encode("utf-8") for ch in token.group())] += 1
    return c

def bpe_train(text: str, merge_times: int = 6) -> list[bytes]:
    # step 0: load vocabulary
    blocks = [b for b in text.split("<|endoftext|>") if b != ""]
    voc = [b"<|endoftext|>"] + [bytes([i]) for i in range(256)]
    # step 1: pre-tokenization
    c = pre_tokenize(blocks[0])
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
<|endoftext|>"""

    voc = bpe_train(text)
    print(voc)
