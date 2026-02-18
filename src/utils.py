from pathlib import Path
import json


def gpt2_bytes_to_unicode() -> dict[int, str]:
    """Map every byte [0, 255] to a printable unicode character (GPT-2 style)."""
    bs = (
        list(range(ord("!"), ord("~") + 1))
        + list(range(ord("¡"), ord("¬") + 1))
        + list(range(ord("®"), ord("ÿ") + 1))
    )
    cs = bs[:]
    n = 0
    for b in range(2**8):
        if b not in bs:
            bs.append(b)
            cs.append(2**8 + n)
            n += 1
    return dict(zip(bs, [chr(x) for x in cs]))


def encode_token(token: bytes, byte_encoder: dict[int, str]) -> str:
    return "".join(byte_encoder[b] for b in token)


def serialize_vocab(vocab: dict[int, bytes], vocab_path: Path) -> None:
    byte_encoder = gpt2_bytes_to_unicode()
    serialized_vocab: dict[str, int] = {}
    for token_id in sorted(vocab):
        token = vocab[token_id]
        serialized_vocab[encode_token(token, byte_encoder)] = token_id

    vocab_path.parent.mkdir(parents=True, exist_ok=True)
    with vocab_path.open("w", encoding="utf-8") as f:
        json.dump(serialized_vocab, f, ensure_ascii=False, indent=2)
        f.write("\n")


def serialize_merges(
    merges: list[tuple[bytes, bytes]], merges_path: Path
) -> None:
    byte_encoder = gpt2_bytes_to_unicode()
    merges_path.parent.mkdir(parents=True, exist_ok=True)
    with merges_path.open("w", encoding="utf-8") as f:
        for left, right in merges:
            left_str = encode_token(left, byte_encoder)
            right_str = encode_token(right, byte_encoder)
            f.write(f"{left_str} {right_str}\n")
