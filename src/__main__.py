from pre_tokenizer import pre_tokenize
from bpe_train import bpe_train, pre_tokenize_file


def main():
    # pre_tokenize_file(
    #     "data/TinyStoriesV2-GPT4-train.txt",
    #     ["<|endoftext|>"],
    #     parallel=True,
    #     profile_workers=True,  # Enable profiling for worker processes
    # )
    vocab, merges = bpe_train(
        "data/TinyStoriesV2-GPT4-train.txt",
        10000,
        ["<|endoftext|>"],
    )
    # vocab, merges = bpe_train(
    #     "tests/fixtures/tinystories_sample.txt", 500, ["<|endoftext|>"]
    # )
    # print(f"Vocab size: {len(vocab)}")
    # print(f"Number of merges: {len(merges)}")
    # print(f"First few merges: {merges[:5]}")


if __name__ == "__main__":
    main()
