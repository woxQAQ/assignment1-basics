from bpe import bpe_train
import cProfile


def main():
    vocab, merges = bpe_train(
        "data/TinyStoriesV2-GPT4-train.txt",
        10000,
        ["<|endoftext|>"],
    )
    # vocab, merges = bpe_train(
    #     "tests/fixtures/tinystories_sample.txt", 500, ["<|endoftext|>"]
    # )
    print(f"Vocab size: {len(vocab)}")
    print(f"Number of merges: {len(merges)}")
    print(f"First few merges: {merges[:5]}")


cprofile_output = "profile_output.prof"
cProfile.run("main()", cprofile_output)
