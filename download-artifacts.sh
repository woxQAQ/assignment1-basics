#!/usr/bin/env bash
# HuggingFace mirror sites for faster access in China
# Alternatives: hf-mirror.com, huggingface.co-mirror.com

MIRROR_BASE="https://hf-mirror.com"

mkdir -p data
cd data

# Download TinyStories dataset via HuggingFace mirror
echo "Downloading TinyStories dataset..."
wget ${MIRROR_BASE}/datasets/roneneldan/TinyStories/resolve/main/TinyStoriesV2-GPT4-train.txt
wget ${MIRROR_BASE}/datasets/roneneldan/TinyStories/resolve/main/TinyStoriesV2-GPT4-valid.txt

# Download OWT dataset via HuggingFace mirror
echo "Downloading OWT dataset..."
wget ${MIRROR_BASE}/datasets/stanford-cs336/owt-sample/resolve/main/owt_train.txt.gz
wget ${MIRROR_BASE}/datasets/stanford-cs336/owt-sample/resolve/main/owt_valid.txt.gz

# Decompress OWT files
echo "Decompressing OWT files..."
gunzip -f owt_train.txt.gz
gunzip -f owt_valid.txt.gz

echo "Download complete!"
ls -lh
