#!/bin/bash

# Setup Environment
source activate corm

# Paths (override these env vars to point at your own data)
INPUT_DIR="${INPUT_DIR:-./data/Pile/jsonl_raw}"
OUTPUT_DIR="${OUTPUT_DIR:-./data/Pile/tokenized}"
VOCAB_FILE="${VOCAB_FILE:-./data/gpt2-vocab.json}"
MERGE_FILE="${MERGE_FILE:-./data/gpt2-merges.txt}"

# Create output directory
mkdir -p $OUTPUT_DIR

echo "Starting tokenization at $(date)"
echo "Input: $INPUT_DIR"
echo "Output: $OUTPUT_DIR"
echo "Workers: 32"
echo "================================"

# Process each shard
for i in $(seq -w 0 29); do
    echo ""
    echo "Processing shard ${i}/29 at $(date)"
    
    python tools/preprocess_data.py \
        --input ${INPUT_DIR}/${i}.jsonl \
        --output-prefix ${OUTPUT_DIR}/${i} \
        --vocab-file ${VOCAB_FILE} \
        --tokenizer-type GPT2BPETokenizer \
        --merge-file ${MERGE_FILE} \
        --append-eod \
        --workers 32
    
    if [ $? -eq 0 ]; then
        echo "✓ Shard ${i} completed successfully"
    else
        echo "✗ ERROR: Shard ${i} failed!"
        exit 1
    fi
done

echo ""
echo "================================"
echo "All shards tokenized successfully at $(date)"
echo ""
echo "Output files:"
ls -lh ${OUTPUT_DIR}/

echo ""
echo "Total output size:"
du -sh ${OUTPUT_DIR}/



# for i in $(seq -w 0 29); do
#     python tools/preprocess_data.py \
#         --input ../pile/${i}.jsonl \
#         --output-prefix ../pile_gpt_test/${i} \
#         --vocab-file ../gpt2-vocab.json \
#         --tokenizer-type GPT2BPETokenizer \
#         --merge-file ../gpt2-merges.txt \
#         --append-eod \
#         --workers 32
# done