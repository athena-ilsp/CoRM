#!/bin/bash

# Activate your Python/conda environment first, e.g.:
# conda activate corm

export LD_LIBRARY_PATH=$CONDA_PREFIX/lib:$LD_LIBRARY_PATH

# Set environment variables for distributed training
export CUDA_DEVICE_MAX_CONNECTIONS=1
export PYTORCH_ALLOC_CONF=expandable_segments:True

export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export NUMEXPR_NUM_THREADS=1

export NCCL_ASYNC_ERROR_HANDLING=1
export NCCL_IB_DISABLE=0
export NCCL_DEBUG=WARN

# Read input parameters
GPUS_PER_NODE=${1:-"4"}
MASTER_ADDR=${MASTER_ADDR:-"localhost"}
MASTER_PORT=${MASTER_PORT:-"6000"}
NNODES=${SLURM_NNODES:-"1"}
NODE_RANK=${RANK:-"0"}
WORLD_SIZE=$(($GPUS_PER_NODE*$NNODES))

# Training parameters
TRAIN_ITERS="${TRAIN_ITERS:-60000}"
MICRO_BATCH_SIZE="${MICRO_BATCH_SIZE:-16}"
PROJECT_NAME="${PROJECT_NAME:-dense-182m-v1}"

# Where to write checkpoints and where your tokenized data / vocab live.
# Override these env vars rather than editing paths inline.
CHECKPOINT_PATH="${CHECKPOINT_DIR:-./checkpoints}/$PROJECT_NAME"
mkdir -p $CHECKPOINT_PATH

DATA_DIR="${DATA_DIR:-./data}"
VOCAB_FILE="${VOCAB_FILE:-$DATA_DIR/gpt2-vocab.json}"
MERGE_FILE="${MERGE_FILE:-$DATA_DIR/gpt2-merges.txt}"

# Space-separated "<weight> <path-prefix>" pairs for one or more tokenized
# Megatron datasets. Set DATA_PATH yourself to point at your own data, e.g.:
#   export DATA_PATH="1.0 ${DATA_DIR}/tokenized/00_text_document"
DATA_PATH="${DATA_PATH:?Set DATA_PATH to your tokenized dataset(s), e.g. \"1.0 /path/to/00_text_document\"}"


# Distributed setup for SLURM
DISTRIBUTED_ARGS=(
    --nproc_per_node $GPUS_PER_NODE
    --nnodes $NNODES
    --node_rank $NODE_RANK
    --master_addr $MASTER_ADDR
    --master_port $MASTER_PORT
)

# Model arguments (182M dense)
MODEL_ARGS=(
    --use-mcore-models
    --disable-bias-linear
    --seq-length 1024
    --max-position-embeddings 1024
    --num-layers 12
    --hidden-size 768
    --ffn-hidden-size $((768 * 4))
    --num-attention-heads 12
    --init-method-std 0.01
    --attention-dropout 0.0
    --hidden-dropout 0.0
    --normalization RMSNorm
    --position-embedding-type rope
    --swiglu
    --untie-embeddings-and-output-weights
    --group-query-attention
    --num-query-groups 4
    --no-masked-softmax-fusion
    --no-position-embedding
    --rotary-base 1000000
    --use-flash-attn
    --attention-backend unfused
)

# Data and training args
DATA_ARGS=(
    --vocab-file $VOCAB_FILE
    --merge-file $MERGE_FILE
    --make-vocab-size-divisible-by 1024
    --data-path $DATA_PATH
    --split 969,30,1
)

TRAINING_ARGS=(
    --micro-batch-size $MICRO_BATCH_SIZE
    --global-batch-size 512
    --lr 5e-4
    --train-iters $TRAIN_ITERS
    --lr-decay-style cosine
    --min-lr 5e-5
    --lr-warmup-fraction 0.01
    --clip-grad 1.0
    --bf16
)

# Model parallelism arguments
MODEL_PARALLEL_ARGS=(
    --tensor-model-parallel-size 1
    --pipeline-model-parallel-size 1
    --use-distributed-optimizer
    --sequence-parallel
)

# Logging arguments
LOGGING_ARGS=(
    --log-interval 10
    --log-throughput
    --save-interval 2500
    --eval-interval 5000
    --eval-iters 100
    --save $CHECKPOINT_PATH
    --load $CHECKPOINT_PATH
    --no-load-optim
    --no-load-rng
    --tensorboard-dir "${CHECKPOINT_PATH}/tensorboard"
)

# Optional: Add WANDB logging if API key is available
if [ -n "${WANDB_API_KEY}" ]; then
    LOGGING_ARGS+=(
        --wandb-project "MoE"
        --wandb-exp-name $PROJECT_NAME
    )
fi

# Run the training
torchrun ${DISTRIBUTED_ARGS[@]} pretrain_gpt.py \
    ${MODEL_ARGS[@]} \
    ${DATA_ARGS[@]} \
    ${TRAINING_ARGS[@]} \
    ${MODEL_PARALLEL_ARGS[@]} \
    ${LOGGING_ARGS[@]} |& tee -a $CHECKPOINT_PATH/train.log
