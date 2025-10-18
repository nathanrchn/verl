#!/bin/bash
#SBATCH --job-name=apertus-sft-8b-2509
#SBATCH --cpus-per-task=288
#SBATCH --gres=gpu:4
#SBATCH --account=a-infra01
#SBATCH --ntasks-per-node=1
#SBATCH --time=02:00:00
#SBATCH --exclusive
#SBATCH --nodes=9
#SBATCH --partition=normal
#SBATCH --environment=verl_swiss
#SBATCH --output=logs/%j/log.out
#SBATCH --error=logs/%j/log.err

set -x

# Cluster and infrastructure
MASTER_PORT=30000
WORK_DIR="/users/$(id -un)/projects/verl"

# Paths
SAVE_FOLDER="/iopsstor/scratch/cscs/$(id -un)/apertus-sft-runs"
DATASET_PATH="/iopsstor/scratch/cscs/nathanrchn/apertus-sft-code-1"

# Model configuration
MODEL_PATH="swiss-ai/Apertus-8B-2509"
MODEL_REVISION="main"
TOKENIZER_PATH="swiss-ai/Apertus-8B-Instruct-2509"
MODEL_DTYPE="bfloat16"

# Data configuration
MICRO_BATCH_SIZE_PER_GPU=8
MAX_LENGTH=4096
TRAIN_BATCH_SIZE=256
VAL_BATCH_SIZE=32
ROLLOUT_BATCH_SIZE=256
ROLLOUT_MAX_LENGTH=4096
TRAIN_FILE="train.parquet"
VAL_FILE="val.parquet"
ROLLOUT_FILE="rollout.parquet"
USE_DYNAMIC_BSZ=false
MAX_TOKEN_LEN_PER_GPU=65536

# Training configuration
LEARNING_RATE="5e-6"
WARMUP_STYLE="linear"
LR_WARMUP_STEPS_RATIO=0.03
TOTAL_EPOCHS=1
TEST_FREQ=128
TOTAL_TRAINING_STEPS=null
WEIGHT_DECAY=0.0

# Experiment configuration
PROJECT_NAME="apertus-sft-code"

# Multi-turn configuration
ENABLE_MULTITURN=true
MESSAGES_KEY="messages"
TOOLS_KEY="tools"
ENABLE_THINKING_KEY="enable_thinking"
CUSTOM_CLS_PATH="verl/utils/dataset/multiturn_sft_dataset.py"
CUSTOM_CLS_NAME="ApertusSFTDataset"

# Rollout
USE_ROLLOUT=true
ROLLOUT_MODEL_PATH="swiss-ai/Apertus-8B-Instruct-2509"

# Logging
LOGGER='["console","wandb"]'

# Generate run name and experiment name from key parameters
MODEL_NAME=$(basename "$MODEL_PATH")
RUN_NAME="${MODEL_NAME}-${MODEL_REVISION}-lr${LEARNING_RATE}-bs${TRAIN_BATCH_SIZE}-test${TEST_FREQ}-warmup${WARMUP_STYLE}-lr_warmup_steps_ratio${LR_WARMUP_STEPS_RATIO}"

nodes=($(scontrol show hostnames "$SLURM_JOB_NODELIST"))
head_node=${nodes[0]}
head_node_ip=$(srun --nodes=1 --ntasks=1 --nodelist=$head_node hostname -i)

current_datetime=$(date '+%Y-%m-%d_%H-%M-%S')
save_path="$SAVE_FOLDER/$RUN_NAME/$current_datetime"
dataset_path=$DATASET_PATH

mkdir -p "$save_path"
cp "$0" "$save_path/run_apertus_8b.sbatch"

if [ "$USE_ROLLOUT" = true ]; then
    rollout_node=${nodes[-1]}
    rollout_node_ip=$(srun --nodes=1 --ntasks=1 --nodelist=$rollout_node hostname -i)
    TRAINING_NODES=$((SLURM_NNODES - 1))
    ROLLOUT_URL="http://${rollout_node_ip}:30000"
else
    TRAINING_NODES=$SLURM_NNODES
    ROLLOUT_URL=null
fi


for local_rank in $(seq 0 $((TRAINING_NODES - 1))); do
    node=${nodes[$local_rank]}

    srun --nodes=1 --ntasks=1 --nodelist=$node --container-writable --output=$save_path/node_${local_rank}.log --error=$save_path/node_${local_rank}.err \
        bash --norc --noprofile -c "\
set -x

export no_proxy="0.0.0.0,$no_proxy"
export NO_PROXY="0.0.0.0,$NO_PROXY"

export NCCL_DEBUG=INFO
export HYDRA_FULL_ERROR=1

cd $WORK_DIR

pip install -e . --no-cache-dir --no-deps

torchrun --nnodes=$TRAINING_NODES --nproc_per_node=4 --node_rank=$local_rank --master_addr=$head_node --master_port=$MASTER_PORT \
    -m verl.trainer.sft_trainer \
    data.train_files=$dataset_path/$TRAIN_FILE \
    data.val_files=$dataset_path/$VAL_FILE \
    data.rollout_files=$dataset_path/$ROLLOUT_FILE \
    data.messages_key=$MESSAGES_KEY \
    data.tools_key=$TOOLS_KEY \
    data.enable_thinking_key=$ENABLE_THINKING_KEY \
    data.micro_batch_size_per_gpu=$MICRO_BATCH_SIZE_PER_GPU \
    data.custom_cls.path=$CUSTOM_CLS_PATH \
    data.custom_cls.name=$CUSTOM_CLS_NAME \
    data.max_length=$MAX_LENGTH \
    data.train_batch_size=$TRAIN_BATCH_SIZE \
    data.val_batch_size=$VAL_BATCH_SIZE \
    data.rollout_batch_size=$ROLLOUT_BATCH_SIZE \
    +data.rollout_max_size=$ROLLOUT_MAX_LENGTH \
    data.use_dynamic_bsz=$USE_DYNAMIC_BSZ \
    data.max_token_len_per_gpu=$MAX_TOKEN_LEN_PER_GPU \
    +data.apply_chat_template_kwargs.truncation=true \
    +data.add_generation_prompt=false \
    model.path=$MODEL_PATH \
    +model.revision=$MODEL_REVISION \
    model.tokenizer_path=$TOKENIZER_PATH \
    model.use_remove_padding=true \
    engine.model_dtype=$MODEL_DTYPE \
    engine.strategy=fsdp \
    optim.lr=$LEARNING_RATE \
    optim.warmup_style=$WARMUP_STYLE \
    optim.weight_decay=$WEIGHT_DECAY \
    optim.lr_warmup_steps_ratio=$LR_WARMUP_STEPS_RATIO \
    rollout_url=$ROLLOUT_URL \
    checkpoint.save_contents='["model","optimizer","extra","hf_model"]' \
    trainer.default_local_dir=$save_path \
    trainer.project_name=$PROJECT_NAME \
    trainer.experiment_name=$RUN_NAME \
    trainer.test_freq=$TEST_FREQ \
    trainer.total_training_steps=$TOTAL_TRAINING_STEPS \
    trainer.total_epochs=$TOTAL_EPOCHS \
    trainer.logger=$LOGGER" &
done

if [ "$USE_ROLLOUT" = true ]; then
    srun --nodes=1 --ntasks=1 --nodelist=$rollout_node --container-writable --output=$save_path/node_rollout.log --error=$save_path/node_rollout.err \
        bash --norc --noprofile -c "\
set -x

export no_proxy="0.0.0.0,$no_proxy"
export NO_PROXY="0.0.0.0,$NO_PROXY"

export NCCL_DEBUG=INFO

python -m sglang.launch_server --model-path=$ROLLOUT_MODEL_PATH --host=0.0.0.0 --port=30000 --decode-log-interval 1 --skip-server-warmup --random-seed 42
" &
fi

wait

echo "[FINISHED]"
