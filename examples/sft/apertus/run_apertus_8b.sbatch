#!/bin/bash
#SBATCH --job-name=apertus-sft-8b-2509
#SBATCH --cpus-per-task=288
#SBATCH --gres=gpu:4
#SBATCH --account=a-infra01
#SBATCH --ntasks-per-node=1
#SBATCH --time=04:00:00
#SBATCH --exclusive
#SBATCH --nodes=16
#SBATCH --partition=normal
#SBATCH --environment=nccl
#SBATCH --output=logs/%j/log.out
#SBATCH --error=logs/%j/log.err

set -x

# Cluster and infrastructure
MASTER_PORT=30000
WORK_DIR="/users/$(id -un)/projects/verl"

# Paths
SAVE_FOLDER="/iopsstor/scratch/cscs/$(id -un)/apertus-sft-runs"
DATASET_PATH="/iopsstor/scratch/cscs/nathanrchn/debug-apertus-sft-mixture-8e-with-eval"

# Model configuration
MODEL_PATH="swiss-ai/Apertus-8B-2509"
MODEL_REVISION="main"
TOKENIZER_PATH="swiss-ai/Apertus-8B-Instruct-2509"
MODEL_DTYPE="bf16"

# Data configuration
MICRO_BATCH_SIZE_PER_GPU=32
MAX_LENGTH=4096
ROLLOUT_MAX_LENGTH=512
MAX_TOKEN_LEN_PER_GPU=32768
TRAIN_FILE="train.parquet"
VAL_FILE="val.parquet"
ROLLOUT_FILE="rollout.parquet"

# Training configuration
LEARNING_RATE="5e-6"
WARMUP_STYLE="linear"
LR_WARMUP_STEPS_RATIO=0.03
TOTAL_EPOCHS=1
TEST_FREQ=32

# Experiment configuration
PROJECT_NAME="apertus-sft"

# Multi-turn configuration
ENABLE_MULTITURN=true
MESSAGES_KEY="messages"
CUSTOM_CLS_PATH="verl/utils/dataset/multiturn_sft_dataset.py"
CUSTOM_CLS_NAME="ApertusSFTDataset"

# Logging
LOGGER='["console","wandb"]'

# Generate run name and experiment name from key parameters
MODEL_NAME=$(basename "$MODEL_PATH")
RUN_NAME="${MODEL_NAME}-${MODEL_REVISION}-lr${LEARNING_RATE}-bs${MICRO_BATCH_SIZE_PER_GPU}-test${TEST_FREQ}"

nodes=($(scontrol show hostnames "$SLURM_JOB_NODELIST"))
head_node=${nodes[0]}

current_datetime=$(date '+%Y-%m-%d_%H-%M-%S')
save_path="$SAVE_FOLDER/$RUN_NAME/$current_datetime"
dataset_path=$DATASET_PATH

mkdir -p "$save_path"
cp "$0" "$save_path/run_apertus_8b.sbatch"

for local_rank in $(seq 0 $((SLURM_NNODES - 1))); do
    node=${nodes[$local_rank]}

    srun --nodes=1 --ntasks=1 --nodelist=$node --container-writable --output=$save_path/node_${local_rank}.log --error=$save_path/node_${local_rank}.err \
        bash --norc --noprofile -c "\
set -x

export no_proxy="0.0.0.0,$no_proxy"
export NO_PROXY="0.0.0.0,$NO_PROXY"

export NCCL_NET="ofi"
export NCCL_DEBUG=INFO
export HYDRA_FULL_ERROR=1

pip install --upgrade pip setuptools wheel --trusted-host pypi.org --trusted-host pypi.python.org --trusted-host files.pythonhosted.org
pip install git+https://github.com/huggingface/transformers.git --no-deps --retries 3 --trusted-host pypi.org --trusted-host pypi.python.org --trusted-host files.pythonhosted.org

cd $WORK_DIR

pip install -e . --no-cache-dir

torchrun --nnodes=${SLURM_NNODES} --nproc_per_node=4 --node_rank=${local_rank} --master_addr=${head_node} --master_port=${MASTER_PORT} \
    -m verl.trainer.sft_trainer \
    data.train_files=$dataset_path/$TRAIN_FILE \
    data.val_files=$dataset_path/$VAL_FILE \
    +data.rollout_files=$dataset_path/$ROLLOUT_FILE \
    +data.multiturn.enable=$ENABLE_MULTITURN \
    +data.multiturn.messages_key=$MESSAGES_KEY \
    data.micro_batch_size_per_gpu=$MICRO_BATCH_SIZE_PER_GPU \
    data.custom_cls.path=$CUSTOM_CLS_PATH \
    data.custom_cls.name=$CUSTOM_CLS_NAME \
    data.max_length=$MAX_LENGTH \
    +data.rollout_max_size=$ROLLOUT_MAX_LENGTH \
    data.max_token_len_per_gpu=$MAX_TOKEN_LEN_PER_GPU \
    +data.apply_chat_template_kwargs.truncation=true \
    +data.add_generation_prompt=false \
    model.path=$MODEL_PATH \
    +model.revision=$MODEL_REVISION \
    model.tokenizer_path=$TOKENIZER_PATH \
    model.use_remove_padding=true \
    engine.model_dtype=$MODEL_DTYPE \
    optim.lr=$LEARNING_RATE \
    optim.warmup_style=$WARMUP_STYLE \
    optim.lr_warmup_steps_ratio=$LR_WARMUP_STEPS_RATIO \
    checkpoint.save_contents='["model","optimizer","extra","hf_model"]' \
    trainer.default_local_dir=$save_path \
    trainer.project_name=$PROJECT_NAME \
    trainer.experiment_name=$RUN_NAME \
    trainer.test_freq=$TEST_FREQ \
    trainer.total_epochs=$TOTAL_EPOCHS \
    trainer.logger=$LOGGER" &
done

wait

echo "[FINISHED]"
