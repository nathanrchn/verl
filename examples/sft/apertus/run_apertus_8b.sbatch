#!/bin/bash
#SBATCH --job-name=apertus-sft-8b-2509
#SBATCH --account=infra01
#SBATCH --time=12:00:00
#SBATCH --exclusive
#SBATCH --nodes=18
#SBATCH --gpus-per-node=4
#SBATCH --ntasks-per-node=5
#SBATCH --mem=460800
#SBATCH --partition=normal
#SBATCH --output=logs/%j/log.out
#SBATCH --error=logs/%j/log.err
#SBATCH --exclude=nid006634,nid006701,nid006948,nid006588,nid006629,nid006910,nid007254,nid007078,nid006619,nid006840,nid006905,nid006941,nid006947,nid006922,nid007074,nid007131,nid007189,nid007129,nid007184,nid007176,nid007177,nid007183,nid007090,nid007551,nid007531,nid007539,nid007558,nid006988,nid006990,nid006987,nid006989,nid007363,nid006606,nid007410,nid007096,nid007566,nid006774,nid007343,nid006867,nid007323,nid007489,nid006676,nid006677,nid007411,nid006848,nid006681,nid007626,nid007612,nid006887,nid006577,nid006729,nid006831,nid007520,nid007589,nid007614,nid006955,nid007592,nid007344,nid007374,nid007134,nid007628,nid007382,nid007141,nid007155,nid007286,nid006589,nid007024,nid007025

set -ex

# Cluster and infrastructure
MASTER_PORT=30000
WORK_DIR="/users/$(id -un)/projects/verl"

# Paths
SAVE_FOLDER="/iopsstor/scratch/cscs/$(id -un)/apertus-sft-runs"
DATASET_PATH="/iopsstor/scratch/cscs/nathanrchn/apertus-full-1"

# Model configuration
MODEL_PATH="/users/nathanrchn/scratch/apertus-sft-runs/Apertus-8B-2509-main-lr5e-6-bs256-test64-warmuplinear-lr_warmup_steps_ratio0.03/2026-01-12_17-41-41/global_step_844/huggingface"
MODEL_REVISION="main"
TOKENIZER_PATH="/users/nathanrchn/scratch/apertus-sft-runs/Apertus-8B-2509-main-lr5e-6-bs256-test64-warmuplinear-lr_warmup_steps_ratio0.03/2026-01-12_17-41-41/global_step_844/huggingface"
MODEL_DTYPE="bfloat16"

# Data configuration
MICRO_BATCH_SIZE_PER_GPU=8
MAX_LENGTH=8192
TRAIN_BATCH_SIZE=512
VAL_BATCH_SIZE=128
ROLLOUT_BATCH_SIZE=8192
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
TEST_FREQ=256
TOTAL_TRAINING_STEPS=null
WEIGHT_DECAY=0.0

# Experiment configuration
PROJECT_NAME="apertus-sft-full"

# Multi-turn configuration
ENABLE_MULTITURN=true
MESSAGES_KEY="messages"
TOOLS_KEY="tools"
ENABLE_THINKING_KEY="enable_thinking"
CUSTOM_CLS_PATH="verl/utils/dataset/multiturn_sft_dataset.py"
CUSTOM_CLS_NAME="ApertusSFTDataset"

# Rollout
USE_ROLLOUT=true
ROLLOUT_MODEL_PATH="/users/nathanrchn/scratch/apertus-sft-runs/Apertus-8B-2509-main-lr5e-6-bs256-test64-warmuplinear-lr_warmup_steps_ratio0.03/2026-01-12_17-41-41/global_step_844/huggingface"

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

# Build project wheel once on the head node and store it in a shared location
WHEEL_DIR="$save_path/wheels"
mkdir -p "$WHEEL_DIR"
srun --nodes=1 --ntasks=1 --nodelist=$head_node --container-writable --environment=verl_swiss_old --kill-on-bad-exit=1 --output=$save_path/wheel_build.log --error=$save_path/wheel_build.err \
    bash --norc --noprofile -c "\
set -ex
cd $WORK_DIR
pip wheel . --no-cache-dir --no-deps -w $WHEEL_DIR"
PACKAGE_WHEEL=$(ls -t "$WHEEL_DIR"/*.whl | head -n1)

if [ "$USE_ROLLOUT" = true ]; then
    rollout_node1=${nodes[-1]}
    rollout_node2=${nodes[-2]}
    rollout_node1_ip=$(srun --nodes=1 --ntasks=1 --nodelist=$rollout_node1 hostname -i)
    rollout_node2_ip=$(srun --nodes=1 --ntasks=1 --nodelist=$rollout_node2 hostname -i)
    TRAINING_NODES=$((SLURM_NNODES - 2))
    ROLLOUT_URL="http://${rollout_node1_ip}:30000"
else
    TRAINING_NODES=$SLURM_NNODES
    ROLLOUT_URL=null
fi


for local_rank in $(seq 0 $((TRAINING_NODES - 1))); do
    node=${nodes[$local_rank]}

    if [ "$local_rank" -eq 0 ]; then
        install_command="pip install /iopsstor/scratch/cscs/nathanrchn/evaluate-0.4.6-py3-none-any.whl"
    else
        install_command=""
    fi

    srun --nodes=1 --ntasks=1 --nodelist=$node --container-writable --environment=verl_swiss_old --kill-on-bad-exit=1 --output=$save_path/node_${local_rank}.log --error=$save_path/node_${local_rank}.err \
        bash --norc --noprofile -c "\
set -ex

echo $node

export no_proxy="0.0.0.0,$no_proxy"
export NO_PROXY="0.0.0.0,$NO_PROXY"

export NCCL_DEBUG=INFO
export HYDRA_FULL_ERROR=1

cd $WORK_DIR

$install_command

pip install --no-deps /iopsstor/scratch/cscs/nathanrchn/nltk-3.9.2-py3-none-any.whl
pip install --no-deps /iopsstor/scratch/cscs/nathanrchn/emoji-2.15.0-py3-none-any.whl
pip install --no-deps /iopsstor/scratch/cscs/nathanrchn/syllapy-0.7.2-py3-none-any.whl
pip install --no-deps /iopsstor/scratch/cscs/nathanrchn/langdetect-1.0.9-py3-none-any.whl
pip install --no-deps /iopsstor/scratch/cscs/nathanrchn/immutabledict-4.2.2-py3-none-any.whl

pip install $PACKAGE_WHEEL --no-cache-dir --no-deps --force-reinstall

torchrun --nnodes=$TRAINING_NODES --nproc_per_node=4 --node_rank=$local_rank --master_addr=$head_node_ip --master_port=$MASTER_PORT \
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
    WORKER_URLS=""
    for node_idx in 1 2; do
        if [ "$node_idx" -eq 1 ]; then
            node=$rollout_node1
            node_ip=$rollout_node1_ip
        else
            node=$rollout_node2
            node_ip=$rollout_node2_ip
        fi

        for i in $(seq 0 3); do
            port=$((50000 + i))
            WORKER_URLS="${WORKER_URLS} http://${node_ip}:${port}"
            srun --nodes=1 --ntasks=1 --nodelist=$node --container-writable --environment=verl_swiss_old --kill-on-bad-exit=1 --gpus-per-task=1 --cpus-per-task=50 --gpu-bind=map_gpu:${i} --overlap --output=$save_path/node_rollout_${node_idx}_$i.log --error=$save_path/node_rollout_${node_idx}_$i.err \
                bash --norc --noprofile -c "\
set -ex

export no_proxy="0.0.0.0,$no_proxy"
export NO_PROXY="0.0.0.0,$NO_PROXY"

export NCCL_DEBUG=WARN

export CUDA_DEVICE_ORDER=PCI_BUS_ID
export CUDA_VISIBLE_DEVICES=$i
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:False

python -m sglang.launch_server --model-path=$ROLLOUT_MODEL_PATH --dtype=bfloat16 --host=0.0.0.0 --port=$port --decode-log-interval=1 --skip-server-warmup --random-seed 42 --grammar-backend llguidance --mem-fraction-static 0.6 --max-running-requests 60" &
        done
    done

    srun --nodes=1 --ntasks=1 --nodelist=$rollout_node1 --container-writable --environment=sglang_router --kill-on-bad-exit=1 --cpus-per-task=50 --overlap --output=$save_path/node_rollout_router.log --error=$save_path/node_rollout_router.err \
        bash --norc --noprofile -c "\
set -ex

export no_proxy="0.0.0.0,$no_proxy"
export NO_PROXY="0.0.0.0,$NO_PROXY"

python -m sglang_router.launch_router --host 0.0.0.0 --port 30000 --worker-urls $WORKER_URLS --model-path $ROLLOUT_MODEL_PATH" &
fi

wait -n
scancel $SLURM_JOB_ID

echo "[FINISHED]"
