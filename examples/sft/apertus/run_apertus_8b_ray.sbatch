#!/bin/bash
#SBATCH --job-name=apertus-sft-8b-ray
#SBATCH --account=infra01
#SBATCH --time=12:00:00
#SBATCH --exclusive
#SBATCH --nodes=18
#SBATCH --gpus-per-node=4
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=200
#SBATCH --mem=460800
#SBATCH --partition=normal
#SBATCH --output=logs/%j/log.out
#SBATCH --error=logs/%j/log.err
#SBATCH --exclude=nid006634,nid006701,nid006948,nid006588,nid006629,nid006910,nid007254,nid007078,nid006619,nid006840,nid006905,nid006941,nid006947,nid006922,nid007074,nid007131,nid007189,nid007129,nid007184,nid007176,nid007177,nid007183,nid007090,nid007551,nid007531,nid007539,nid007558,nid006988,nid006990,nid006987,nid006989,nid007363,nid006606,nid007410,nid007096,nid007566,nid006774,nid007343,nid006867,nid007323,nid007489,nid006676,nid006677,nid007411,nid006848,nid006681,nid007626,nid007612,nid006887,nid006577,nid006729,nid006831,nid007520,nid007589,nid007614,nid006955,nid007592,nid007344,nid007374,nid007134,nid007628,nid007382,nid007141,nid007155,nid007286,nid006589,nid007024,nid007025

set -ex

# Cluster and infrastructure
RAY_PORT=6379
WORK_DIR="/users/$(id -un)/projects/verl"

# Paths
SAVE_FOLDER="/iopsstor/scratch/cscs/$(id -un)/apertus-sft-runs"
DATASET_PATH="/capstor/store/cscs/swissai/infra01/reasoning/users/nathanrchn/data/apertus-format-following-5"

# Model configuration
MODEL_PATH="swiss-ai/Apertus-8B-2509"
MODEL_REVISION="main"
TOKENIZER_PATH="swiss-ai/Apertus-8B-Instruct-2509"
MODEL_DTYPE="bfloat16"

# Data configuration
MICRO_BATCH_SIZE_PER_GPU=8
MAX_LENGTH=8192
TRAIN_BATCH_SIZE=512
VAL_BATCH_SIZE=128
TRAIN_FILE="train.parquet"
VAL_FILE="val.parquet"
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
PROJECT_NAME="apertus-sft-format-following-ray"

# Multi-turn configuration
ENABLE_MULTITURN=true
MESSAGES_KEY="messages"
TOOLS_KEY="tools"
ENABLE_THINKING_KEY="enable_thinking"
CUSTOM_CLS_PATH="verl/utils/dataset/multiturn_sft_dataset.py"
CUSTOM_CLS_NAME="ApertusSFTDataset"

# Evaluation configuration
EVAL_ENABLE=true
EVAL_NNODES=4
EVAL_N_GPUS_PER_NODE=4
EVAL_ROLLOUT_NAME="sglang"

# Logging
LOGGER='["console","wandb"]'

# Generate run name and experiment name from key parameters
MODEL_NAME=$(basename "$MODEL_PATH")
RUN_NAME="${MODEL_NAME}-${MODEL_REVISION}-lr${LEARNING_RATE}-bs${TRAIN_BATCH_SIZE}-test${TEST_FREQ}-warmup${WARMUP_STYLE}-lr_warmup_steps_ratio${LR_WARMUP_STEPS_RATIO}-ray"

# Environment
VERL_ENVIRONMENT="/capstor/store/cscs/swissai/infra01/reasoning/imgs/projects/verl_swiss:1/env.toml"

# Get node information
nodes=($(scontrol show hostnames "$SLURM_JOB_NODELIST"))
head_node=${nodes[0]}
head_node_ip=$(srun --nodes=1 --ntasks=1 --nodelist=$head_node hostname -i)

# If we detect a space character in the head node IP, convert it to an ipv4 address
if [[ "$head_node_ip" == *" "* ]]; then
    IFS=' ' read -ra ADDR <<<"$head_node_ip"
    if [[ ${#ADDR[0]} -gt 16 ]]; then
        head_node_ip=${ADDR[1]}
    else
        head_node_ip=${ADDR[0]}
    fi
    echo "IPV6 address detected. We split the IPV4 address as $head_node_ip"
fi

ip_head=$head_node_ip:$RAY_PORT
export ip_head
echo "IP Head: $ip_head"

current_datetime=$(date '+%Y-%m-%d_%H-%M-%S')
save_path="$SAVE_FOLDER/$RUN_NAME/$current_datetime"
dataset_path=$DATASET_PATH

mkdir -p "$save_path"
cp "$0" "$save_path/run_apertus_8b_ray.sbatch"

# Build project wheel once on the head node
WHEEL_DIR="$save_path/wheels"
mkdir -p "$WHEEL_DIR"
srun --nodes=1 --ntasks=1 --nodelist=$head_node --container-writable --environment=$VERL_ENVIRONMENT --kill-on-bad-exit=1 --output=$save_path/wheel_build.log --error=$save_path/wheel_build.err \
    bash --norc --noprofile -c "\
set -ex
cd $WORK_DIR
pip wheel . --no-cache-dir --no-deps -w $WHEEL_DIR"
PACKAGE_WHEEL=$(ls -t "$WHEEL_DIR"/*.whl | head -n1)

# Start Ray head node
echo "Starting Ray HEAD at $head_node"
srun --nodes=1 --ntasks=1 --nodelist=$head_node --container-writable --environment=$VERL_ENVIRONMENT --output=$save_path/ray_head.log --error=$save_path/ray_head.err \
    bash --norc --noprofile -c "\
set -ex

export no_proxy=\"0.0.0.0,\$no_proxy\"
export NO_PROXY=\"0.0.0.0,\$NO_PROXY\"

cd $WORK_DIR

# Install dependencies
pip install --no-deps /capstor/store/cscs/swissai/infra01/reasoning/users/nathanrchn/wheels/evaluate-0.4.6-py3-none-any.whl
pip install --no-deps /capstor/store/cscs/swissai/infra01/reasoning/users/nathanrchn/wheels/nltk-3.9.2-py3-none-any.whl
pip install --no-deps /capstor/store/cscs/swissai/infra01/reasoning/users/nathanrchn/wheels/emoji-2.15.0-py3-none-any.whl
pip install --no-deps /capstor/store/cscs/swissai/infra01/reasoning/users/nathanrchn/wheels/syllapy-0.7.2-py3-none-any.whl
pip install --no-deps /capstor/store/cscs/swissai/infra01/reasoning/users/nathanrchn/wheels/langdetect-1.0.9-py3-none-any.whl
pip install --no-deps /capstor/store/cscs/swissai/infra01/reasoning/users/nathanrchn/wheels/immutabledict-4.2.2-py3-none-any.whl
pip install $PACKAGE_WHEEL --no-cache-dir --no-deps --force-reinstall

# Start Ray head
ray start --head --node-ip-address=\"$head_node_ip\" --port=$RAY_PORT \
    --num-cpus \"\${SLURM_CPUS_PER_TASK}\" --num-gpus \"\${SLURM_GPUS_PER_NODE}\" --block" &

# Wait for Ray head to start
sleep 10

# Start Ray worker nodes
worker_num=$((SLURM_JOB_NUM_NODES - 1))
for ((i = 1; i <= worker_num; i++)); do
    node=${nodes[$i]}
    echo "Starting Ray WORKER $i at $node"
    srun --nodes=1 --ntasks=1 --nodelist=$node --container-writable --environment=$VERL_ENVIRONMENT --output=$save_path/ray_worker_${i}.log --error=$save_path/ray_worker_${i}.err \
        bash --norc --noprofile -c "\
set -ex

export no_proxy=\"0.0.0.0,\$no_proxy\"
export NO_PROXY=\"0.0.0.0,\$NO_PROXY\"

cd $WORK_DIR

# Install dependencies
pip install --no-deps /capstor/store/cscs/swissai/infra01/reasoning/users/nathanrchn/wheels/nltk-3.9.2-py3-none-any.whl
pip install --no-deps /capstor/store/cscs/swissai/infra01/reasoning/users/nathanrchn/wheels/emoji-2.15.0-py3-none-any.whl
pip install --no-deps /capstor/store/cscs/swissai/infra01/reasoning/users/nathanrchn/wheels/syllapy-0.7.2-py3-none-any.whl
pip install --no-deps /capstor/store/cscs/swissai/infra01/reasoning/users/nathanrchn/wheels/langdetect-1.0.9-py3-none-any.whl
pip install --no-deps /capstor/store/cscs/swissai/infra01/reasoning/users/nathanrchn/wheels/immutabledict-4.2.2-py3-none-any.whl
pip install $PACKAGE_WHEEL --no-cache-dir --no-deps --force-reinstall

# Start Ray worker
ray start --address \"$ip_head\" \
    --num-cpus \"\${SLURM_CPUS_PER_TASK}\" --num-gpus \"\${SLURM_GPUS_PER_NODE}\" --block" &
    sleep 5
done

# Wait for all Ray workers to connect
sleep 10

# Run the SFT Ray trainer on the head node
PYTHONUNBUFFERED=1 srun --overlap --nodes=1 --ntasks=1 --nodelist=$head_node --container-writable --environment=$VERL_ENVIRONMENT --kill-on-bad-exit=1 --output=$save_path/trainer.log --error=$save_path/trainer.err \
    bash --norc --noprofile -c "\
set -ex

export no_proxy=\"0.0.0.0,\$no_proxy\"
export NO_PROXY=\"0.0.0.0,\$NO_PROXY\"

export NCCL_DEBUG=INFO
export HYDRA_FULL_ERROR=1

cd $WORK_DIR

python -m verl.trainer.sft_trainer_ray \
    data.train_files=$dataset_path/$TRAIN_FILE \
    data.val_files=$dataset_path/$VAL_FILE \
    data.messages_key=$MESSAGES_KEY \
    data.tools_key=$TOOLS_KEY \
    data.enable_thinking_key=$ENABLE_THINKING_KEY \
    data.micro_batch_size_per_gpu=$MICRO_BATCH_SIZE_PER_GPU \
    data.custom_cls.path=$CUSTOM_CLS_PATH \
    data.custom_cls.name=$CUSTOM_CLS_NAME \
    data.max_length=$MAX_LENGTH \
    data.train_batch_size=$TRAIN_BATCH_SIZE \
    data.val_batch_size=$VAL_BATCH_SIZE \
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
    checkpoint.save_contents='[\"model\",\"optimizer\",\"extra\",\"hf_model\"]' \
    trainer.default_local_dir=$save_path \
    trainer.project_name=$PROJECT_NAME \
    trainer.experiment_name=$RUN_NAME \
    trainer.test_freq=$TEST_FREQ \
    trainer.total_training_steps=$TOTAL_TRAINING_STEPS \
    trainer.total_epochs=$TOTAL_EPOCHS \
    trainer.logger=$LOGGER \
    trainer.nnodes=$SLURM_NNODES \
    trainer.n_gpus_per_node=$SLURM_GPUS_PER_NODE \
    eval.enable=$EVAL_ENABLE \
    eval.rollout.name=$EVAL_ROLLOUT_NAME \
    eval.rollout.nnodes=$EVAL_NNODES \
    eval.rollout.n_gpus_per_node=$EVAL_N_GPUS_PER_NODE \
    eval.benchmarks='[\"gsm8k\",\"gsm8k_thinking\",\"math_500\",\"averitec\",\"ifeval\",\"ifbench\",\"humaneval\",\"humaneval_thinking\",\"mbpp_thinking\"]'" &

wait -n
scancel $SLURM_JOB_ID

echo "[FINISHED]"
