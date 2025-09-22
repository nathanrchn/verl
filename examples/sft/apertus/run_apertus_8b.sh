set -x

if [ "$#" -lt 2 ]; then
    echo "Usage: run_apertus_8b.sh <nproc_per_node> <dataset_path> <save_path> [other_configs...]"
    exit 1
fi

nproc_per_node=$1
dataset_path=$2
save_path=$3

shift 3

torchrun --nnodes=1 --nproc_per_node=$nproc_per_node \
    -m verl.trainer.sft_trainer \
    data.train_files=$dataset_path/train.parquet \
    data.val_files=$dataset_path/val.parquet \
    +data.rollout_files=$dataset_path/rollout.parquet \
    +data.multiturn.enable=true \
    +data.multiturn.messages_key=messages \
    data.micro_batch_size_per_gpu=4 \
    data.custom_cls.path=verl/utils/dataset/multiturn_sft_dataset.py \
    data.custom_cls.name=ApertusSFTDataset \
    +data.apply_chat_template_kwargs.truncation=true \
    model.path=swiss-ai/Apertus-8B-2509 \
    model.tokenizer_path=swiss-ai/Apertus-8B-Instruct-2509 \
    model.use_remove_padding=true \
    trainer.default_local_dir=$save_path \
    trainer.project_name=apertus-sft \
    trainer.experiment_name=apertus-sft-8b-2509 \
    trainer.test_freq=1 \
    trainer.logger='["console","wandb"]' $@
