import os
from datasets import load_from_disk

INPUT_DATASET_PATH = (
    "/capstor/store/cscs/swissai/infra01/posttrain_data/06_sft_mixtures_newformat_linearised/apertus-sft-mixture-8e"
)
OUTPUT_DATASET_PATH = "/iopsstor/scratch/cscs/nathanrchn/debug-apertus-sft-mixture-8e-with-eval"

TRAIN_SPLIT_SIZE = 1_048_576
VAL_SPLIT_SIZE = 2048
ROLLOUT_SPLIT_SIZE = 256

dataset = load_from_disk(INPUT_DATASET_PATH)["train"]

dataset = dataset.shuffle(seed=42)


def convert_to_standard_format(x):
    messages = []
    developer_message = None
    for message in x["messages"]:
        if message["role"] == "developer":
            assert developer_message is None
            developer_message = message
        else:
            messages.append(message)

    x["messages"] = messages
    x["tools"] = developer_message["content"]["tools"]
    x["enable_thinking"] = developer_message["content"]["has_thinking"]
    return x


train_dataset = dataset.select(range(TRAIN_SPLIT_SIZE)).map(convert_to_standard_format, num_proc=64)

val_offset = TRAIN_SPLIT_SIZE + 1
val_dataset = dataset.select(range(val_offset, val_offset + VAL_SPLIT_SIZE)).map(
    convert_to_standard_format, num_proc=64
)


def remove_last_message(x):
    x["messages"] = x["messages"][:-1]
    return x


rollout_offset = val_offset + VAL_SPLIT_SIZE + 1
rollout_dataset = (
    dataset.select(range(rollout_offset, rollout_offset + ROLLOUT_SPLIT_SIZE))
    .map(convert_to_standard_format, num_proc=64)
    .map(remove_last_message, num_proc=64)
)

print(train_dataset)
print(val_dataset)
print(rollout_dataset)

train_dataset.to_parquet(os.path.join(OUTPUT_DATASET_PATH, "train.parquet"))
val_dataset.to_parquet(os.path.join(OUTPUT_DATASET_PATH, "val.parquet"))
rollout_dataset.to_parquet(os.path.join(OUTPUT_DATASET_PATH, "rollout.parquet"))
