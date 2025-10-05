import os
from json import loads, dumps
from transformers import AutoTokenizer
from datasets import load_from_disk, load_dataset, concatenate_datasets

INPUT_DATASET_PATH = (
    "/capstor/store/cscs/swissai/infra01/posttrain_data/06_sft_mixtures_newformat_linearised/apertus-sft-mixture-8e"
)
OUTPUT_DATASET_PATH = "/iopsstor/scratch/cscs/nathanrchn/debug-apertus-sft-mixture-8e-with-eval-full"
MODEL_PATH = "swiss-ai/Apertus-8B-Instruct-2509"

TRAIN_SPLIT_SIZE = 3_938_000
VAL_SPLIT_SIZE = 2048
ROLLOUT_SPLIT_SIZE = 256
ROLLOUT_MAX_LENGTH = 512

dataset = load_from_disk(INPUT_DATASET_PATH)["train"]

dataset = dataset.shuffle(seed=42)

# def uses_answer_tool(x):
#     for message in x["messages"]:
#         if message["role"] == "assistant":
#             for block in message["content"]["blocks"]:
#                 if block["type"] == "tool_calls":
#                     for call in block["calls"]:
#                         if call["name"] == "display_answers":
#                             return True
#     return False

# dataset = dataset.filter(uses_answer_tool, num_proc=64)

# print(dataset)

# def has_answer_tool(x):
#     for message in x["messages"]:
#         if message["role"] == "developer":
#             tools_str = message["content"]["tools"]
#             tools = loads(tools_str) if tools_str is not None and tools_str != "" else []
#             for tool in tools:
#                 if tool["name"] == "display_answers":
#                     return True
#     return False

# dataset = dataset.filter(has_answer_tool, num_proc=64)

# print(dataset)
# exit()

tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH)

ANSWERS_TOOL = {
    "name": "display_answers",
    "description": "Display the answers to the user",
    "parameters": {
        "type": "object",
        "properties": {
            "answers": {
                "type": "array",
                "items": {"type": "string"},
                "description": "The answers to the user",
            },
        },
        "required": ["answers"],
    },
}


def add_answer_tool_if_missing(x):
    uses_answer_tool = False
    developer_message = None
    for message in x["messages"]:
        if message["role"] == "developer":
            assert developer_message is None
            developer_message = message
        elif message["role"] == "assistant":
            for block in message["content"]["blocks"]:
                if block["type"] == "tool_calls":
                    for call in block["calls"]:
                        if call["name"] == "display_answer":
                            uses_answer_tool = True
                            break
                if uses_answer_tool:
                    break
            if uses_answer_tool:
                break

    assert developer_message is not None
    tools_str = developer_message["content"]["tools"]
    tools = loads(tools_str) if tools_str is not None and tools_str != "" else []
    has_answer_tool = False
    for tool in tools:
        if tool["name"] == "display_answers":
            has_answer_tool = True
            break
    if not has_answer_tool:
        tools.append(ANSWERS_TOOL)
    x["tools"] = dumps(tools) if tools else ""
    return x


dataset = dataset.map(add_answer_tool_if_missing, num_proc=64)


def convert_to_standard_format(x):
    messages = []
    developer_message = None
    for message in x["messages"]:
        if message["role"] == "developer":
            assert developer_message is None
            developer_message = message
        else:
            messages.append(message)

    o = {}
    o["messages"] = dumps(messages)
    o["tools"] = developer_message["content"]["tools"]
    o["enable_thinking"] = developer_message["content"]["has_thinking"]
    return o


train_dataset = dataset.select(range(TRAIN_SPLIT_SIZE)).map(
    convert_to_standard_format,
    num_proc=64,
    remove_columns=list(set(dataset.column_names) - set(["messages", "tools", "enable_thinking"])),
)

val_offset = TRAIN_SPLIT_SIZE + 1
val_dataset = dataset.select(range(val_offset, val_offset + VAL_SPLIT_SIZE)).map(
    convert_to_standard_format,
    num_proc=64,
    remove_columns=list(set(dataset.column_names) - set(["messages", "tools", "enable_thinking"])),
)


def filter_messages(x):
    input_ids = tokenizer.apply_chat_template(
        loads(x["messages"]),
        tools=loads(x["tools"]) if x["tools"] is not None and x["tools"] != "" else None,
        enable_thinking=x["enable_thinking"],
        add_generation_prompt=True,
    )

    return len(input_ids) < ROLLOUT_MAX_LENGTH


def remove_last_message(x):
    x["messages"] = dumps(loads(x["messages"])[:-1])
    return x


def add_rollout_params(x):
    x["rollout_params"] = dumps({"id": "default"})
    return x


rollout_offset = val_offset + VAL_SPLIT_SIZE + 1
rollout_dataset = (
    dataset.skip(rollout_offset)
    .map(
        convert_to_standard_format,
        num_proc=64,
        remove_columns=list(set(dataset.column_names) - set(["messages", "tools", "enable_thinking"])),
    )
    .filter(filter_messages, num_proc=64)
    .select(range(ROLLOUT_SPLIT_SIZE))
    .map(remove_last_message, num_proc=64)
    .map(add_rollout_params, num_proc=64)
)


def gsm8k_to_standard_format(x):
    o = {}
    o["messages"] = dumps(
        [
            {"role": "system", "content": {"text": ""}},
            {"role": "user", "content": {"parts": [{"type": "text", "text": x["question"]}]}},
        ]
    )
    o["tools"] = dumps([ANSWERS_TOOL])
    o["rollout_params"] = dumps({"id": "gsm8k", "answer": x["answer"].split("#### ")[-1]})
    o["enable_thinking"] = False
    return o


gsm8k_dataset = load_dataset("openai/gsm8k", name="main", split="test").map(
    gsm8k_to_standard_format, num_proc=64, remove_columns=["question", "answer"]
)

rollout_dataset = concatenate_datasets([rollout_dataset, gsm8k_dataset])

print(train_dataset)
print(val_dataset)
print(rollout_dataset)

train_dataset.to_parquet(os.path.join(OUTPUT_DATASET_PATH, "train.parquet"))
val_dataset.to_parquet(os.path.join(OUTPUT_DATASET_PATH, "val.parquet"))
rollout_dataset.to_parquet(os.path.join(OUTPUT_DATASET_PATH, "rollout.parquet"))
