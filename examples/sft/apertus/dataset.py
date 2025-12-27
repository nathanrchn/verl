import os
from json import loads, dumps
from transformers import AutoTokenizer
from datasets import load_from_disk, load_dataset, concatenate_datasets

INPUT_DATASET_PATH = "/capstor/store/cscs/swissai/infra01/reasoning/data/sft_1.1/mixtures-linearised/format-following-3"
OUTPUT_DATASET_PATH = "/iopsstor/scratch/cscs/nathanrchn/apertus-format-following-3"
MODEL_PATH = "swiss-ai/Apertus-8B-Instruct-2509"

VAL_SPLIT_SIZE = 128

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


dataset = dataset.map(convert_to_standard_format, num_proc=64, remove_columns=list(set(dataset.column_names) - set(["messages", "tools", "enable_thinking"])))


train_dataset = dataset.select(range(len(dataset) - VAL_SPLIT_SIZE - 1))

val_offset = len(dataset) - VAL_SPLIT_SIZE
val_dataset = dataset.select(range(val_offset, val_offset + VAL_SPLIT_SIZE))


# def filter_messages(x):
#     input_ids = tokenizer.apply_chat_template(
#         loads(x["messages"]),
#         tools=loads(x["tools"]) if x["tools"] is not None and x["tools"] != "" else None,
#         enable_thinking=x["enable_thinking"],
#         add_generation_prompt=True,
#     )

#     return len(input_ids) < ROLLOUT_MAX_LENGTH


# def remove_last_message(x):
#     x["messages"] = dumps(loads(x["messages"])[:-1])
#     return x


# def add_rollout_params(x):
#     x["rollout_params"] = dumps({"id": "default"})
#     return x


# rollout_offset = val_offset + VAL_SPLIT_SIZE + 1
# rollout_dataset = (
#     dataset.skip(rollout_offset)
#     .map(
#         convert_to_standard_format,
#         num_proc=64,
#         remove_columns=list(set(dataset.column_names) - set(["messages", "tools", "enable_thinking"])),
#     )
#     .filter(filter_messages, num_proc=64)
#     .select(range(ROLLOUT_SPLIT_SIZE))
#     .map(remove_last_message, num_proc=64)
#     .map(add_rollout_params, num_proc=64)
# )


def gsm8k_to_standard_format(x):
    o = {}
    o["messages"] = dumps(
        [
            {"role": "system", "content": {"text": ""}},
            {"role": "user", "content": {"parts": [{"type": "text", "text": x["question"]}]}},
        ]
    )
    o["tools"] = dumps([ANSWERS_TOOL])
    o["rollout_params"] = dumps({
        "id": "gsm8k",
        "answer": x["answer"].split("#### ")[-1].replace(",", "").strip(),
        "sampling_params": {
            "skip_special_tokens": False,
            "max_new_tokens": 2048,
        },
    })
    o["enable_thinking"] = False
    return o


gsm8k_dataset = load_dataset("openai/gsm8k", name="main", split="test").map(
    gsm8k_to_standard_format, num_proc=64, remove_columns=["question", "answer"]
)

def ifbench_to_standard_format(x):
    o = {}
    o["messages"] = dumps(
        [
            {"role": "system", "content": {"text": ""}},
            {"role": "user", "content": {"parts": [{"type": "text", "text": x["prompt"]}]}},
        ]
    )
    o["tools"] = ""
    o["rollout_params"] = dumps({
        "id": "ifbench",
        "instruction_id_list": x["instruction_id_list"],
        "kwargs": x["kwargs"],
        "sampling_params": {
            "temperature": 0,
            "max_new_tokens": 4096,
        },
    })
    o["enable_thinking"] = True
    return o

ifbench_dataset = load_dataset("allenai/IFBench_test", split="train").map(
    ifbench_to_standard_format, num_proc=64, remove_columns=["key", "prompt", "instruction_id_list", "kwargs"]
)

rollout_dataset = concatenate_datasets([gsm8k_dataset, ifbench_dataset])

print(train_dataset)
print(val_dataset)
print(rollout_dataset)

train_dataset.to_parquet(os.path.join(OUTPUT_DATASET_PATH, "train.parquet"))
val_dataset.to_parquet(os.path.join(OUTPUT_DATASET_PATH, "val.parquet"))
rollout_dataset.to_parquet(os.path.join(OUTPUT_DATASET_PATH, "rollout.parquet"))
