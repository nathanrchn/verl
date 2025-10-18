import os
from json import loads, dumps
from transformers import AutoTokenizer
from datasets import load_from_disk, load_dataset

INPUT_DATASET_PATH = "/capstor/store/cscs/swissai/infra01/reasoning/data/sft_1.1/mixtures-linearised/apertus-sft-code-1"
OUTPUT_DATASET_PATH = "/iopsstor/scratch/cscs/nathanrchn/apertus-sft-code-1"
MODEL_PATH = "swiss-ai/Apertus-8B-Instruct-2509"

TRAIN_SPLIT_SIZE = 531937
VAL_SPLIT_SIZE = 32

dataset = load_from_disk(INPUT_DATASET_PATH)["train"]

dataset = dataset.shuffle(seed=42)

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


def humaneval_to_standard_format(x):
    o = {}
    o["messages"] = dumps(
        [
            {"role": "system", "content": {"text": ""}},
            {
                "role": "user",
                "content": {
                    "parts": [
                        {
                            "type": "text",
                            "text": f"Write a solution to the following problem and make sure that it passes the tests:\n```python\n{x['prompt']}\n```\n",
                        }
                    ]
                },
            },
            {
                "role": "assistant",
                "content": {
                    "blocks": [
                        {"type": "response", "text": f"Here is the completed function:\n```python\n{x['prompt']}\n"}
                    ]
                },
            },
        ]
    )
    o["tools"] = dumps([])
    o["rollout_params"] = dumps(
        {
            "id": "humaneval",
            "task_id": x["task_id"],
            "test": x["test"],
            "prompt": x["prompt"],
            "entry_point": x["entry_point"],
            "sampling_params": {
                "temperature": 0.8,
                "top_p": 0.95,
                "max_new_tokens": 1024,
                "stop": ["\nclass", "\ndef", "\n#", "\nif", "\nprint"],
                "n": 10,
            },
            "apply_chat_template_kwargs": {"continue_final_message": True},
        }
    )
    o["enable_thinking"] = False
    return o


humaneval_dataset = load_dataset("openai/openai_humaneval", split="test").map(
    humaneval_to_standard_format,
    num_proc=64,
    remove_columns=["task_id", "prompt", "canonical_solution", "test", "entry_point"],
)

rollout_dataset = humaneval_dataset

print(train_dataset)
print(val_dataset)
print(rollout_dataset)

train_dataset.to_parquet(os.path.join(OUTPUT_DATASET_PATH, "train.parquet"))
val_dataset.to_parquet(os.path.join(OUTPUT_DATASET_PATH, "val.parquet"))
rollout_dataset.to_parquet(os.path.join(OUTPUT_DATASET_PATH, "rollout.parquet"))
