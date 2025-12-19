import os
from json import loads, dumps
from transformers import AutoTokenizer
from datasets import load_from_disk, load_dataset, concatenate_datasets

INPUT_DATASET_PATH = "/capstor/store/cscs/swissai/infra01/reasoning/data/sft_1.1/mixtures-linearised/apertus-sft-code-1"
OUTPUT_DATASET_PATH = "/iopsstor/scratch/cscs/nathanrchn/apertus-sft-code-1"
MODEL_PATH = "swiss-ai/Apertus-8B-Instruct-2509"

TRAIN_SPLIT_SIZE = 531905
VAL_SPLIT_SIZE = 64
SEQ_LENGTH = 8192

dataset = load_from_disk(INPUT_DATASET_PATH)["train"]

dataset = dataset.shuffle(seed=42)


def remove_reasoning(x):
    for message in x["messages"]:
        if message["role"] == "developer":
            message["content"]["has_thinking"] = False
        elif message["role"] == "assistant":
            blocks = []
            for block in message["content"]["blocks"]:
                if block["type"] != "thoughts":
                    blocks.append(block)
            message["content"]["blocks"] = blocks
    return x


# dataset = dataset.map(remove_reasoning, num_proc=64)

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
                        if call["name"] == "display_answers":
                            uses_answer_tool = True
                            break
                if uses_answer_tool:
                    break
            if uses_answer_tool:
                break

    if not uses_answer_tool:
        return x

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
    developer_message["content"]["tools"] = dumps(tools) if tools else ""
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


dataset = dataset.map(
    convert_to_standard_format,
    num_proc=64,
    remove_columns=list(set(dataset.column_names) - set(["messages", "tools", "enable_thinking"])),
)

def remove_excess_tokens(x):
    input_ids = tokenizer.apply_chat_template(
        loads(x["messages"]),
        tools=loads(x["tools"]) if x["tools"] is not None and x["tools"] != "" else None,
        enable_thinking=x["enable_thinking"],
    )

    if len(input_ids) > SEQ_LENGTH:
        for message in loads(x["messages"]):
            if message["role"] == "developer":
                message["content"]["has_thinking"] = False
            elif message["role"] == "assistant":
                blocks = []
                for block in message["content"]["blocks"]:
                    if block["type"] != "thoughts":
                        blocks.append(block)
                message["content"]["blocks"] = blocks

    return x

dataset = dataset.map(remove_excess_tokens, num_proc=256)

train_dataset = dataset.select(range(TRAIN_SPLIT_SIZE))

val_offset = TRAIN_SPLIT_SIZE + 1
val_dataset = dataset.select(range(val_offset, val_offset + VAL_SPLIT_SIZE))


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
                "stop": ["\nclass", "\ndef", "\n#", "\nif", "\nprint", "\n```", "<|assistant_end|>", "</s>"],
                "n": 10,
            },
            "apply_chat_template_kwargs": {"continue_assistant_message": True},
        }
    )
    o["enable_thinking"] = False
    return o


humaneval_dataset = load_dataset("openai/openai_humaneval", split="test").map(
    humaneval_to_standard_format,
    num_proc=64,
    remove_columns=["task_id", "prompt", "canonical_solution", "test", "entry_point"],
)


def humaneval_thinking_to_standard_format(x):
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
        ]
    )
    o["tools"] = dumps([])
    o["rollout_params"] = dumps(
        {
            "id": "default,humaneval_thinking",
            "task_id": x["task_id"],
            "test": x["test"],
            "prompt": x["prompt"],
            "entry_point": x["entry_point"],
            "sampling_params": {
                "temperature": 0.8,
                "top_p": 0.95,
                "max_new_tokens": 32768,
                "n": 10,
                "skip_special_tokens": False,
            },
            "apply_chat_template_kwargs": {},
        }
    )
    o["enable_thinking"] = True
    return o


humaneval_thinking_dataset = load_dataset("openai/openai_humaneval", split="test").map(
    humaneval_thinking_to_standard_format,
    num_proc=64,
    remove_columns=["task_id", "prompt", "canonical_solution", "test", "entry_point"],
)

def mbpp_thinking_to_standard_format(x):
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
                            "text": f"You are an expert Python programmer, and here is your task:\n{x['prompt']}\nYour code should pass these tests:\n{'\n'.join(x['test_list'])}",
                        }
                    ]
                },
            },
        ]
    )
    o["tools"] = dumps([])
    o["rollout_params"] = dumps(
        {
            "id": "default,mbpp_thinking",
            "task_id": x["task_id"],
            "prompt": x["prompt"],
            "test_imports": x["test_imports"],
            "test_list": x["test_list"],
            "sampling_params": {
                "temperature": 0.8,
                "top_p": 0.95,
                "max_new_tokens": 32768,
                "skip_special_tokens": False,
            },
            "apply_chat_template_kwargs": {},
        }
    )
    o["enable_thinking"] = True
    return o

mbpp_thinking_dataset = load_dataset("google-research-datasets/mbpp", name="sanitized", split="test").map(
    mbpp_thinking_to_standard_format,
    num_proc=64,
    remove_columns=["source_file", "task_id", "prompt", "code", "test_imports", "test_list"],
)

rollout_dataset = humaneval_dataset # concatenate_datasets([humaneval_thinking_dataset, mbpp_thinking_dataset])

print(train_dataset)
print(val_dataset)
print(rollout_dataset)

train_dataset.to_parquet(os.path.join(OUTPUT_DATASET_PATH, "train.parquet"))
val_dataset.to_parquet(os.path.join(OUTPUT_DATASET_PATH, "val.parquet"))
rollout_dataset.to_parquet(os.path.join(OUTPUT_DATASET_PATH, "rollout.parquet"))
