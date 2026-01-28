"""
Dataset preparation for Apertus SFT training.

Loads training data and creates rollout evaluation datasets from HuggingFace:
- GSM8K, MATH-500, Omni-MATH, OlympiadBench (Math reasoning)
- AVeriTeC (Fact verification)
- IFEval, IFBench (Instruction following)
- HumanEval, MBPP (Coding)
"""
import os
from json import loads, dumps
from datasets import load_from_disk
from transformers import AutoTokenizer

INPUT_DATASET_PATH = "/capstor/store/cscs/swissai/infra01/reasoning/data/sft_1.1/mixtures-linearised/full-1"
OUTPUT_DATASET_PATH = "/iopsstor/scratch/cscs/nathanrchn/apertus-full-1"
MODEL_PATH = "swiss-ai/Apertus-8B-Instruct-2509"

VAL_SPLIT_SIZE = 512


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

# Grammar to force tool use for display_answers
ANSWERS_TOOL_GRAMMAR = """%llguidance {}
start: (text_or_thinking)? tool_calls
text_or_thinking: TEXT | (<|inner_prefix|> TEXT <|inner_suffix|>)

tool_calls: <|tools_prefix|> %json {
    "type": "array",
    "minItems": 1,
    "items": {
        "type": "object",
        "properties": {
            "display_answers": {
                "type": "object",
                "properties": {
                    "answers": {
                        "type": "array",
                        "items": {"type": "string"}
                    }
                },
                "required": ["answers"]
            }
        },
        "required": ["display_answers"]
    }
} <|tools_suffix|>

TEXT: /(.|\n)+/
"""


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
    remove_columns=list(set(dataset.column_names) - set(["messages", "tools", "enable_thinking"]))
)

train_dataset = dataset.select(range(len(dataset) - VAL_SPLIT_SIZE - 1))
val_offset = len(dataset) - VAL_SPLIT_SIZE
val_dataset = dataset.select(range(val_offset, val_offset + VAL_SPLIT_SIZE))

print(train_dataset)
print(val_dataset)

train_dataset.to_parquet(os.path.join(OUTPUT_DATASET_PATH, "train.parquet"))
val_dataset.to_parquet(os.path.join(OUTPUT_DATASET_PATH, "val.parquet"))
