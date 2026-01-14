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
from transformers import AutoTokenizer
from datasets import load_from_disk, load_dataset, concatenate_datasets

# Configuration
INPUT_DATASET_PATH = "/capstor/store/cscs/swissai/infra01/reasoning/data/sft_1.1/mixtures-linearised/full-1"
OUTPUT_DATASET_PATH = "/iopsstor/scratch/cscs/nathanrchn/apertus-full-1"
MODEL_PATH = "swiss-ai/Apertus-8B-Instruct-2509"

VAL_SPLIT_SIZE = 512

# =============================================================================
# Load Main Training Dataset
# =============================================================================

dataset = load_from_disk(INPUT_DATASET_PATH)["train"]
dataset = dataset.shuffle(seed=42)

tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH)

# =============================================================================
# Tool Definitions
# =============================================================================

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


# =============================================================================
# Training Data Processing
# =============================================================================

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

# =============================================================================
# Rollout Dataset Helpers
# =============================================================================

# System prompt for AVeriTeC fact verification
AVERITEC_GUIDE = "Display one of the following answers: 'Supported', 'Refuted', 'Conflicting Evidence/Cherrypicking', or 'Not Enough Evidence'."


def make_rollout_sample(
    question: str,
    answer: str = "",
    task_id: str = "",
    task_name: str = "",
    system_prompt: str = "",
    max_new_tokens: int = 2048,
    with_tools: bool = True,
    grammar: str = None,
    extra_sampling_params: dict = None,
    enable_thinking: bool = False,
    kwargs: dict = None,
) -> dict:
    """Create a standard rollout sample."""
    sampling_params = {
        "skip_special_tokens": False,
        "max_new_tokens": max_new_tokens,
    }
    if extra_sampling_params:
        sampling_params.update(extra_sampling_params)

    if grammar:
        sampling_params["ebnf"] = grammar
    elif with_tools:
        sampling_params["ebnf"] = ANSWERS_TOOL_GRAMMAR

    rollout_params = {
        "id": task_id,
        "task_name": task_name,
        "answer": answer,
        "sampling_params": sampling_params,
        "use_tool": with_tools,
        "kwargs": kwargs or {},
    }

    return {
        "messages": dumps([
            {"role": "system", "content": {"text": system_prompt}},
            {"role": "user", "content": {"parts": [{"type": "text", "text": question}]}},
        ]),
        "tools": dumps([ANSWERS_TOOL]) if with_tools else "",
        "rollout_params": dumps(rollout_params),
        "enable_thinking": enable_thinking,
    }


# =============================================================================
# Math Evaluation Datasets
# =============================================================================

# GSM8K (HuggingFace)
def gsm8k_to_standard_format(x):
    answer = x["answer"].split("#### ")[-1].replace(",", "").strip()
    return make_rollout_sample(
        question=x["question"],
        answer=answer,
        task_id="gsm8k",
        task_name="gsm8k",
        extra_sampling_params={"temperature": 0.8},
    )


gsm8k_dataset = load_dataset("openai/gsm8k", name="main", split="test").map(
    gsm8k_to_standard_format,
    num_proc=64,
    remove_columns=["question", "answer"]
)

# GSM8K (Thinking)
def gsm8k_thinking_to_standard_format(x):
    answer = x["answer"].split("#### ")[-1].replace(",", "").strip()
    return make_rollout_sample(
        question=x["question"],
        answer=answer,
        task_id="gsm8k",
        task_name="gsm8k_thinking",
        enable_thinking=True,
        max_new_tokens=4096,
        extra_sampling_params={"temperature": 0.8},
    )

gsm8k_thinking_dataset = load_dataset("openai/gsm8k", name="main", split="test").map(
    gsm8k_thinking_to_standard_format,
    num_proc=64,
    remove_columns=["question", "answer"]
)

# MATH-500 (HuggingFace)
def math_500_to_standard_format(x):
    return make_rollout_sample(
        question=x["problem"],
        answer=x["answer"],
        task_id="math_500",
        task_name="math_500",
        extra_sampling_params={"temperature": 0.8},
    )


math_500_dataset = load_dataset("HuggingFaceH4/MATH-500", split="test", trust_remote_code=True).map(
    math_500_to_standard_format,
    num_proc=64,
    remove_columns=["problem", "solution", "answer", "subject", "level", "unique_id"]
)

# # Omni-MATH (HuggingFace)
# def omni_math_to_standard_format(x):
#     # Determine difficulty level for task name
#     difficulty = x.get("difficulty", 5)
#     if difficulty <= 3:
#         task_name = "omni_math_easy"
#     elif difficulty <= 6:
#         task_name = "omni_math_med"
#     else:
#         task_name = "omni_math_hard"

#     return make_rollout_sample(
#         question=x["problem"],
#         answer=x["answer"],
#         task_id=task_name,
#         task_name=task_name,
#     )


# omni_math_dataset = load_dataset("KbsdJames/Omni-MATH", split="test", trust_remote_code=True).map(
#     omni_math_to_standard_format,
#     num_proc=64,
#     remove_columns=["domain", "difficulty", "problem", "solution", "answer", "source"]
# )

# # OlympiadBench (HuggingFace - text-only English math)
# def olympiad_bench_to_standard_format(x):
#     # Handle list answers
#     answer = x["final_answer"]
#     if isinstance(answer, list):
#         answer = ", ".join(str(a) for a in answer)
#     return make_rollout_sample(
#         question=x["question"],
#         answer=str(answer),
#         task_id="olympiad_bench",
#         task_name="olympiad_bench",
#     )


# olympiad_bench_dataset = load_dataset(
#     "Hothan/OlympiadBench",
#     "OE_TO_maths_en_COMP",
#     split="train",
#     trust_remote_code=True
# ).map(
#     olympiad_bench_to_standard_format,
#     num_proc=64,
#     remove_columns=[
#         "id", "question", "solution", "final_answer", "context",
#         "image_1", "image_2", "image_3", "image_4", "image_5",
#         "image_6", "image_7", "image_8", "image_9", "modality",
#         "difficulty", "is_multiple_answer", "unit", "answer_type",
#         "error", "question_type", "subfield", "subject", "language"
#     ]
# )

# =============================================================================
# Fact Verification Datasets
# =============================================================================

# AVeriTeC (HuggingFace)
def averitec_to_standard_format(x):
    return make_rollout_sample(
        question=x["claim"],
        answer=x["label"],
        task_id="averitec",
        task_name="averitec",
        system_prompt=AVERITEC_GUIDE,
        max_new_tokens=512,
        extra_sampling_params={"temperature": 0.8},
    )


averitec_dataset = load_dataset("pminervini/averitec", split="dev", trust_remote_code=True).map(
    averitec_to_standard_format,
    num_proc=64,
    remove_columns=[
        "cached_original_claim_url", "speaker", "required_reannotation",
        "reporting_source", "label", "claim_types", "fact_checking_article",
        "fact_checking_strategies", "claim", "justification", "location_ISO_code",
        "original_claim_url", "questions", "claim_date"
    ]
)

# =============================================================================
# Instruction Following Datasets
# =============================================================================

def ifeval_to_standard_format(x):
    return make_rollout_sample(
        question=x["prompt"],
        task_id="ifeval",
        task_name="ifeval",
        with_tools=False,
        extra_sampling_params={"temperature": 0.8},
        kwargs={
            "instruction_id_list": x["instruction_id_list"],
            "instruction_kwargs": x["kwargs"],
        }
    )


ifeval_dataset = load_dataset("google/IFEval", split="train").map(
    ifeval_to_standard_format,
    num_proc=64,
    remove_columns=["key", "prompt", "instruction_id_list", "kwargs"]
)


def ifbench_to_standard_format(x):
    return make_rollout_sample(
        question=x["prompt"],
        task_id="ifbench",
        task_name="ifbench",
        with_tools=False,
        extra_sampling_params={"temperature": 0.8},
        kwargs={
            "instruction_id_list": x["instruction_id_list"],
            "instruction_kwargs": x["kwargs"],
        }
    )


ifbench_dataset = load_dataset("allenai/IFBench_test", split="train").map(
    ifbench_to_standard_format,
    num_proc=64,
    remove_columns=["key", "prompt", "instruction_id_list", "kwargs"]
)

# =============================================================================
# Coding Datasets
# =============================================================================

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
            "task_name": "humaneval",
            "test": x["test"],
            "prompt": x["prompt"],
            "entry_point": x["entry_point"],
            "sampling_params": {
                "temperature": 0.8,
                "top_p": 0.95,
                "max_new_tokens": 2048,
                "n": 10,
                "stop": ["```"]
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
            "id": "humaneval_thinking",
            "task_id": x["task_id"],
            "task_name": "humaneval_thinking",
            "test": x["test"],
            "prompt": x["prompt"],
            "entry_point": x["entry_point"],
            "sampling_params": {
                "temperature": 0.8,
                "top_p": 0.95,
                "max_new_tokens": 16384,
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
            "id": "mbpp_thinking",
            "task_id": x["task_id"],
            "task_name": "mbpp_thinking",
            "prompt": x["prompt"],
            "test_imports": x["test_imports"],
            "test_list": x["test_list"],
            "sampling_params": {
                "temperature": 0.8,
                "top_p": 0.95,
                "max_new_tokens": 16384,
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

# =============================================================================
# Combine All Rollout Datasets
# =============================================================================

rollout_dataset = concatenate_datasets([
    # Math reasoning
    gsm8k_dataset,
    gsm8k_thinking_dataset,
    math_500_dataset,
    # omni_math_dataset,
    # olympiad_bench_dataset,
    # Fact verification
    averitec_dataset,
    # Instruction following
    ifeval_dataset,
    ifbench_dataset,
    # Coding
    humaneval_dataset,
    humaneval_thinking_dataset,
    # mbpp_thinking_dataset,
])

# =============================================================================
# Output
# =============================================================================

print("=" * 60)
print("Dataset Summary")
print("=" * 60)
print(f"\nTraining dataset: {len(train_dataset)} samples")
print(f"Validation dataset: {len(val_dataset)} samples")
print(f"Rollout dataset: {len(rollout_dataset)} samples")
print("\nRollout dataset breakdown:")
print(f"  - GSM8K: {len(gsm8k_dataset)}")
print(f"  - GSM8K (Thinking): {len(gsm8k_thinking_dataset)}")
print(f"  - MATH-500: {len(math_500_dataset)}")
# print(f"  - Omni-MATH: {len(omni_math_dataset)}")
# print(f"  - OlympiadBench: {len(olympiad_bench_dataset)}")
print(f"  - AVeriTeC: {len(averitec_dataset)}")
print(f"  - IFEval: {len(ifeval_dataset)}")
print(f"  - IFBench: {len(ifbench_dataset)}")
print(f"  - HumanEval: {len(humaneval_dataset)}")
print(f"  - HumanEval (Thinking): {len(humaneval_thinking_dataset)}")
# print(f"  - MBPP (Thinking): {len(mbpp_thinking_dataset)}")

# Save datasets
train_dataset.to_parquet(os.path.join(OUTPUT_DATASET_PATH, "train.parquet"))
val_dataset.to_parquet(os.path.join(OUTPUT_DATASET_PATH, "val.parquet"))
rollout_dataset.to_parquet(os.path.join(OUTPUT_DATASET_PATH, "rollout.parquet"))

print(f"\nDatasets saved to: {OUTPUT_DATASET_PATH}")
