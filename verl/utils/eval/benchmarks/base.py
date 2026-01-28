"""
Base classes for evaluation benchmarks.
"""
from dataclasses import dataclass, field
from typing import Any, Callable, Optional


@dataclass
class BenchmarkConfig:
    """
    Complete benchmark configuration.
    All settings defined here, not in user config.
    """

    name: str
    task_type: str  # "math", "code", "ifeval", "fact_verification"
    dataset_name: str  # HuggingFace dataset name
    dataset_split: str = "test"
    dataset_subset: Optional[str] = None  # e.g., "main" for gsm8k
    max_samples: int = -1  # -1 for all
    sampling_params: dict = field(default_factory=lambda: {"temperature": 0.8})
    max_new_tokens: int = 2048
    n_samples: int = 1  # For pass@k
    enable_thinking: bool = False
    with_tools: bool = True
    system_prompt: str = ""

    # Each benchmark file provides its own transform_fn and compute_metrics_fn
    transform_fn: Optional[Callable] = None
    compute_metrics_fn: Optional[Callable] = None


def make_rollout_sample(
    question: str,
    answer: str,
    task_id: str,
    task_name: str,
    extra_sampling_params: Optional[dict] = None,
    system_prompt: str = "",
    enable_thinking: bool = False,
    with_tools: bool = True,
) -> dict:
    """
    Create a standard rollout sample format for evaluation.

    Args:
        question: Input question/prompt
        answer: Reference answer
        task_id: Unique task identifier
        task_name: Benchmark name
        extra_sampling_params: Additional sampling parameters
        system_prompt: System prompt to use
        enable_thinking: Whether to enable thinking mode
        with_tools: Whether to enable tool use

    Returns:
        Dictionary in standard rollout format
    """
    sample = {
        "input": question,
        "answer": answer,
        "task_id": task_id,
        "task_name": task_name,
        "system_prompt": system_prompt,
        "enable_thinking": enable_thinking,
        "with_tools": with_tools,
    }

    if extra_sampling_params:
        sample["sampling_params"] = extra_sampling_params

    return sample
