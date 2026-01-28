"""
HumanEval with thinking mode benchmark implementation.
"""
from verl.utils.eval.benchmarks.base import BenchmarkConfig, make_rollout_sample
from verl.utils.eval.benchmarks.registry import register_benchmark
from verl.utils.eval.tasks import get_task, RolloutParams


def transform(example, tokenizer, config):
    """
    Transform HumanEval example to standard rollout format with thinking enabled.

    This version enables thinking mode for extended reasoning before code generation.
    """
    sample = make_rollout_sample(
        question=f"Write a solution to the following problem and make sure that it passes the tests:\n```python\n{example['prompt']}\n```\n",
        answer="",  # No reference answer, evaluated via test execution
        task_id=example["task_id"],
        task_name="humaneval_thinking",
        enable_thinking=True,
        with_tools=False,
        extra_sampling_params={
            "temperature": 0.8,
            "top_p": 0.95,
            "n": 10,  # Generate 10 samples for pass@k
        },
    )

    # Add code-specific metadata for evaluation (goes into kwargs)
    sample["test"] = example["test"]
    sample["prompt"] = example["prompt"]
    sample["entry_point"] = example["entry_point"]

    return sample


def compute_metrics(outputs, params_dict):
    """
    Compute HumanEval pass@k metrics using task registry.
    """
    params = RolloutParams.from_dict(params_dict) if isinstance(params_dict, dict) else params_dict
    task_fn = get_task(params.task_name)
    if task_fn is None:
        task_fn = get_task("humaneval_thinking")
    return task_fn(outputs, params)


CONFIG = BenchmarkConfig(
    name="humaneval_thinking",
    task_type="code",
    dataset_name="openai/openai_humaneval",
    dataset_split="test",
    sampling_params={
        "temperature": 0.8,
        "top_p": 0.95,
        "n": 10,
    },
    max_new_tokens=16384,
    n_samples=10,  # For pass@k
    enable_thinking=True,
    with_tools=False,
    transform_fn=transform,
    compute_metrics_fn=compute_metrics,
)

# Register benchmark
register_benchmark(CONFIG)
