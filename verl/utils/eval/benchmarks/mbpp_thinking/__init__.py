"""
MBPP (Mostly Basic Python Problems) with thinking mode benchmark implementation.
"""
from verl.utils.eval.benchmarks.base import BenchmarkConfig, make_rollout_sample
from verl.utils.eval.benchmarks.registry import register_benchmark
from verl.utils.eval.tasks import get_task, RolloutParams


def transform(example, tokenizer, config):
    """
    Transform MBPP example to standard rollout format with thinking enabled.

    MBPP provides a prompt and test cases that the solution should pass.
    """
    # Format the test cases for display
    test_cases_str = "\n".join(example["test_list"])

    sample = make_rollout_sample(
        question=f"You are an expert Python programmer, and here is your task:\n{example['prompt']}\nYour code should pass these tests:\n{test_cases_str}",
        answer="",  # No reference answer, evaluated via test execution
        task_id=str(example["task_id"]),
        task_name="mbpp_thinking",
        enable_thinking=True,
        with_tools=False,
        extra_sampling_params={
            "temperature": 0.8,
            "top_p": 0.95,
        },
    )

    # Add code-specific metadata for evaluation (goes into kwargs)
    sample["prompt"] = example["prompt"]
    sample["test_imports"] = example.get("test_imports", [])
    sample["test_list"] = example["test_list"]

    return sample


def compute_metrics(outputs, params_dict):
    """
    Compute MBPP pass@k metrics using task registry.
    """
    params = RolloutParams.from_dict(params_dict) if isinstance(params_dict, dict) else params_dict
    task_fn = get_task(params.task_name)
    if task_fn is None:
        task_fn = get_task("mbpp_thinking")
    return task_fn(outputs, params)


CONFIG = BenchmarkConfig(
    name="mbpp_thinking",
    task_type="code",
    dataset_name="google-research-datasets/mbpp",
    dataset_subset="sanitized",
    dataset_split="test",
    sampling_params={
        "temperature": 0.8,
        "top_p": 0.95,
    },
    max_new_tokens=16384,
    enable_thinking=True,
    with_tools=False,
    transform_fn=transform,
    compute_metrics_fn=compute_metrics,
)

# Register benchmark
register_benchmark(CONFIG)
