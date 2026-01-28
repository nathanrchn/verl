"""
HumanEval code generation benchmark implementation.
"""
from verl.utils.eval.benchmarks.base import BenchmarkConfig, make_rollout_sample
from verl.utils.eval.benchmarks.registry import register_benchmark
from verl.utils.eval.tasks import get_task, RolloutParams


def transform(example, tokenizer, config):
    """
    Transform HumanEval example to standard rollout format.

    Note: This creates a specialized format for code generation with:
    - Partial assistant message for code continuation
    - Stop tokens for code block completion
    - Multiple samples (n=10) for pass@k evaluation
    """
    sample = make_rollout_sample(
        question=f"Write a solution to the following problem and make sure that it passes the tests:\n```python\n{example['prompt']}\n```\n",
        answer="",  # No reference answer, evaluated via test execution
        task_id=example["task_id"],
        task_name="humaneval",
        with_tools=False,
        extra_sampling_params={
            "temperature": 0.8,
            "top_p": 0.95,
            "n": 10,  # Generate 10 samples for pass@k
            "stop": ["```"],
        },
    )

    # Add code-specific metadata for evaluation (goes into kwargs)
    sample["test"] = example["test"]
    sample["prompt"] = example["prompt"]
    sample["entry_point"] = example["entry_point"]

    # Special handling for code continuation
    sample["continue_assistant_message"] = True
    sample["assistant_prefix"] = f"Here is the completed function:\n```python\n{example['prompt']}\n"

    return sample


def compute_metrics(outputs, params_dict):
    """
    Compute HumanEval pass@k metrics using task registry.
    """
    params = RolloutParams.from_dict(params_dict) if isinstance(params_dict, dict) else params_dict
    task_fn = get_task(params.task_name)
    if task_fn is None:
        task_fn = get_task("humaneval")
    return task_fn(outputs, params)


CONFIG = BenchmarkConfig(
    name="humaneval",
    task_type="code",
    dataset_name="openai/openai_humaneval",
    dataset_split="test",
    sampling_params={
        "temperature": 0.8,
        "top_p": 0.95,
        "n": 10,
        "stop": ["```"],
    },
    max_new_tokens=2048,
    n_samples=10,  # For pass@k
    with_tools=False,
    transform_fn=transform,
    compute_metrics_fn=compute_metrics,
)

# Register benchmark
register_benchmark(CONFIG)
