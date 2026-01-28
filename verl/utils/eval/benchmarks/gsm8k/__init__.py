"""
GSM8K benchmark implementation.
"""
from verl.utils.eval.benchmarks.base import BenchmarkConfig, make_rollout_sample
from verl.utils.eval.benchmarks.registry import register_benchmark
from verl.utils.eval.tasks import get_task, RolloutParams


def transform(example, tokenizer, config):
    """Transform GSM8K example to standard rollout format."""
    answer = example["answer"].split("#### ")[-1].replace(",", "").strip()
    sample = make_rollout_sample(
        question=example["question"],
        answer=answer,
        task_id="gsm8k",
        task_name="gsm8k",
        extra_sampling_params={"temperature": 0.8},
    )
    # Add use_tool flag for task evaluation
    sample["use_tool"] = True
    return sample


def compute_metrics(outputs, params_dict):
    """Compute GSM8K accuracy using task registry."""
    # Convert to RolloutParams
    params = RolloutParams.from_dict(params_dict) if isinstance(params_dict, dict) else params_dict

    # Get task from registry
    task_fn = get_task(params.task_name)
    if task_fn is None:
        task_fn = get_task("gsm8k")

    return task_fn(outputs, params)


CONFIG = BenchmarkConfig(
    name="gsm8k",
    task_type="math",
    dataset_name="openai/gsm8k",
    dataset_subset="main",
    dataset_split="test",
    sampling_params={"temperature": 0.8},
    max_new_tokens=2048,
    transform_fn=transform,
    compute_metrics_fn=compute_metrics,
)

# Register benchmark
register_benchmark(CONFIG)
