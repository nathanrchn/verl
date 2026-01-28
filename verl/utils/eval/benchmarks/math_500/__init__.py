"""
MATH-500 benchmark implementation.
"""
from verl.utils.eval.benchmarks.base import BenchmarkConfig, make_rollout_sample
from verl.utils.eval.benchmarks.registry import register_benchmark
from verl.utils.eval.tasks import get_task, RolloutParams


def transform(example, tokenizer, config):
    """Transform MATH-500 example to standard rollout format."""
    sample = make_rollout_sample(
        question=example["problem"],
        answer=example["answer"],
        task_id="math_500",
        task_name="math_500",
        extra_sampling_params={"temperature": 0.8},
    )
    sample["use_tool"] = True
    return sample


def compute_metrics(outputs, params_dict):
    """Compute MATH-500 accuracy using task registry."""
    params = RolloutParams.from_dict(params_dict) if isinstance(params_dict, dict) else params_dict
    task_fn = get_task(params.task_name)
    if task_fn is None:
        task_fn = get_task("math_500")
    return task_fn(outputs, params)


CONFIG = BenchmarkConfig(
    name="math_500",
    task_type="math",
    dataset_name="HuggingFaceH4/MATH-500",
    dataset_split="test",
    sampling_params={"temperature": 0.8},
    max_new_tokens=2048,
    transform_fn=transform,
    compute_metrics_fn=compute_metrics,
)

# Register benchmark
register_benchmark(CONFIG)
