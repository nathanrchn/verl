"""
IFEval (Instruction Following Evaluation) benchmark implementation.

Note: This benchmark requires additional instruction checker files:
- instructions_registry.py
- instructions.py
- instructions_util.py

These files should be obtained from the IFEval repository.
"""
from verl.utils.eval.benchmarks.base import BenchmarkConfig, make_rollout_sample
from verl.utils.eval.benchmarks.registry import register_benchmark
from verl.utils.eval.tasks import get_task, RolloutParams


def transform(example, tokenizer, config):
    """Transform IFEval example to standard rollout format."""
    sample = make_rollout_sample(
        question=example["prompt"],
        answer="",  # No reference answer for IFEval
        task_id="ifeval",
        task_name="ifeval",
        with_tools=False,
        extra_sampling_params={"temperature": 0.8},
    )
    # Add instruction metadata for evaluation (goes into kwargs)
    sample["instruction_id_list"] = example.get("instruction_id_list", [])
    sample["instruction_kwargs"] = example.get("kwargs", {})
    return sample


def compute_metrics(outputs, params_dict):
    """
    Compute IFEval instruction following metrics using task registry.
    """
    params = RolloutParams.from_dict(params_dict) if isinstance(params_dict, dict) else params_dict
    task_fn = get_task(params.task_name)
    if task_fn is None:
        task_fn = get_task("ifeval")
    return task_fn(outputs, params)


CONFIG = BenchmarkConfig(
    name="ifeval",
    task_type="ifeval",
    dataset_name="google/IFEval",
    dataset_split="train",
    sampling_params={"temperature": 0.8},
    max_new_tokens=2048,
    with_tools=False,
    transform_fn=transform,
    compute_metrics_fn=compute_metrics,
)

# Register benchmark
register_benchmark(CONFIG)
