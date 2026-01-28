"""
AVeriTeC fact verification benchmark implementation.
"""
from verl.utils.eval.benchmarks.base import BenchmarkConfig, make_rollout_sample
from verl.utils.eval.benchmarks.registry import register_benchmark
from verl.utils.eval.tasks import get_task, RolloutParams


AVERITEC_GUIDE = "Display one of the following answers: 'Supported', 'Refuted', 'Conflicting Evidence/Cherrypicking', or 'Not Enough Evidence'."


def transform(example, tokenizer, config):
    """Transform AVeriTeC example to standard rollout format."""
    sample = make_rollout_sample(
        question=example["claim"],
        answer=example["label"],
        task_id="averitec",
        task_name="averitec",
        system_prompt=AVERITEC_GUIDE,
        extra_sampling_params={"temperature": 0.8},
    )
    sample["use_tool"] = True
    return sample


def compute_metrics(outputs, params_dict):
    """Compute AVeriTeC accuracy using task registry."""
    params = RolloutParams.from_dict(params_dict) if isinstance(params_dict, dict) else params_dict
    task_fn = get_task(params.task_name)
    if task_fn is None:
        task_fn = get_task("averitec")
    return task_fn(outputs, params)


CONFIG = BenchmarkConfig(
    name="averitec",
    task_type="fact_verification",
    dataset_name="pminervini/averitec",
    dataset_split="dev",
    sampling_params={"temperature": 0.8},
    max_new_tokens=512,
    system_prompt=AVERITEC_GUIDE,
    transform_fn=transform,
    compute_metrics_fn=compute_metrics,
)

# Register benchmark
register_benchmark(CONFIG)
