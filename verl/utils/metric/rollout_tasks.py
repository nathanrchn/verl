from typing import Any

from .utils import compute_text_ttr, compute_token_ttr


def default_task(output: dict[str, Any], rollout_params: dict[str, Any]) -> dict[str, float]:
    metrics = {}
    output_ids = output["output_ids"]
    output_text = output["text"]

    metrics["token_ttr"] = compute_token_ttr(output_ids)
    metrics["token_3gram_ttr"] = compute_token_ttr(output_ids, 3)
    metrics["text_ttr"] = compute_text_ttr(output_text)
    metrics["length"] = len(output_ids)

    return metrics


TASK_REGISTRY = {
    "default": default_task,
}


def get_task(task_id: str | None) -> callable:
    if task_id is None or task_id not in TASK_REGISTRY:
        return default_task
    return TASK_REGISTRY[task_id]
