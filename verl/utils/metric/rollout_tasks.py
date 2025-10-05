from typing import Any

from json import loads

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


def gsm8k_task(output: dict[str, Any], rollout_params: dict[str, Any]) -> dict[str, float]:
    metrics = {}
    output_text = output["text"]

    true_answer = rollout_params["answer"]

    metrics["gsm8k_accuracy"] = 0.0
    metrics["gsm8k_call_tool"] = 0.0
    metrics["gsm8k_call_answer_tool"] = 0.0
    if "<|tools_prefix|>" in output_text:
        metrics["gsm8k_call_tool"] = 1.0
        tool_calls = output_text.split("<|tools_prefix|>")[1].split("<|tools_suffix|>")[0]
        tool_calls = loads(tool_calls)
        for tool_call in tool_calls:
            if "display_answers" in tool_call:
                arguments = loads(tool_call["display_answers"])
                if "answers" in arguments:
                    answers = arguments["answers"]
                    for answer in answers:
                        if answer == true_answer:
                            metrics["gsm8k_accuracy"] = 1.0
                            break
                metrics["gsm8k_call_answer_tool"] = 1.0
                break

    return metrics


TASK_REGISTRY = {
    "default": default_task,
    "gsm8k": gsm8k_task,
}


def get_task(task_id: str | None) -> callable:
    if task_id is None or task_id not in TASK_REGISTRY:
        return default_task
    return TASK_REGISTRY[task_id]
