from typing import Any

import os
from json import loads
import evaluate as hf_evaluate

from .utils import compute_text_ttr, compute_token_ttr


os.environ["HF_ALLOW_CODE_EVAL"] = "1"


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
    metrics["gsm8k_call_tool_flexible"] = 0.0
    metrics["gsm8k_call_answer_tool_flexible"] = 0.0
    if "<|tools_prefix|>" in output_text:
        try:
            tool_calls = output_text.split("<|tools_prefix|>")[1].split("<|tools_suffix|>")[0]
            tool_calls = loads(tool_calls)
            metrics["gsm8k_call_tool"] = 1.0
            for tool_call in tool_calls:
                if "display_answers" in tool_call:
                    arguments = tool_call["display_answers"]
                    if "answers" in arguments:
                        answers = arguments["answers"]
                        for answer in answers:
                            if answer == true_answer:
                                metrics["gsm8k_accuracy"] = 1.0
                                break
                    metrics["gsm8k_call_answer_tool"] = 1.0
                    break
        except Exception:
            try:
                tool_call = loads(output_text.split("<|tools_prefix|>[")[1])
                metrics["gsm8k_call_tool_flexible"] = 1.0
                if "display_answers" in tool_call:
                    arguments = tool_call["display_answers"]
                    if "answers" in arguments:
                        answers = arguments["answers"]
                        for answer in answers:
                            if answer == true_answer:
                                metrics["gsm8k_accuracy"] = 1.0
                                break
                    metrics["gsm8k_call_answer_tool_flexible"] = 1.0
            except Exception:
                pass

    return metrics


HUMANEVAL_CODE_EVAL = hf_evaluate.load("code_eval")


def humaneval_task(output: dict[str, Any], rollout_params: dict[str, Any]) -> dict[str, float]:
    metrics = {}
    output_text = output["text"]

    pass_at_k, _ = HUMANEVAL_CODE_EVAL.compute(
        references=[f"{rollout_params['test']}\ncheck({rollout_params['entry_point']})"],
        predictions=[[rollout_params["prompt"] + ot for ot in output_text]],
        k=[10],
    )
    metrics["humaneval_pass@10"] = pass_at_k["pass@10"]
    return metrics


TASK_REGISTRY = {
    "default": default_task,
    "gsm8k": gsm8k_task,
    "humaneval": humaneval_task,
}


def get_task(task_id: str | None) -> callable:
    if task_id is None or task_id not in TASK_REGISTRY:
        return default_task
    return TASK_REGISTRY[task_id]
