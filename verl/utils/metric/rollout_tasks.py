from typing import Any

import os
from json import loads

try:
    import evaluate as hf_evaluate
except:
    hf_evaluate = None

from .utils import compute_text_ttr, compute_token_ttr


os.environ["HF_ALLOW_CODE_EVAL"] = "1"


def default_task(output: dict[str, Any] | list[dict[str, Any]], rollout_params: dict[str, Any]) -> dict[str, float]:
    metrics = {}
    if isinstance(output, dict):
        output_ids = output["output_ids"]
        output_text = output["text"]
        metrics["token_ttr"] = compute_token_ttr(output_ids)
        metrics["token_3gram_ttr"] = compute_token_ttr(output_ids, 3)
        metrics["text_ttr"] = compute_text_ttr(output_text)
        metrics["length"] = len(output_ids)
    else:
        all_metrics = []
        for o in output:
            all_metrics.append(default_task(o, rollout_params))
        
        aggregated_metrics = {}
        for metric_dict in all_metrics:
            for key, value in metric_dict.items():
                if key not in aggregated_metrics:
                    aggregated_metrics[key] = []
                aggregated_metrics[key].append(value)
        
        for key, values in aggregated_metrics.items():
            metrics[key] = sum(values) / len(values)
    return metrics


def gsm8k_task(output: dict[str, Any] | list[dict[str, Any]], rollout_params: dict[str, Any]) -> dict[str, float]:
    metrics = {}
    
    if isinstance(output, dict):
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
    else:
        all_metrics = []
        for o in output:
            all_metrics.append(gsm8k_task(o, rollout_params))
        
        aggregated_metrics = {}
        for metric_dict in all_metrics:
            for key, value in metric_dict.items():
                if key not in aggregated_metrics:
                    aggregated_metrics[key] = []
                aggregated_metrics[key].append(value)
        
        for key, values in aggregated_metrics.items():
            metrics[key] = sum(values) / len(values)

    return metrics

if hf_evaluate is not None:
    HUMANEVAL_CODE_EVAL = hf_evaluate.load("code_eval")
else:
    HUMANEVAL_CODE_EVAL = None


def humaneval_task(outputs: list[dict[str, Any]], rollout_params: dict[str, Any]) -> dict[str, float]:
    metrics = {}
    output_texts = [output["text"] for output in outputs]

    pass_at_k, _ = HUMANEVAL_CODE_EVAL.compute(
        references=[f"{rollout_params['test']}\ncheck({rollout_params['entry_point']})"],
        predictions=[[rollout_params["prompt"] + ot for ot in output_texts]],
        k=[10],
    )
    metrics["humaneval_pass@10"] = float(pass_at_k["pass@10"])
    return metrics


def humaneval_thinking_task(outputs: list[dict[str, Any]], rollout_params: dict[str, Any]) -> dict[str, float]:
    metrics = {}
    output_texts = [output["text"] for output in outputs]

    metrics["format_ratio"] = 0.0
    for output_text in output_texts:
        if "```python" in output_text and "```" in output_text.split("```python")[1]:
            metrics["format_ratio"] += 1
    metrics["format_ratio"] /= len(output_texts)

    def extract_code(text: str) -> str:
        if "<|inner_suffix|>" in text:
            text = text.split("<|inner_suffix|>")[1] # skip the thinking part
        if "```python" not in text:
            return ""
        code = text.split("```python")[1].split("```")[0]
        return code.strip()

    pass_at_k, _ = HUMANEVAL_CODE_EVAL.compute(
        references=[f"{rollout_params['test']}\ncheck({rollout_params['entry_point']})"],
        predictions=[[extract_code(ot) for ot in output_texts]],
        k=[10],
    )
    metrics["humaneval_thinking_pass@10"] = float(pass_at_k["pass@10"])
    return metrics


from .ifeval.instructions_registry import INSTRUCTION_DICT as IFEVAL_INSTRUCTION_DICT
from .ifbench.instructions_registry import INSTRUCTION_DICT as IFBENCH_INSTRUCTION_DICT


def parse_non_reasoning_apertus(output_text: str) -> str:
    if "<|inner_suffix|>" in output_text:
        output_text = output_text.split("<|inner_suffix|>")[1] # skip the thinking part
    return output_text.strip()


def ifeval_task(outputs: list[dict[str, Any]], rollout_params: dict[str, Any]) -> dict[str, float]:
    metrics = {}
    output_texts = [parse_non_reasoning_apertus(output["text"]) for output in outputs]

    results = []
    for output_text in output_texts:
        output_results = []
        for instruction_id, instruction_kwargs in zip(rollout_params["instruction_id_list"], rollout_params["kwargs"]):
            instruction_cls = IFEVAL_INSTRUCTION_DICT[instruction_id]
            instruction_obj = instruction_cls(instruction_id)

            _ = instruction_obj.build_description(**{k: v for k, v in instruction_kwargs.items() if v is not None})

            try:
                output_results.append(instruction_obj.check_following(output_text))
            except Exception as e:
                print(f"Error checking instruction {instruction_id} for output {output_text}: {e}")
                output_results.append(False)
        
        results.append(all(output_results))

    metrics["ifeval_accuracy"] = float(sum(results) / len(results))
    return metrics


def ifbench_task(outputs: list[dict[str, Any]], rollout_params: dict[str, Any]) -> dict[str, float]:
    metrics = {}
    output_texts = [parse_non_reasoning_apertus(output["text"]) for output in outputs]

    results = []
    for output_text in output_texts:
        output_results = []
        for instruction_id, instruction_kwargs in zip(rollout_params["instruction_id_list"], rollout_params["kwargs"]):
            instruction_cls = IFBENCH_INSTRUCTION_DICT[instruction_id]
            instruction_obj = instruction_cls(instruction_id)

            _ = instruction_obj.build_description(**{k: v for k, v in instruction_kwargs.items() if v is not None})

            try:
                output_results.append(instruction_obj.check_following(output_text))
            except Exception as e:
                print(f"Error checking instruction {instruction_id} for output {output_text}: {e}")
                output_results.append(False)
        
        results.append(all(output_results))

    metrics["ifbench_accuracy"] = float(sum(results) / len(results))
    return metrics


TASK_REGISTRY = {
    "default": default_task,
    "gsm8k": gsm8k_task,
    "humaneval": humaneval_task,
    "humaneval_thinking": humaneval_thinking_task,
    "ifeval": ifeval_task,
    "ifbench": ifbench_task,
}


def get_task(task_id: str | None) -> callable:
    if task_id is None or task_id not in TASK_REGISTRY:
        return default_task
    return TASK_REGISTRY[task_id]
