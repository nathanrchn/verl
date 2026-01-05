"""
Rollout evaluation tasks for training-time metrics.

Each task is a callable that computes metrics from model outputs.
Tasks are registered in TASK_REGISTRY and can be referenced by ID in rollout_params.
"""
from __future__ import annotations
from typing import Any, Callable, Optional, Union, Dict, List
import os
from json import loads
from dataclasses import dataclass, field

@dataclass
class RolloutParams:
    """Dataclass for task rollout parameters."""
    id: str
    task_name: str
    answer: Any = None
    use_tool: bool = True
    sampling_params: dict = field(default_factory=dict)
    kwargs: dict = field(default_factory=dict)

    @classmethod
    def from_dict(cls, data: dict) -> RolloutParams:
        # Extract known fields
        known_fields = {"id", "task_name", "answer", "use_tool", "sampling_params", "kwargs"}
        kwargs = data.get("kwargs", {})

        # Put any extra fields into kwargs
        for key, value in data.items():
            if key not in known_fields:
                kwargs[key] = value

        return cls(
            id=data.get("id", ""),
            task_name=data.get("task_name", ""),
            answer=data.get("answer"),
            use_tool=data.get("use_tool", True),
            sampling_params=data.get("sampling_params", {}),
            kwargs=kwargs,
        )


try:
    import evaluate as hf_evaluate
except ImportError:
    hf_evaluate = None

from .utils import compute_text_ttr, compute_token_ttr
from .scorer import get_scorer


os.environ["HF_ALLOW_CODE_EVAL"] = "1"


def extract_answer_from_tool_call(output_text: str) -> str | None:
    """
    Extract answer from tool call in model output.

    Looks for display_answers tool calls in the format:
    <|tools_prefix|>[{"display_answers": {"answers": ["..."]}}]<|tools_suffix|>

    Returns:
        The first answer string if found, None otherwise.
    """
    if "<|tools_prefix|>" not in output_text:
        return None

    try:
        tool_calls_str = output_text.split("<|tools_prefix|>")[1].split("<|tools_suffix|>")[0]
        tool_calls = loads(tool_calls_str)
        for tool_call in tool_calls:
            if "display_answers" in tool_call:
                arguments = tool_call["display_answers"]
                if "answers" in arguments:
                    answers = arguments["answers"]
                    if answers and len(answers) > 0:
                        return str(answers[0])
        return None
    except Exception:
        # Try flexible parsing for malformed JSON
        try:
            tool_call = loads(output_text.split("<|tools_prefix|>[")[1])
            if "display_answers" in tool_call:
                arguments = tool_call["display_answers"]
                if "answers" in arguments:
                    answers = arguments["answers"]
                    if answers and len(answers) > 0:
                        return str(answers[0])
        except Exception:
            pass
        return None


def _aggregate_metrics(all_metrics: list[dict[str, float]]) -> dict[str, float]:
    """Aggregate a list of metric dicts by averaging values."""
    aggregated = {}
    for metric_dict in all_metrics:
        for key, value in metric_dict.items():
            if key not in aggregated:
                aggregated[key] = []
            aggregated[key].append(value)

    return {key: sum(values) / len(values) for key, values in aggregated.items()}


# =============================================================================
# Default Metrics - TTR and length (computed for all tasks)
# =============================================================================

def compute_default_metrics(output: dict[str, Any]) -> dict[str, float]:
    """
    Compute default metrics for a single output.

    Returns:
        - token_ttr: 1-gram token type-ratio
        - token_3gram_ttr: 3-gram token type-ratio
        - text_ttr: word-level type-ratio
        - length: output length in tokens
    """
    output_ids = output["output_ids"]
    output_text = output["text"]
    return {
        "token_ttr": compute_token_ttr(output_ids),
        "token_3gram_ttr": compute_token_ttr(output_ids, 3),
        "text_ttr": compute_text_ttr(output_text),
        "length": len(output_ids),
    }


# =============================================================================
# Math Tasks - Using AutoScoringJudge
# =============================================================================

def math_task(output: dict[str, Any] | list[dict[str, Any]], params: RolloutParams) -> dict[str, float]:
    """
    Mathematical reasoning evaluation using AutoScoringJudge.

    Supports: GSM8K, MATH-500, AMC12, OLYMPIAD_BENCH, MATH_ARENA, OMNI_MATH_*

    The AutoScoringJudge handles:
    - Exact string match
    - Numerical equality with tolerance
    - Mathematical expression equality (LaTeX)
    - Interval equality

    Metrics (organized by benchmark):
        - {task_name}/accuracy: correct answer rate
        - {task_name}/valid: valid tool call rate
    """
    task_name = params.task_name

    if isinstance(output, dict):
        output_text = output["text"]
        true_answer = str(params.answer)

        metrics = {
            f"{task_name}/accuracy": 0.0,
            f"{task_name}/valid": 0.0,
        }

        predicted_answer = extract_answer_from_tool_call(output_text)
        if predicted_answer is None and not params.use_tool:
            predicted_answer = output_text.strip()

        if predicted_answer:
            metrics[f"{task_name}/valid"] = 1.0
            scorer = get_scorer()
            try:
                if scorer.judge(true_answer, predicted_answer):
                    metrics[f"{task_name}/accuracy"] = 1.0
            except Exception:
                pass

        return metrics
    else:
        return _aggregate_metrics([math_task(o, params) for o in output])


# =============================================================================
# Fact Verification Tasks
# =============================================================================

def fact_verification_task(output: dict[str, Any] | list[dict[str, Any]], params: RolloutParams) -> dict[str, float]:
    """
    Fact verification evaluation (FEVER, FEVEROUS, AVeriTeC).

    Labels vary by dataset:
    - FEVER/FEVEROUS: SUPPORTS, REFUTES, NOT ENOUGH INFO
    - AVeriTeC: Supported, Refuted, Conflicting Evidence/Cherrypicking, Not Enough Evidence

    Metrics (organized by benchmark):
        - {task_name}/accuracy: correct label rate
        - {task_name}/valid: valid tool call rate
    """
    task_name = params.task_name

    if isinstance(output, dict):
        output_text = output["text"]
        true_answer = str(params.answer).lower().strip()

        metrics = {
            f"{task_name}/accuracy": 0.0,
            f"{task_name}/valid": 0.0,
        }

        predicted_answer = extract_answer_from_tool_call(output_text)
        if predicted_answer is None and not params.use_tool:
            predicted_answer = output_text.strip()

        if predicted_answer:
            metrics[f"{task_name}/valid"] = 1.0
            predicted_lower = predicted_answer.lower().strip()

            # Check for exact or partial match
            if predicted_lower == true_answer:
                metrics[f"{task_name}/accuracy"] = 1.0
            elif true_answer in predicted_lower or predicted_lower in true_answer:
                metrics[f"{task_name}/accuracy"] = 1.0

        return metrics
    else:
        return _aggregate_metrics([fact_verification_task(o, params) for o in output])


# =============================================================================
# Multiple Choice Tasks
# =============================================================================

def mcq_task(output: dict[str, Any] | list[dict[str, Any]], params: RolloutParams) -> dict[str, float]:
    """
    Multiple choice question evaluation (GPQA, ACP_BENCH).

    Labels: A, B, C, D, E

    Metrics (organized by benchmark):
        - {task_name}/accuracy: correct answer rate
        - {task_name}/valid: valid tool call rate
    """
    task_name = params.task_name

    if isinstance(output, dict):
        output_text = output["text"]
        true_answer = str(params.answer).upper().strip()

        metrics = {
            f"{task_name}/accuracy": 0.0,
            f"{task_name}/valid": 0.0,
        }

        predicted_answer = extract_answer_from_tool_call(output_text)
        if predicted_answer is None and not params.use_tool:
            predicted_answer = output_text.strip()

        if predicted_answer:
            metrics[f"{task_name}/valid"] = 1.0
            predicted_upper = predicted_answer.upper().strip()

            # Check first character for MCQ
            if len(predicted_upper) > 0:
                first_char = predicted_upper[0]
                if first_char in "ABCDE" and first_char == true_answer:
                    metrics[f"{task_name}/accuracy"] = 1.0
                elif predicted_upper == true_answer:
                    metrics[f"{task_name}/accuracy"] = 1.0

        return metrics
    else:
        return _aggregate_metrics([mcq_task(o, params) for o in output])


# =============================================================================
# Code Evaluation Tasks
# =============================================================================

if hf_evaluate is not None:
    HUMANEVAL_CODE_EVAL = hf_evaluate.load("code_eval")
else:
    HUMANEVAL_CODE_EVAL = None


def humaneval_task(outputs: list[dict[str, Any]], params: RolloutParams) -> dict[str, float]:
    """
    HumanEval code generation evaluation.

    Requires: HuggingFace evaluate library with code_eval

    Metrics:
        - humaneval/pass@10: pass@10 metric
    """
    if HUMANEVAL_CODE_EVAL is None:
        return {"humaneval/pass@10": 0.0}

    output_texts = [output["text"] for output in outputs]

    test = params.kwargs.get("test", "")
    entry_point = params.kwargs.get("entry_point", "")
    prompt = params.kwargs.get("prompt", "")

    pass_at_k, _ = HUMANEVAL_CODE_EVAL.compute(
        references=[f"{test}\ncheck({entry_point})"],
        predictions=[[prompt + ot for ot in output_texts]],
        k=[10],
    )
    return {"humaneval/pass@10": float(pass_at_k["pass@10"])}


def humaneval_thinking_task(outputs: list[dict[str, Any]], params: RolloutParams) -> dict[str, float]:
    """
    HumanEval with extended thinking/reasoning.

    Extracts code from ```python blocks after skipping the thinking section.

    Metrics:
        - humaneval_thinking/format_ratio: ratio of outputs with proper markdown code blocks
        - humaneval_thinking/pass@10: pass@10 for extracted code
    """
    if HUMANEVAL_CODE_EVAL is None:
        return {"humaneval_thinking/format_ratio": 0.0, "humaneval_thinking/pass@10": 0.0}

    output_texts = [output["text"] for output in outputs]

    # Calculate format ratio
    format_count = sum(
        1 for text in output_texts
        if "```python" in text and "```" in text.split("```python")[1]
    )
    format_ratio = format_count / len(output_texts)

    def extract_code(text: str) -> str:
        if "<|inner_suffix|>" in text:
            text = text.split("<|inner_suffix|>")[1]  # skip the thinking part
        if "```python" not in text:
            return ""
        return text.split("```python")[1].split("```")[0].strip()

    test = params.kwargs.get("test", "")
    entry_point = params.kwargs.get("entry_point", "")

    pass_at_k, _ = HUMANEVAL_CODE_EVAL.compute(
        references=[f"{test}\ncheck({entry_point})"],
        predictions=[[extract_code(ot) for ot in output_texts]],
        k=[10],
    )

    return {
        "humaneval_thinking/format_ratio": format_ratio,
        "humaneval_thinking/pass@10": float(pass_at_k["pass@10"]),
    }


# =============================================================================
# Instruction Following Tasks
# =============================================================================

from .ifeval.instructions_registry import INSTRUCTION_DICT as IFEVAL_INSTRUCTION_DICT
from .ifbench.instructions_registry import INSTRUCTION_DICT as IFBENCH_INSTRUCTION_DICT


def parse_non_reasoning_apertus(output_text: str) -> str:
    """Extract non-reasoning content from Apertus output."""
    if "<|inner_suffix|>" in output_text:
        output_text = output_text.split("<|inner_suffix|>")[1]
    return output_text.strip()


def ifeval_task(outputs: list[dict[str, Any]], params: RolloutParams) -> dict[str, float]:
    """
    IFEval instruction following evaluation.

    Checks if outputs satisfy all specified instructions.

    Metrics:
        - ifeval/accuracy: ratio of outputs satisfying ALL instructions
    """
    output_texts = [parse_non_reasoning_apertus(output["text"]) for output in outputs]

    instruction_id_list = params.kwargs.get("instruction_id_list", [])
    instruction_kwargs_list = params.kwargs.get("instruction_kwargs", [])

    results = []
    for output_text in output_texts:
        output_results = []
        for instruction_id, instruction_kwargs in zip(
            instruction_id_list,
            instruction_kwargs_list
        ):
            instruction_cls = IFEVAL_INSTRUCTION_DICT[instruction_id]
            instruction_obj = instruction_cls(instruction_id)
            _ = instruction_obj.build_description(**{
                k: v for k, v in instruction_kwargs.items() if v is not None
            })

            try:
                output_results.append(instruction_obj.check_following(output_text))
            except Exception as e:
                print(f"Error checking instruction {instruction_id}: {e}")
                output_results.append(False)

        results.append(all(output_results))

    return {"ifeval/accuracy": float(sum(results) / len(results))}


def ifbench_task(outputs: list[dict[str, Any]], params: RolloutParams) -> dict[str, float]:
    """
    IFBench instruction following evaluation (out-of-distribution).

    Same structure as IFEval but with different instruction set.

    Metrics:
        - ifbench/accuracy: ratio of outputs satisfying ALL instructions
    """
    output_texts = [parse_non_reasoning_apertus(output["text"]) for output in outputs]

    instruction_id_list = params.kwargs.get("instruction_id_list", [])
    instruction_kwargs_list = params.kwargs.get("instruction_kwargs", [])

    results = []
    for output_text in output_texts:
        output_results = []
        for instruction_id, instruction_kwargs in zip(
            instruction_id_list,
            instruction_kwargs_list
        ):
            instruction_cls = IFBENCH_INSTRUCTION_DICT[instruction_id]
            instruction_obj = instruction_cls(instruction_id)
            _ = instruction_obj.build_description(**{
                k: v for k, v in instruction_kwargs.items() if v is not None
            })

            try:
                output_results.append(instruction_obj.check_following(output_text))
            except Exception as e:
                print(f"Error checking instruction {instruction_id}: {e}")
                output_results.append(False)

        results.append(all(output_results))

    return {"ifbench/accuracy": float(sum(results) / len(results))}


# =============================================================================
# Task Registry
# =============================================================================

TASK_REGISTRY: dict[str, Callable[[Union[dict[str, Any], list[dict[str, Any]]], RolloutParams], dict[str, float]]] = {
    # Math reasoning (all use math_task with AutoScoringJudge)
    "math": math_task,
    "gsm8k": math_task,
    "math_500": math_task,
    "olympiad_bench": math_task,
    "omni_math": math_task,
    "omni_math_easy": math_task,
    "omni_math_med": math_task,
    "omni_math_hard": math_task,

    # Fact verification
    "fact_verification": fact_verification_task,
    "averitec": fact_verification_task,

    # Multiple choice
    "mcq": mcq_task,

    # Code evaluation
    "humaneval": humaneval_task,
    "humaneval_thinking": humaneval_thinking_task,

    # Instruction following
    "ifeval": ifeval_task,
    "ifbench": ifbench_task,
}


def get_task(task_id: str | None) -> Callable[[Union[dict[str, Any], list[dict[str, Any]]], RolloutParams], dict[str, float]] | None:
    """Get a task function by ID, returning None if not found."""
    if task_id is None or task_id not in TASK_REGISTRY:
        return None
    return TASK_REGISTRY[task_id]
