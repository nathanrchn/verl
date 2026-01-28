"""
Metrics computation for evaluation.
"""
import re
from collections import Counter
from typing import Any


def compute_token_ttr(token_ids: list[int]) -> float:
    """
    Compute Token Type Ratio (TTR) for degeneration detection.
    TTR = unique_tokens / total_tokens

    Args:
        token_ids: List of token IDs

    Returns:
        TTR value between 0 and 1. Lower values indicate more repetition.
    """
    if not token_ids:
        return 1.0

    unique_tokens = len(set(token_ids))
    total_tokens = len(token_ids)
    return unique_tokens / total_tokens


def is_degenerating(token_ids: list[int], threshold: float = 0.9) -> bool:
    """
    Check if generation is degenerating based on TTR threshold.

    Args:
        token_ids: List of token IDs
        threshold: Degeneration threshold (default 0.9)

    Returns:
        True if (1 - TTR) > threshold, indicating degeneration
    """
    ttr = compute_token_ttr(token_ids)
    return (1 - ttr) > threshold


def extract_math_answer(text: str) -> str:
    """
    Extract the final answer from a math problem solution.
    Handles various formats like "The answer is X", "#### X", etc.

    Args:
        text: Generated text containing the answer

    Returns:
        Extracted answer string
    """
    # Try to find answer after "####"
    if "####" in text:
        return text.split("####")[-1].strip()

    # Try to find "The answer is X" pattern
    patterns = [
        r"[Tt]he answer is[:\s]+([^\n.]+)",
        r"[Ff]inal answer[:\s]+([^\n.]+)",
        r"[Aa]nswer[:\s]+([^\n.]+)",
    ]

    for pattern in patterns:
        match = re.search(pattern, text)
        if match:
            return match.group(1).strip()

    # Return last line as fallback
    lines = text.strip().split("\n")
    return lines[-1].strip() if lines else ""


def normalize_math_answer(answer: str) -> str:
    """
    Normalize mathematical answer for comparison.
    Removes commas, dollar signs, and extra whitespace.

    Args:
        answer: Answer string to normalize

    Returns:
        Normalized answer string
    """
    answer = answer.replace(",", "").replace("$", "").strip()
    # Remove any boxed{} latex formatting
    answer = re.sub(r"\\boxed{([^}]+)}", r"\1", answer)
    return answer.lower()


def math_task(outputs: list[dict], params: dict) -> dict:
    """
    Compute accuracy for math tasks (GSM8K, MATH).

    Args:
        outputs: List of generation outputs with 'text' and 'answer' fields
        params: Additional parameters

    Returns:
        Dictionary with accuracy metric
    """
    correct = 0
    total = len(outputs)

    for output in outputs:
        generated = extract_math_answer(output.get("text", ""))
        reference = output.get("answer", "")

        generated_norm = normalize_math_answer(generated)
        reference_norm = normalize_math_answer(reference)

        if generated_norm == reference_norm:
            correct += 1

    accuracy = correct / total if total > 0 else 0.0
    return {
        "accuracy": accuracy,
        "correct": correct,
        "total": total,
    }


def code_task(outputs: list[dict], params: dict) -> dict:
    """
    Compute pass@k for code tasks using the evaluate library.
    This is a placeholder - actual implementation uses evaluate library's
    code_eval metric which handles sandboxed execution.

    Args:
        outputs: List of generation outputs
        params: Additional parameters including 'k' for pass@k

    Returns:
        Dictionary with pass@k metric
    """
    try:
        from evaluate import load
        code_eval = load("code_eval")

        # Prepare data for evaluation
        predictions = [output.get("text", "") for output in outputs]
        references = [output.get("test_cases", []) for output in outputs]

        k = params.get("k", [1, 10])
        results = code_eval.compute(
            predictions=predictions,
            references=references,
            k=k,
        )
        return results
    except Exception as e:
        # Fallback if evaluate library not available
        return {
            "pass@1": 0.0,
            "pass@10": 0.0,
            "error": str(e),
        }


def fact_verification_task(outputs: list[dict], params: dict) -> dict:
    """
    Compute metrics for fact verification tasks (AVeriTeC).
    Checks if generated answer matches reference label.

    Args:
        outputs: List of generation outputs
        params: Additional parameters

    Returns:
        Dictionary with accuracy metric
    """
    correct = 0
    total = len(outputs)

    for output in outputs:
        generated = output.get("text", "").lower().strip()
        reference = output.get("label", "").lower().strip()

        # Check if reference label appears in generated text
        if reference in generated:
            correct += 1

    accuracy = correct / total if total > 0 else 0.0
    return {
        "accuracy": accuracy,
        "correct": correct,
        "total": total,
    }


def ifeval_task(outputs: list[dict], params: dict) -> dict:
    """
    Compute metrics for instruction following tasks (IFEval, IFBench).
    This is a placeholder - actual implementation requires instruction checker.

    Args:
        outputs: List of generation outputs
        params: Additional parameters

    Returns:
        Dictionary with instruction following metrics
    """
    # This will be implemented with the instruction checker from the benchmark
    return {
        "instruction_following_accuracy": 0.0,
        "strict_accuracy": 0.0,
        "total": len(outputs),
    }
