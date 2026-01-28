"""
Utility functions for evaluation metrics.
"""
from collections import Counter


def compute_token_ttr(token_ids: list[int], n: int = 1) -> float:
    """
    Compute n-gram Token Type Ratio (TTR).

    TTR = unique_ngrams / total_ngrams

    Args:
        token_ids: List of token IDs
        n: n-gram size (default 1 for unigrams)

    Returns:
        TTR value between 0 and 1. Lower values indicate more repetition.
    """
    if not token_ids or len(token_ids) < n:
        return 1.0

    if n == 1:
        unique_tokens = len(set(token_ids))
        total_tokens = len(token_ids)
        return unique_tokens / total_tokens

    # For n-grams > 1
    ngrams = []
    for i in range(len(token_ids) - n + 1):
        ngram = tuple(token_ids[i:i + n])
        ngrams.append(ngram)

    if not ngrams:
        return 1.0

    unique_ngrams = len(set(ngrams))
    total_ngrams = len(ngrams)
    return unique_ngrams / total_ngrams


def compute_text_ttr(text: str) -> float:
    """
    Compute word-level Type-Token Ratio (TTR).

    Args:
        text: Input text string

    Returns:
        TTR value between 0 and 1
    """
    words = text.lower().split()
    if not words:
        return 1.0

    unique_words = len(set(words))
    total_words = len(words)
    return unique_words / total_words


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
