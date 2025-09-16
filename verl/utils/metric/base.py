import abc
from typing import Dict, Optional

import torch


class BaseMetric(abc.ABC):
    """Abstract base class for SFT evaluation metrics.

    Implementations should accumulate state across batches in update(),
    optionally perform distributed reductions in reduce(), and return a
    flat dict of scalar metrics in compute().
    """

    def __init__(self, config: Optional[object] = None):
        self.config = config
        self.reset()

    @abc.abstractmethod
    def reset(self) -> None:
        """Reset internal accumulators for a new evaluation run."""

    @abc.abstractmethod
    def update(
        self,
        shift_logits: torch.Tensor,
        shift_labels: torch.Tensor,
        loss_mask: torch.Tensor,
        tokenizer=None,
    ) -> None:
        """Accumulate statistics from a batch.

        Args:
            shift_logits: [batch, tokens, vocab] logits aligned with labels.
            shift_labels: [batch, tokens] label token ids.
            loss_mask:   [batch, tokens] mask (1 for valid, 0 for ignore).
            tokenizer:   Optional tokenizer for decoding-based metrics.
        """

    def reduce(self) -> None:
        """All-reduce accumulated statistics across distributed ranks.

        Default implementation is a no-op. Subclasses may override to
        perform torch.distributed.all_reduce on their accumulators.
        """

    @abc.abstractmethod
    def compute(self, prefix: Optional[str] = None) -> Dict[str, float]:
        """Return finalized scalar metrics.

        Args:
            prefix: Optional metric key prefix, e.g., "val/".
        """
