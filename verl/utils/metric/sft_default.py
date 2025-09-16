from typing import Dict, Optional

import torch

from .base import BaseMetric


class DefaultSFTMetrics(BaseMetric):
    """Default SFT metrics: loss, accuracy, and perplexity.

    This metric accumulates token-level cross-entropy loss and accuracy using
    masked positions only.
    """

    def reset(self) -> None:
        # Lazily initialize on first update using the provided tensors' device
        self.sum_neg_log_likelihood = None
        self.sum_correct_tokens = None
        self.sum_valid_tokens = None

    def update(
        self,
        shift_logits: torch.Tensor,
        shift_labels: torch.Tensor,
        loss_mask: torch.Tensor,
        tokenizer=None,
    ) -> None:
        # shift_logits: [B, T, V], shift_labels: [B, T], loss_mask: [B, T]
        # compute per-token negative log likelihood
        log_probs = torch.log_softmax(shift_logits, dim=-1)
        # gather log probs at target labels
        nll = -torch.gather(log_probs, dim=-1, index=shift_labels.unsqueeze(-1)).squeeze(-1)

        valid_mask = loss_mask.to(dtype=nll.dtype)
        nll = nll * valid_mask

        # accuracy
        preds = torch.argmax(shift_logits, dim=-1)
        correct = (preds == shift_labels).to(dtype=nll.dtype) * valid_mask

        # initialize accumulators on the same device as inputs
        if self.sum_neg_log_likelihood is None:
            self.sum_neg_log_likelihood = torch.tensor(0.0, device=shift_logits.device, dtype=torch.float32)
            self.sum_correct_tokens = torch.tensor(0.0, device=shift_logits.device, dtype=torch.float32)
            self.sum_valid_tokens = torch.tensor(0.0, device=shift_logits.device, dtype=torch.float32)

        # accumulate
        self.sum_neg_log_likelihood += nll.sum()
        self.sum_correct_tokens += correct.sum()
        self.sum_valid_tokens += valid_mask.sum()

    def reduce(self) -> None:
        if (
            self.sum_neg_log_likelihood is not None
            and torch.distributed.is_available()
            and torch.distributed.is_initialized()
        ):
            torch.distributed.all_reduce(self.sum_neg_log_likelihood, op=torch.distributed.ReduceOp.SUM)
            torch.distributed.all_reduce(self.sum_correct_tokens, op=torch.distributed.ReduceOp.SUM)
            torch.distributed.all_reduce(self.sum_valid_tokens, op=torch.distributed.ReduceOp.SUM)

    def compute(self, prefix: Optional[str] = None) -> Dict[str, float]:
        eps = 1e-8
        total = (self.sum_valid_tokens.item() if self.sum_valid_tokens is not None else 0.0) + eps
        avg_loss = (self.sum_neg_log_likelihood.item() if self.sum_neg_log_likelihood is not None else 0.0) / total
        acc = (self.sum_correct_tokens.item() if self.sum_correct_tokens is not None else 0.0) / total
        ppl = float(torch.exp(torch.tensor(avg_loss)))
        pfx = "" if prefix is None else prefix
        return {
            f"{pfx}loss": float(avg_loss),
            f"{pfx}accuracy": float(acc),
            f"{pfx}ppl": float(ppl),
        }
