"""
Generation logger for logging sampled generations to tracking backends.
"""
from typing import Any
import random


class GenerationLogger:
    """Log sampled generations (256) to wandb as a table."""

    def log(self, loggers: dict[str, Any], generations: list[dict], step: int):
        """
        Log generation data to enabled backends.

        Args:
            loggers: Dictionary of logger instances keyed by backend name
            generations: List of generation dictionaries
            step: Training step number
        """
        if "wandb" in loggers:
            self._log_to_wandb(loggers["wandb"], generations, step)

    def _log_to_wandb(self, wandb_logger: Any, generations: list[dict], step: int):
        """
        Log generations to WandB as a table.

        Args:
            wandb_logger: WandB logger instance
            generations: List of generation dictionaries
            step: Training step number
        """
        try:
            import wandb

            columns = [
                "task_id",
                "task_name",
                "input",
                "output",
                "finish_reason",
                "metrics",
                "sampling_params",
            ]
            table = wandb.Table(columns=columns)

            for gen in generations:
                table.add_data(
                    gen.get("task_id", ""),
                    gen.get("task_name", ""),
                    gen.get("input", "")[:1000],  # Truncate long inputs
                    gen.get("output", "")[:2000],  # Truncate long outputs
                    gen.get("finish_reason", ""),
                    str(gen.get("metrics", {})),
                    str(gen.get("sampling_params", {})),
                )

            wandb_logger.log({"eval/generations": table}, step=step)
        except ImportError:
            print("Warning: wandb not installed, skipping generation logging")
        except Exception as e:
            print(f"Warning: Failed to log generations to wandb: {e}")

    def sample_generations(
        self, all_generations: list[dict], num_samples: int = 256
    ) -> list[dict]:
        """
        Sample a subset of generations for logging.

        Args:
            all_generations: Full list of generations
            num_samples: Number of samples to take (default 256)

        Returns:
            Sampled list of generations
        """
        if len(all_generations) <= num_samples:
            return all_generations

        return random.sample(all_generations, num_samples)
