"""
Configuration for training-time evaluation.
"""
from dataclasses import dataclass, field
from typing import Optional
from omegaconf import DictConfig


@dataclass
class RolloutConfig:
    """Configuration for Ray-managed rollout workers."""
    name: str = "sglang"  # or "vllm"
    nnodes: int = 1
    n_gpus_per_node: int = 8
    tensor_model_parallel_size: int = 1
    # Additional rollout-specific config can be added here


@dataclass
class EvalConfig:
    """Configuration for training-time evaluation."""
    enable: bool = True
    num_workers: int = 128  # For metric computation ProcessPool
    log_generations: int = 256  # Number of generations to log
    degeneration_threshold: float = 0.9
    device: str = "cuda"

    # Ray-managed rollout worker config
    rollout: RolloutConfig = field(default_factory=RolloutConfig)
    rollout_config: Optional[DictConfig] = None  # Full rollout worker config

    # Benchmark list - just names, all config comes from registry
    benchmarks: list[str] = field(default_factory=list)

    @classmethod
    def from_omega(cls, omega_conf: DictConfig) -> "EvalConfig":
        """Create EvalConfig from OmegaConf config."""
        rollout_cfg = RolloutConfig(
            name=omega_conf.rollout.get("name", "sglang"),
            nnodes=omega_conf.rollout.get("nnodes", 1),
            n_gpus_per_node=omega_conf.rollout.get("n_gpus_per_node", 8),
            tensor_model_parallel_size=omega_conf.rollout.get("tensor_model_parallel_size", 1),
        )

        return cls(
            enable=omega_conf.get("enable", True),
            num_workers=omega_conf.get("num_workers", 128),
            log_generations=omega_conf.get("log_generations", 256),
            degeneration_threshold=omega_conf.get("degeneration_threshold", 0.9),
            device=omega_conf.get("device", "cuda"),
            rollout=rollout_cfg,
            rollout_config=omega_conf.get("rollout_config"),
            benchmarks=list(omega_conf.get("benchmarks", [])),
        )
