"""
Benchmark registry for auto-discovery of available benchmarks.
"""
from typing import Optional
from verl.utils.eval.benchmarks.base import BenchmarkConfig


# Global registry - will be populated as benchmarks are imported
BENCHMARK_REGISTRY: dict[str, BenchmarkConfig] = {}


def register_benchmark(config: BenchmarkConfig):
    """
    Register a benchmark configuration.

    Args:
        config: BenchmarkConfig instance to register
    """
    BENCHMARK_REGISTRY[config.name] = config


def get_benchmark(name: str) -> BenchmarkConfig:
    """
    Get benchmark config by name.

    Args:
        name: Benchmark name

    Returns:
        BenchmarkConfig instance

    Raises:
        KeyError: If benchmark not found
    """
    if name not in BENCHMARK_REGISTRY:
        available = list(BENCHMARK_REGISTRY.keys())
        raise KeyError(
            f"Unknown benchmark: {name}. Available benchmarks: {available}"
        )
    return BENCHMARK_REGISTRY[name]


def list_benchmarks() -> list[str]:
    """
    List all available benchmark names.

    Returns:
        List of registered benchmark names
    """
    return list(BENCHMARK_REGISTRY.keys())


# Import all benchmarks to populate registry
# These imports will trigger registration via each benchmark's __init__.py
try:
    from verl.utils.eval.benchmarks import gsm8k
except ImportError:
    pass

try:
    from verl.utils.eval.benchmarks import gsm8k_thinking
except ImportError:
    pass

try:
    from verl.utils.eval.benchmarks import math_500
except ImportError:
    pass

try:
    from verl.utils.eval.benchmarks import averitec
except ImportError:
    pass

try:
    from verl.utils.eval.benchmarks import ifeval
except ImportError:
    pass

try:
    from verl.utils.eval.benchmarks import ifbench
except ImportError:
    pass

try:
    from verl.utils.eval.benchmarks import humaneval
except ImportError:
    pass

try:
    from verl.utils.eval.benchmarks import humaneval_thinking
except ImportError:
    pass

try:
    from verl.utils.eval.benchmarks import mbpp_thinking
except ImportError:
    pass
