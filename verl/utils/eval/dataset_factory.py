"""
Lazy dataset factory for evaluation benchmarks.
"""
from functools import partial
from typing import Any
from datasets import load_dataset, Dataset


class DatasetFactory:
    """Creates evaluation datasets on-demand, not during training init."""

    _cache: dict[str, Dataset] = {}

    @classmethod
    def create(
        cls,
        benchmark_name: str,
        tokenizer: Any,
        config: Any,  # BenchmarkConfig
    ) -> Dataset:
        """
        Lazily create and cache benchmark dataset.
        Only called when evaluation is triggered, not at training start.
        Config comes from BENCHMARK_REGISTRY, not user config.

        Args:
            benchmark_name: Name of the benchmark
            tokenizer: Tokenizer instance
            config: BenchmarkConfig from registry

        Returns:
            Transformed dataset ready for evaluation
        """
        cache_key = benchmark_name
        if cache_key in cls._cache:
            return cls._cache[cache_key]

        # Load from HuggingFace
        load_kwargs = {
            "path": config.dataset_name,
            "split": config.dataset_split,
            "trust_remote_code": True,
        }
        if config.dataset_subset:
            load_kwargs["name"] = config.dataset_subset

        dataset = load_dataset(**load_kwargs)

        if config.max_samples > 0:
            dataset = dataset.select(range(min(config.max_samples, len(dataset))))

        # Transform to standard format using benchmark's transform function
        if config.transform_fn:
            dataset = dataset.map(
                partial(config.transform_fn, tokenizer=tokenizer, config=config),
                num_proc=8,
            )

        cls._cache[cache_key] = dataset
        return dataset

    @classmethod
    def clear_cache(cls):
        """Clear the dataset cache."""
        cls._cache.clear()
