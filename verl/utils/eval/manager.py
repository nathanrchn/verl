"""
Evaluation manager for training-time evaluation with Ray-managed workers.
"""
import asyncio
import random
from concurrent.futures import ProcessPoolExecutor
from typing import Any, Optional

import ray
from omegaconf import DictConfig

from verl.experimental.fully_async_policy.engine_workers import DetachAsyncRolloutWorker
from verl.single_controller.ray import RayClassWithInitArgs, RayResourcePool, RayWorkerGroup
from verl.utils.eval.benchmarks.registry import get_benchmark
from verl.utils.eval.config import EvalConfig
from verl.utils.eval.dataset_factory import DatasetFactory
from verl.utils.eval.generation_logger import GenerationLogger
from verl.utils.eval.utils import is_degenerating

# Constants for degeneration handling
DEGENERATION_THRESHOLD = 0.9
MAX_NEW_TOKENS_WINDOW = 1024


class EvalManager:
    """
    Manages training-time evaluation with:
    - Ray-managed DetachAsyncRolloutWorker instances
    - NCCL weight sync from training workers
    - Configurable benchmark list (names only, config from registry)
    - Lazy dataset loading
    - Degeneration detection and handling
    - Async metric computation
    - Generation logging (256 samples)
    """

    def __init__(
        self,
        config: EvalConfig,
        tokenizer: Any,
        actor_worker_group: RayWorkerGroup,  # Training workers for weight sync
    ):
        """
        Initialize evaluation manager.

        Args:
            config: Evaluation configuration
            tokenizer: Tokenizer instance
            actor_worker_group: Training worker group for weight sync
        """
        self.config = config
        self.tokenizer = tokenizer
        self.actor_wg = actor_worker_group
        self.benchmark_datasets = {}  # Lazy loaded
        self.process_pool = ProcessPoolExecutor(max_workers=config.num_workers)
        self.generation_logger = GenerationLogger()

        # Build rollout worker group
        self._build_rollout_workers()

        # Initialize NCCL collective for weight sync
        self._init_weight_sync_group()

    def _build_rollout_workers(self):
        """Create Ray-managed DetachAsyncRolloutWorker instances."""
        n_gpus = self.config.rollout.n_gpus_per_node
        nnodes = self.config.rollout.nnodes

        self.rollout_resource_pool = RayResourcePool(
            process_on_nodes=[n_gpus] * nnodes
        )

        # Build rollout config compatible with DetachAsyncRolloutWorker
        rollout_config = self._build_rollout_config()

        ray_cls = RayClassWithInitArgs(
            ray.remote(DetachAsyncRolloutWorker),
            config=rollout_config,
            role="rollout",
        )

        self.rollout_wg = RayWorkerGroup(
            resource_pool=self.rollout_resource_pool,
            ray_cls_with_init=ray_cls,
            device_name=self.config.device,
        )
        self.rollout_wg.init_model()

    def _build_rollout_config(self) -> DictConfig:
        """
        Build config compatible with DetachAsyncRolloutWorker.
        Uses the rollout_config from user config if provided,
        otherwise builds a basic config.
        """
        if self.config.rollout_config is not None:
            return self.config.rollout_config

        # Build a minimal config - actual implementation would need more details
        from omegaconf import OmegaConf

        config = OmegaConf.create({
            "rollout": {
                "name": self.config.rollout.name,
                "tensor_model_parallel_size": self.config.rollout.tensor_model_parallel_size,
                "engine_kwargs": {
                    "sglang": {
                        "grammar_backend": "llguidance",
                    }
                },
            },
            "hybrid_engine": False,
        })
        print(config)
        return config

    def _init_weight_sync_group(self):
        """Initialize NCCL collective group for actor -> rollout weight sync."""
        from ray.util.collective import collective

        # Get weights info from actor workers
        self.weights_info = self.actor_wg.get_actor_weights_info()[0]
        self.rollout_wg.set_actor_weights_info(self.weights_info)

        # Create NCCL collective group
        all_workers = self.actor_wg.workers + self.rollout_wg.workers
        try:
            collective.create_collective_group(
                all_workers,
                len(all_workers),
                list(range(len(all_workers))),
                backend="nccl",
                group_name="actor_rollout_eval",
            )
        except Exception as e:
            print(f"Warning: Failed to create NCCL collective group: {e}")
            print("Weight synchronization may not work properly")

    def sync_weights(self):
        """Synchronize weights from actor to rollout workers via NCCL broadcast."""
        try:
            self.actor_wg.sync_rollout_weights("actor_rollout_eval")
            ray.get(self.rollout_wg.sync_rollout_weights("actor_rollout_eval"))
        except Exception as e:
            print(f"Warning: Weight sync failed: {e}")

    def _load_benchmark(self, benchmark_name: str):
        """
        Lazy load benchmark dataset on first use. Config comes from registry.

        Args:
            benchmark_name: Name of the benchmark

        Returns:
            Loaded and transformed dataset
        """
        if benchmark_name not in self.benchmark_datasets:
            # Get full config from registry (not user config)
            bench_config = get_benchmark(benchmark_name)
            self.benchmark_datasets[benchmark_name] = DatasetFactory.create(
                benchmark_name,
                self.tokenizer,
                bench_config,
            )
        return self.benchmark_datasets[benchmark_name]

    async def evaluate(self, step: int) -> tuple[dict, list]:
        """
        Run evaluation on all configured benchmarks.

        Args:
            step: Training step number

        Returns:
            Tuple of (metrics_dict, sampled_generations_list)
        """
        # Sync weights before evaluation
        self.sync_weights()

        metrics = {}
        all_generations = []

        # self.config.benchmarks is just a list of names
        for benchmark_name in self.config.benchmarks:
            try:
                bench_config = get_benchmark(benchmark_name)
                dataset = self._load_benchmark(benchmark_name)
                bench_metrics, generations = await self._evaluate_benchmark(
                    bench_config, dataset
                )
                # Prefix metrics with benchmark name
                for key, value in bench_metrics.items():
                    metrics[f"{benchmark_name}/{key}"] = value
                all_generations.extend(generations)
            except Exception as e:
                print(f"Warning: Failed to evaluate {benchmark_name}: {e}")
                metrics[f"{benchmark_name}/error"] = str(e)

        # Sample 256 generations for logging
        sampled = self.generation_logger.sample_generations(
            all_generations, self.config.log_generations
        )
        return metrics, sampled

    async def _evaluate_benchmark(
        self, bench_config: Any, dataset: Any
    ) -> tuple[dict, list]:
        """
        Evaluate a single benchmark.

        Args:
            bench_config: Benchmark configuration
            dataset: Transformed dataset

        Returns:
            Tuple of (metrics_dict, generations_list)
        """
        from verl.utils.eval.tasks import RolloutParams

        generations = []

        # Generate for each example in the dataset
        for example in dataset:
            try:
                # Build prompt from example
                input_ids = self._prepare_input_ids(example)
                sampling_params = self._build_sampling_params(bench_config, example)

                # Generate with degeneration check
                output = await self._generate_with_degeneration_check(
                    input_ids, sampling_params
                )

                # Store generation for metrics and logging
                # Include all fields that might be needed for task evaluation
                generation = {
                    "task_id": example.get("task_id", ""),
                    "task_name": example.get("task_name", ""),
                    "input": example.get("input", ""),
                    "text": output.get("text", ""),
                    "output_ids": output.get("output_ids", []),
                    "answer": example.get("answer", ""),
                    "finish_reason": output["meta_info"].get("finish_reason", {}),
                    "sampling_params": sampling_params,
                }
                generations.append(generation)
            except Exception as e:
                print(f"Warning: Failed to generate for example: {e}")

        # Compute metrics using benchmark's compute_metrics_fn
        if bench_config.compute_metrics_fn:
            try:
                # Build RolloutParams from first example (for metadata)
                # All examples should have the same task_name and use_tool settings
                first_example = dataset[0] if len(dataset) > 0 else {}

                # Collect metadata that should go into kwargs
                kwargs = {}
                for key in ["test", "prompt", "entry_point", "test_list", "test_imports",
                           "instruction_id_list", "instruction_kwargs"]:
                    if key in first_example:
                        kwargs[key] = first_example[key]

                params_dict = {
                    "id": first_example.get("task_id", ""),
                    "task_name": first_example.get("task_name", ""),
                    "answer": first_example.get("answer", ""),
                    "use_tool": first_example.get("use_tool", True),
                    "sampling_params": sampling_params,
                    "kwargs": kwargs,
                }

                metrics = bench_config.compute_metrics_fn(generations, params_dict)
            except Exception as e:
                print(f"Warning: Failed to compute metrics: {e}")
                import traceback
                traceback.print_exc()
                metrics = {"error": str(e)}
        else:
            metrics = {}

        return metrics, generations

    def _prepare_input_ids(self, example: dict) -> list[int]:
        """
        Prepare input IDs from example.

        Args:
            example: Example dictionary

        Returns:
            List of token IDs
        """
        # If input_ids already present, use them
        if "input_ids" in example:
            return example["input_ids"]

        # Otherwise, tokenize the input text
        input_text = example.get("input", "")
        if input_text:
            return self.tokenizer.encode(input_text)

        return []

    def _build_sampling_params(self, bench_config: Any, example: dict) -> dict:
        """
        Build sampling parameters for generation.

        Args:
            bench_config: Benchmark configuration
            example: Example dictionary

        Returns:
            Dictionary of sampling parameters
        """
        params = bench_config.sampling_params.copy()
        params["max_new_tokens"] = bench_config.max_new_tokens

        # Override with example-specific params if present
        if "sampling_params" in example:
            params.update(example["sampling_params"])

        return params

    async def _generate_with_degeneration_check(
        self,
        input_ids: list[int],
        sampling_params: dict,
    ) -> dict:
        """
        Generate with degeneration detection and windowed continuation.

        Args:
            input_ids: Input token IDs
            sampling_params: Sampling parameters

        Returns:
            Generation output dictionary
        """
        max_new_tokens = sampling_params.get("max_new_tokens", 2048)
        aggregated_output = {"text": "", "output_ids": [], "meta_info": {}}

        while max_new_tokens > 0:
            window_params = {
                **sampling_params,
                "max_new_tokens": min(max_new_tokens, MAX_NEW_TOKENS_WINDOW),
            }

            output = await self._generate_single(input_ids, window_params)
            aggregated_output["text"] += output.get("text", "")
            aggregated_output["output_ids"] += output.get("output_ids", [])
            aggregated_output["meta_info"] = output.get("meta_info", {})

            # Check for natural stop
            finish_reason = output.get("meta_info", {}).get("finish_reason", {})
            if finish_reason.get("type") != "length":
                return aggregated_output

            # Check for degeneration
            if is_degenerating(
                output.get("output_ids", []), self.config.degeneration_threshold
            ):
                aggregated_output["meta_info"]["finish_reason"] = {
                    "type": "degenerating"
                }
                return aggregated_output

            max_new_tokens -= len(output.get("output_ids", []))
            input_ids = input_ids + output.get("output_ids", [])

        return aggregated_output

    async def _generate_single(
        self, input_ids: list[int], sampling_params: dict
    ) -> dict:
        """
        Generate a single completion using rollout workers.

        Args:
            input_ids: Input token IDs
            sampling_params: Sampling parameters

        Returns:
            Generation output dictionary
        """
        # Prepare prompt for rollout worker
        prompt = {"input_ids": input_ids, "sampling_params": sampling_params}

        # Call generate_batch on rollout workers
        try:
            results = await self.rollout_wg.generate_batch([prompt])
            return results[0] if results else {"text": "", "output_ids": [], "meta_info": {}}
        except Exception as e:
            print(f"Warning: Generation failed: {e}")
            return {
                "text": "",
                "output_ids": [],
                "meta_info": {"finish_reason": {"type": "error"}, "error": str(e)},
            }

    def _check_degeneration(self, output_ids: list[int]) -> bool:
        """
        Check if output is degenerating (TTR < threshold).

        Args:
            output_ids: Output token IDs

        Returns:
            True if degenerating
        """
        return is_degenerating(output_ids, self.config.degeneration_threshold)
