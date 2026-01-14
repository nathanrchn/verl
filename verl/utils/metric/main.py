#!/usr/bin/env python3
"""
Standalone evaluation script for post-training model evaluation.

Usage:
    python -m verl.utils.metric.main \
        --model-path /path/to/model \
        --dataset-path /path/to/rollout.parquet \
        --rollout-url http://localhost:30000 \
        --wandb-project my-project \
        --wandb-run-name my-eval-run
"""
import time
import argparse
from json import dumps

import wandb
from transformers import AutoTokenizer

from verl.utils.dataset.multiturn_sft_dataset import ApertusSFTDataset
from verl.utils.metric.async_rollout_metrics import AsyncRolloutMetrics


def main():
    parser = argparse.ArgumentParser(description="Standalone evaluation script")
    parser.add_argument("--model-path", type=str, required=True, help="Path to the model (for tokenizer)")
    parser.add_argument("--dataset-path", type=str, required=True, help="Path to the rollout dataset (parquet)")
    parser.add_argument("--rollout-url", type=str, default="http://localhost:30000", help="URL of the rollout server")
    parser.add_argument("--rollout-batch-size", type=int, default=8192, help="Batch size for rollout")
    parser.add_argument("--max-length", type=int, default=4096, help="Max sequence length")
    parser.add_argument("--num-workers", type=int, default=128, help="Number of workers for metric computation")

    # Wandb arguments
    parser.add_argument("--wandb-project", type=str, default=None, help="Wandb project name")
    parser.add_argument("--wandb-run-name", type=str, default=None, help="Wandb run name")
    parser.add_argument("--wandb-entity", type=str, default=None, help="Wandb entity")
    parser.add_argument("--log-generations", type=int, default=256, help="Number of generations to log to wandb")

    args = parser.parse_args()

    # Initialize wandb if project is specified
    if args.wandb_project:
        wandb.init(
            project=args.wandb_project,
            name=args.wandb_run_name,
            entity=args.wandb_entity,
            config={
                "model_path": args.model_path,
                "dataset_path": args.dataset_path,
                "rollout_batch_size": args.rollout_batch_size,
            },
        )

    # Load tokenizer
    print(f"Loading tokenizer from {args.model_path}...")
    tokenizer = AutoTokenizer.from_pretrained(args.model_path)

    # Build dataset config
    dataset_config = {
        "max_length": args.max_length,
        "truncation": "right",
        "add_generation_prompt": True,
    }

    # Load dataset
    print(f"Loading dataset from {args.dataset_path}...")
    dataset = ApertusSFTDataset(
        parquet_files=args.dataset_path,
        tokenizer=tokenizer,
        config=dataset_config,
    )
    print(f"Loaded {len(dataset)} samples")

    # Create evaluator in standalone mode
    evaluator = AsyncRolloutMetrics(
        rollout_dataset=dataset,
        rollout_url=args.rollout_url,
        rollout_batch_size=args.rollout_batch_size,
        pad_token_id=tokenizer.pad_token_id or 3,
        num_workers=args.num_workers,
        standalone=True,
    )

    # Run evaluation
    print("\nStarting evaluation...")
    start_time = time.time()
    metrics, generations = evaluator.compute_metrics()
    total_time = time.time() - start_time

    # Print metrics
    print("\n" + "=" * 60)
    print("Evaluation Results")
    print("=" * 60)
    for metric_name, value in sorted(metrics.items()):
        print(f"  {metric_name}: {value:.4f}")
    print(f"\nTotal evaluation time: {total_time:.2f}s")

    # Log to wandb
    if args.wandb_project:
        wandb.log(metrics)

        # Log generations as a table
        if generations and args.log_generations > 0:
            sampled_gens = generations[:args.log_generations]
            table_data = []
            for gen in sampled_gens:
                row = [
                    gen.get("task_id", ""),
                    gen.get("task_name", ""),
                    gen.get("input", "")[:2000],
                    gen.get("output", "")[:2000],
                    gen.get("finish_reason", ""),
                    dumps(gen.get("metrics", {})),
                    dumps(gen.get("sampling_params", {})),
                ]
                table_data.append(row)

            table = wandb.Table(
                columns=["task_id", "task_name", "input", "output", "finish_reason", "metrics", "sampling_params"],
                data=table_data,
            )
            wandb.log({"generations": table})

        wandb.finish()

    # Cleanup
    evaluator.shutdown()
    print("\nEvaluation complete!")


if __name__ == "__main__":
    main()
