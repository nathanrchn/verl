#!/usr/bin/env python3
"""
Launch standalone evaluation job.

Usage:
    python -m verl.utils.metric.launch_eval \
        --model-path /path/to/model \
        --dataset-path /path/to/rollout.parquet \
        --wandb-project my-project
"""
import os
import argparse
import subprocess
from datetime import datetime
from pathlib import Path

SBATCH_TEMPLATE = '''#!/bin/bash
#SBATCH --job-name=eval-{run_name}
#SBATCH --account={account}
#SBATCH --time={time}
#SBATCH --exclusive
#SBATCH --nodes=2
#SBATCH --gpus-per-node=4
#SBATCH --ntasks-per-node=5
#SBATCH --mem=460800
#SBATCH --partition={partition}
#SBATCH --output={output_dir}/%j/log.out
#SBATCH --error={output_dir}/%j/log.err

set -ex

MODEL_PATH="{model_path}"
DATASET_PATH="{dataset_path}"
WORK_DIR="/users/$(id -un)/projects/verl"

nodes=($(scontrol show hostnames "$SLURM_JOB_NODELIST"))
node1=${{nodes[0]}}
node2=${{nodes[1]}}
node1_ip=$(srun --nodes=1 --ntasks=1 --nodelist=$node1 hostname -i)
node2_ip=$(srun --nodes=1 --ntasks=1 --nodelist=$node2 hostname -i)

save_path="{output_dir}/$SLURM_JOB_ID"
mkdir -p "$save_path"

# Build wheel
WHEEL_DIR="$save_path/wheels"
mkdir -p "$WHEEL_DIR"
srun --nodes=1 --ntasks=1 --nodelist=$node1 --container-writable --environment=verl_swiss_old --kill-on-bad-exit=1 --output=$save_path/wheel.log --error=$save_path/wheel.err \
    bash --norc --noprofile -c "cd $WORK_DIR && pip wheel . --no-cache-dir --no-deps -w $WHEEL_DIR"
WHEEL=$(ls -t "$WHEEL_DIR"/*.whl | head -n1)

# Start 8 sglang workers (4 per node)
WORKER_URLS=""
for node_idx in 1 2; do
    [ "$node_idx" -eq 1 ] && node=$node1 && node_ip=$node1_ip || node=$node2 && node_ip=$node2_ip
    for gpu in 0 1 2 3; do
        port=$((50000 + gpu))
        WORKER_URLS="${{WORKER_URLS}} http://${{node_ip}}:${{port}}"
        srun --nodes=1 --ntasks=1 --nodelist=$node --container-writable --environment=verl_swiss_old --kill-on-bad-exit=1 \
            --gpus-per-task=1 --cpus-per-task=50 --gpu-bind=map_gpu:${{gpu}} --overlap \
            --output=$save_path/worker_${{node_idx}}_${{gpu}}.log --error=$save_path/worker_${{node_idx}}_${{gpu}}.err \
            bash --norc --noprofile -c "
export CUDA_DEVICE_ORDER=PCI_BUS_ID
export CUDA_VISIBLE_DEVICES=$gpu
python -m sglang.launch_server --model-path=$MODEL_PATH --dtype=bfloat16 --host=0.0.0.0 --port=$port \
    --skip-server-warmup --random-seed 42 --grammar-backend llguidance --mem-fraction-static 0.6 --max-running-requests 60" &
    done
done

# Start router
srun --nodes=1 --ntasks=1 --nodelist=$node1 --container-writable --environment=sglang_router --kill-on-bad-exit=1 \
    --cpus-per-task=50 --overlap --output=$save_path/router.log --error=$save_path/router.err \
    bash --norc --noprofile -c "python -m sglang_router.launch_router --host 0.0.0.0 --port 30000 --worker-urls $WORKER_URLS --model-path $MODEL_PATH --policy round_robin" &

sleep 30

# Run evaluation
srun --nodes=1 --ntasks=1 --nodelist=$node1 --container-writable --environment=verl_swiss_old --kill-on-bad-exit=1 \
    --cpus-per-task=50 --overlap --output=$save_path/eval.log --error=$save_path/eval.err \
    bash --norc --noprofile -c "
cd $WORK_DIR
pip install /iopsstor/scratch/cscs/nathanrchn/evaluate-0.4.6-py3-none-any.whl
pip install --no-deps /iopsstor/scratch/cscs/nathanrchn/nltk-3.9.2-py3-none-any.whl
pip install --no-deps /iopsstor/scratch/cscs/nathanrchn/emoji-2.15.0-py3-none-any.whl
pip install --no-deps /iopsstor/scratch/cscs/nathanrchn/syllapy-0.7.2-py3-none-any.whl
pip install --no-deps /iopsstor/scratch/cscs/nathanrchn/langdetect-1.0.9-py3-none-any.whl
pip install --no-deps /iopsstor/scratch/cscs/nathanrchn/immutabledict-4.2.2-py3-none-any.whl
pip install $WHEEL --no-cache-dir --no-deps --force-reinstall

python -m verl.utils.metric.main \
    --model-path $MODEL_PATH \
    --dataset-path $DATASET_PATH \
    --rollout-url http://${{node1_ip}}:30000 \
    {eval_args}" &

wait -n
scancel $SLURM_JOB_ID
'''


def main():
    parser = argparse.ArgumentParser(description="Launch evaluation job")
    parser.add_argument("--model-path", required=True, help="Path to model")
    parser.add_argument("--dataset-path", required=True, help="Path to rollout.parquet")
    parser.add_argument("--wandb-project", default="", help="Wandb project")
    parser.add_argument("--wandb-run-name", default="", help="Wandb run name")
    parser.add_argument("--max-samples", type=int, default=None, help="Max samples to evaluate")
    parser.add_argument("--time", default="0:30:00", help="Time limit")
    parser.add_argument("--partition", default="normal", help="SLURM partition")
    parser.add_argument("--account", default="infra01", help="SLURM account")
    parser.add_argument("--output-dir", default="./eval_logs", help="Output directory")
    args = parser.parse_args()

    output_dir = Path(args.output_dir).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    # Build eval args
    eval_args = []
    if args.wandb_project:
        eval_args.append(f"--wandb-project {args.wandb_project}")
    if args.wandb_run_name:
        eval_args.append(f"--wandb-run-name {args.wandb_run_name}")
    if args.max_samples:
        eval_args.append(f"--max-samples {args.max_samples}")

    run_name = args.wandb_run_name or Path(args.model_path).name[:20]

    sbatch = SBATCH_TEMPLATE.format(
        run_name=run_name,
        account=args.account,
        time=args.time,
        partition=args.partition,
        output_dir=output_dir,
        model_path=args.model_path,
        dataset_path=args.dataset_path,
        eval_args=" \\\n    ".join(eval_args) if eval_args else "",
    )

    sbatch_path = output_dir / f"eval_{datetime.now():%Y%m%d_%H%M%S}.sbatch"
    sbatch_path.write_text(sbatch)

    print(f"Submitting: {sbatch_path}")
    result = subprocess.run(["sbatch", str(sbatch_path)], capture_output=True, text=True)
    print(result.stdout.strip())
    if result.returncode != 0:
        print(f"Error: {result.stderr}")


if __name__ == "__main__":
    main()
