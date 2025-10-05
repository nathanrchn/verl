import time
from concurrent.futures import Future, ThreadPoolExecutor
from typing import Any, Iterator

import torch
import torch.distributed
from requests import post
from sglang.srt.utils import init_custom_process_group
from sglang.utils import wait_for_server
from torch.utils.data import DataLoader

from verl.utils.dataset.multiturn_sft_dataset import MultiTurnSFTDataset
from verl.utils.dataset.rl_dataset import collate_fn
from verl.utils.device import get_device_name
from verl.utils.metric.rollout_tasks import get_task
from verl.workers.rollout.sglang_rollout.utils import get_named_tensor_buckets

DEFAULT_SAMPLING_PARAMS = {
    "temperature": 0.0,
    "max_new_tokens": 256,
}


class AsyncRolloutMetrics:
    def __init__(
        self,
        rollout_dataset: MultiTurnSFTDataset,
        rollout_url: str,
        master_address: str,
        master_port: int,
        rollout_batch_size: int,
        pad_token_id: int,
    ):
        self.rollout_dataset = rollout_dataset
        self.rollout_url = rollout_url
        self.master_address = master_address
        self.master_port = master_port
        self.rollout_batch_size = rollout_batch_size
        self.pad_token_id = pad_token_id

        self._build_rollout_dataloader()
        self._init_weight_update_group()

        self.compute_metrics_step = 0
        self.compute_metrics_future = None

    def _build_rollout_dataloader(self):
        self.rollout_dataloader = DataLoader(
            self.rollout_dataset,
            batch_size=self.rollout_batch_size,
            collate_fn=collate_fn,
            num_workers=8,
            pin_memory=True,
            pin_memory_device=get_device_name(),
        )

    def _init_weight_update_group(self):
        wait_for_server(self.rollout_url)

        future = ThreadPoolExecutor().submit(
            post,
            url=f"{self.rollout_url}/init_weights_update_group",
            json={
                "master_address": self.master_address,
                "master_port": self.master_port,
                "rank_offset": 1,
                "world_size": 2,
                "group_name": "weight_update_group",
                "backend": "nccl",
            },
        )

        self.weight_update_store = torch.distributed.TCPStore(
            host_name=self.master_address,
            port=self.master_port,
            world_size=2,
            is_master=True,
            timeout=torch.distributed.default_pg_timeout,
        )

        self.weight_update_group = init_custom_process_group(
            backend="nccl",
            init_method=f"tcp://{self.master_address}:{self.master_port}",
            world_size=2,
            rank=0,
            group_name="weight_update_group",
        )

        future.result()

    def update_rollout_engine(self, per_tensor_param: Iterator[tuple[str, torch.Tensor]]):
        start_time = time.time()

        bucket_bytes = 512 << 20
        for params_batch in get_named_tensor_buckets(per_tensor_param, bucket_bytes):
            names, dtypes, shapes = [], [], []
            for name, param in params_batch:
                names.append(name)
                dtypes.append(str(param.dtype).split(".")[-1])
                shapes.append(list(param.shape))

            future = ThreadPoolExecutor().submit(
                post,
                url=f"{self.rollout_url}/update_weights_from_distributed",
                json={
                    "names": names,
                    "dtypes": dtypes,
                    "shapes": shapes,
                },
            )

            handles = []
            for name, param in params_batch:
                handles.append(torch.distributed.broadcast(param, src=0, group=self.weight_update_group, async_op=True))

            for handle in handles:
                handle.wait()

            future.result()

        print(f"update_rollout_engine time: {time.time() - start_time}")

    def wait_compute_metrics(self) -> tuple[dict[str, float], int]:
        return (
            self.compute_metrics_future.result() if self.compute_metrics_future is not None else {},
            self.compute_metrics_step,
        )

    def async_compute_metrics(self, step: int) -> tuple[Future[dict[str, float]], int]:
        output = self.wait_compute_metrics()

        self.compute_metrics_step = step
        self.compute_metrics_future = ThreadPoolExecutor().submit(self._async_compute_metrics)
        return output

    def _get_sampling_params(self, rollout_params: dict[str, Any]) -> dict[str, Any]:
        sampling_params = DEFAULT_SAMPLING_PARAMS.copy()
        sampling_params.update(rollout_params.get("sampling_params", {}))
        return sampling_params

    def _async_compute_metrics(self) -> dict[str, float]:
        metrics = {}
        for batch in self.rollout_dataloader:
            unpadded_input_ids = [ids[ids != self.pad_token_id].tolist() for ids in batch["input_ids"]]
            sampling_params = [self._get_sampling_params(rollout_param) for rollout_param in batch["rollout_params"]]

            outputs = post(
                url=f"{self.rollout_url}/generate",
                json={
                    "input_ids": unpadded_input_ids,
                    "sampling_params": sampling_params,
                },
            ).json()

            for output, rollout_param in zip(outputs, batch["rollout_params"], strict=False):
                task_id = rollout_param.get("id", None)
                task_fn = get_task(task_id)

                new_metrics = task_fn(output, rollout_param)

                for metric_name, metric_value in new_metrics.items():
                    if metric_name not in metrics:
                        metrics[metric_name] = []
                    metrics[metric_name].append(metric_value)

        aggregated_metrics = {}
        for metric_name, values in metrics.items():
            if values:
                aggregated_metrics[f"rollout/{metric_name}"] = sum(values) / len(values)

        return aggregated_metrics
