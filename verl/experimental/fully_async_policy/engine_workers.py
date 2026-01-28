# Copyright 2025 Bytedance Ltd. and/or its affiliates
# Copyright 2025 Meituan Ltd. and/or its affiliates
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import logging
import os
import time

import torch
import torch.distributed
from omegaconf import DictConfig

from verl.experimental.fully_async_policy.base_detach_sync import BaseDetachNcclSync
from verl.single_controller.base.decorator import Dispatch, register
from verl.utils.device import get_device_name, get_torch_device
from verl.workers.engine_workers import ActorRolloutRefWorker

logger = logging.getLogger(__file__)
logger.setLevel(os.getenv("VERL_LOGGING_LEVEL", "WARN"))

device_name = get_device_name()

__all__ = ["DetachActorWorker", "DetachAsyncRolloutWorker"]


class DetachNcclSync(BaseDetachNcclSync, ActorRolloutRefWorker):
    def __init__(self, config: DictConfig, role: str):
        BaseDetachNcclSync.__init__(self, config, role)
        ActorRolloutRefWorker.__init__(self, config, role)

        # _is_offload_param will be set after init_model() when actor is available
        self._is_offload_param = False

    @register(dispatch_mode=Dispatch.ONE_TO_ALL)
    def init_model(self):
        # Call parent init_model first
        super().init_model()

        # Now check offload state from the engine config
        if self._is_actor and hasattr(self, "actor") and self.actor is not None:
            engine_config = getattr(self.actor, "engine_config", None)
            if engine_config is not None:
                self._is_offload_param = getattr(engine_config, "param_offload", False)

    def _get_actor_params(self):
        pass

    @register(dispatch_mode=Dispatch.ONE_TO_ALL, blocking=False)
    def sync_rollout_weights(self, sync_group_name="actor_rollout"):
        assert (self._is_actor or self._is_rollout) and not self.config.hybrid_engine
        assert hasattr(self, "_weights_info") and self._weights_info is not None

        # Load model to GPU if offloaded
        if self._is_actor and self._is_offload_param:
            self.actor.engine.to("device", model=True, optimizer=False, grad=False)

        # Get parameters from actor
        params = self._get_actor_params() if self._is_actor else None

        rollout_name = self.config.rollout.name
        inference_model = None

        if self._is_rollout and (not self._is_actor):
            if rollout_name == "vllm":
                inference_model = BaseDetachNcclSync.get_inference_model(self.rollout)
                from verl.utils.vllm.patch import patch_vllm_moe_model_weight_loader

                patch_vllm_moe_model_weight_loader(inference_model)
            elif rollout_name == "sglang":
                inference_model = self.rollout._engine
                # For ServerAdapter, _engine might be None and needs async initialization
                if inference_model is None:
                    print("[sync_rollout_weights] Initialize server adapter engine")

                    async def init_engine():
                        if hasattr(self.rollout, "_init_server_adapter"):
                            await self.rollout._init_server_adapter()
                        else:
                            print("[sync_rollout_weights] No _init_server_adapter method found")
                        return self.rollout._engine

                    inference_model = self._run_async_safely(init_engine())
                    if inference_model is None:
                        raise RuntimeError(
                            f"Failed to initialize rollout engine. "
                            f"rollout type: {type(self.rollout)}, "
                            f"has _init_server_adapter: {hasattr(self.rollout, '_init_server_adapter')}"
                        )
            else:
                raise NotImplementedError(f"Unknown rollout name: {rollout_name}")

        if rollout_name == "sglang" and self._is_rollout:
            self._sync_sglang_weights(inference_model, params, sync_group_name)
        else:
            self._sync_vllm_weights(inference_model, params, sync_group_name)

        # Offload model back to CPU if needed
        if self._is_actor and self._is_offload_param:
            self.actor.engine.to("cpu", model=True, optimizer=False, grad=False)

        get_torch_device().empty_cache()

    def cache_actor_weights_to_cpu(self):
        self.cpu_named_params = {}
        if self._is_actor:
            params = self._get_actor_params()
            local_rank = torch.distributed.get_rank()
            world_size = torch.distributed.get_world_size()

            for tensor_idx, (key, _, _) in enumerate(self._weights_info):
                origin_data = params[key]
                if hasattr(origin_data, "full_tensor"):
                    origin_data = origin_data.full_tensor()

                if tensor_idx % world_size == local_rank:
                    self.cpu_named_params[key] = origin_data.to("cpu", non_blocking=True)
            get_torch_device().synchronize()

    @register(dispatch_mode=Dispatch.ONE_TO_ALL, blocking=False)
    def sync_rollout_weights_by_checkpoint(self, sync_group_name="actor_rollout"):
        assert (self._is_actor or self._is_rollout) and not self.config.hybrid_engine
        assert hasattr(self, "_weights_info") and self._weights_info is not None

        # Load model to GPU
        load_start_time = time.time()
        if self._is_actor and self._is_offload_param:
            self.actor.engine.to("device", model=True, optimizer=False, grad=False)
        load_duration = time.time() - load_start_time

        from ray.util.collective import collective

        # Cache actor weights to CPU and measure the time taken
        cache_start_time = time.time()
        self.cache_actor_weights_to_cpu()
        cache_end_time = time.time()
        cache_duration = cache_end_time - cache_start_time

        # Register the cached weights into the checkpoint engine
        self.checkpoint_engine.register_checkpoint(self._weights_info, self.cpu_named_params)
        register_end_time = time.time()
        register_duration = register_end_time - cache_end_time
        self.cpu_named_params = {}

        collective.barrier(group_name=sync_group_name)
        update_start_time = time.time()

        inference_model = None
        if self._is_rollout:
            inference_model = BaseDetachNcclSync.get_inference_model(self.rollout)
            from verl.utils.vllm.patch import patch_vllm_moe_model_weight_loader

            patch_vllm_moe_model_weight_loader(inference_model)

        # Update the checkpoint with the inference model and broadcast weights
        self.checkpoint_engine.update_checkpoint(
            inference_model=inference_model,
            group_name=sync_group_name,
            overlap_broadcast_and_consume=self.config.checkpoint_engine.overlap_broadcast_and_consume,
        )

        update_end_time = time.time()
        update_duration = update_end_time - update_start_time

        offload_start_time = time.time()
        if self._is_actor and self._is_offload_param:
            self.actor.engine.to("cpu", model=True, optimizer=False, grad=False)
        offload_duration = time.time() - offload_start_time

        print(
            f"sync_rollout_weights_by_checkpoint finish!, rank:{torch.distributed.get_rank()},"
            f" is_actor:{self._is_actor}, is_rollout:{self._is_rollout},"
            f" total cost:{update_end_time - cache_start_time} seconds, while cache cost {cache_duration} seconds, "
            f" register cost {register_duration} seconds, update cost {update_duration} seconds"
        )

        if self._is_actor and self._is_offload_param:
            print(
                f"sync_rollout_weights_by_checkpoint load model to gpu cost {load_duration} seconds,"
                f" offload model to cpu cost {offload_duration} seconds"
            )

    @register(dispatch_mode=Dispatch.ONE_TO_ALL)
    def save_model_to_cpu(self, n):
        if not hasattr(self, "cpu_saved_models"):
            self.cpu_saved_models = {}
        # Get model state dict from engine
        params = self._get_actor_params()
        self.cpu_saved_models[n] = {k: v.cpu().clone() for k, v in params.items()}

    @register(dispatch_mode=Dispatch.ONE_TO_ALL)
    def restore_model_from_cpu(self, n):
        if n in self.cpu_saved_models:
            saved_state = self.cpu_saved_models[n]
            # Load state back into the model
            # This requires engine-specific implementation
            # For now, we use the engine's load mechanism
            model = self.actor.engine.module if hasattr(self.actor.engine, "module") else None
            if model is not None:
                current_device = next(model.parameters()).device
                state_to_load = {k: v.to(current_device) for k, v in saved_state.items()}
                model.load_state_dict(state_to_load, strict=False)

    @register(dispatch_mode=Dispatch.ONE_TO_ALL)
    def clear_cpu_model(self, n):
        if hasattr(self, "cpu_saved_models") and n in self.cpu_saved_models:
            del self.cpu_saved_models[n]


class DetachActorWorker(DetachNcclSync):
    def __init__(self, config: DictConfig, role: str):
        print("[DetachActorWorker] Initializing via DetachNcclSync...")
        DetachNcclSync.__init__(self, config, role)

    def _get_actor_params(self):
        assert self._is_actor

        # Use engine's get_per_tensor_param method
        per_tensor_param, peft_config = self.actor.engine.get_per_tensor_param(
            layered_summon=getattr(self, "layered_summon", False),
            base_sync_done=True,
        )

        # Convert generator to dict
        params = {}
        for name, tensor in per_tensor_param:
            if hasattr(tensor, "full_tensor"):
                tensor = tensor.full_tensor()
            params[name] = tensor

        return params

    @register(dispatch_mode=Dispatch.ONE_TO_ALL)
    def get_actor_weights_info(self):
        assert self._is_actor
        if hasattr(self, "_weights_info") and self._weights_info is not None:
            return self._weights_info

        # Load model to GPU if offloaded
        if self._is_offload_param:
            self.actor.engine.to("device", model=True, optimizer=False, grad=False)

        # Get parameters and extract metadata
        params = self._get_actor_params()
        ret = []
        for key, tensor in params.items():
            ret.append((key, tuple(tensor.shape), tensor.dtype))

        self._weights_info = ret

        # Note: We don't offload here as sync_rollout_weights will be called immediately after
        # If offload is needed, it will happen in sync_rollout_weights

        return ret


class DetachAsyncRolloutWorker(DetachNcclSync):
    def __init__(self, config: DictConfig, role: str):
        print(f"[DetachAsyncRolloutWorker] {DetachAsyncRolloutWorker.__mro__}")
        DetachNcclSync.__init__(self, config, role)

    @register(dispatch_mode=Dispatch.ONE_TO_ALL)
    def set_actor_weights_info(self, weights_info):
        assert self._is_rollout
        self._weights_info = weights_info

    @register(dispatch_mode=Dispatch.DP_COMPUTE_PROTO, blocking=False)
    async def generate_batch(self, prompts: list[dict]) -> list[dict]:
        assert self._is_rollout

        rollout_name = self.config.rollout.name

        if rollout_name == "sglang":
            # Ensure the engine is initialized
            if self.rollout._engine is None:
                await self.rollout._init_server_adapter()

            # Generate for each prompt
            results = []
            for prompt in prompts:
                input_ids = prompt.get("input_ids", [])
                sampling_params = prompt.get("sampling_params", {})

                # Call the async generate method
                output = await self.rollout._engine.generate(
                    input_ids=input_ids,
                    sampling_params=sampling_params,
                )
                results.append(output)

            return results

        elif rollout_name == "vllm":
            # vLLM generation implementation
            results = []
            for prompt in prompts:
                input_ids = prompt.get("input_ids", [])
                sampling_params = prompt.get("sampling_params", {})

                # For vLLM, use the inference model's generate method
                inference_model = BaseDetachNcclSync.get_inference_model(self.rollout)
                # vLLM generate implementation would go here
                # This is a placeholder - vLLM has different API
                output = {
                    "text": "",
                    "output_ids": [],
                    "meta_info": {"finish_reason": {"type": "not_implemented"}},
                }
                results.append(output)

            return results

        else:
            raise NotImplementedError(f"Unknown rollout name: {rollout_name}")
