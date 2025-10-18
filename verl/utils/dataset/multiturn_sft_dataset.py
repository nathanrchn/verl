# Copyright 2024 Bytedance Ltd. and/or its affiliates
# Copyright 2025 ModelBest Inc. and/or its affiliates

# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at

#     http://www.apache.org/licenses/LICENSE-2.0

# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""
Multi-turn SFT dataset that supports training on conversation data with multiple turns
"""

import logging
from typing import Any, Optional

import numpy as np
import pandas as pd
import torch
from json import loads
from omegaconf import ListConfig
from torch.utils.data import Dataset
from transformers import PreTrainedTokenizer

from verl.utils import hf_tokenizer
from verl.utils.fs import copy_local_path_from_hdfs
from verl.utils.model import compute_position_id_with_mask
from verl.utils.torch_functional import pad_sequence_to_length, postprocess_data


def convert_nested_value_to_list_recursive(data_item):
    if isinstance(data_item, dict):
        return {k: convert_nested_value_to_list_recursive(v) for k, v in data_item.items()}
    elif isinstance(data_item, list):
        return [convert_nested_value_to_list_recursive(elem) for elem in data_item]
    elif isinstance(data_item, np.ndarray):
        # Convert to list, then recursively process the elements of the new list
        return convert_nested_value_to_list_recursive(data_item.tolist())
    else:
        # Base case: item is already a primitive type (int, str, float, bool, etc.)
        return data_item


class MultiTurnSFTDataset(Dataset):
    """
    Dataset for multi-turn conversations where each assistant response should be trained
    """

    def __init__(self, parquet_files: str | list[str], tokenizer, config=None):
        # Set defaults and extract parameters from config if provided
        config = config or {}
        self.pad_mode = config.get("pad_mode", "right")
        assert self.pad_mode in ["right", "left_right"], (
            f"Expect pad_mode to be 'right' or 'left_right'. Got {self.pad_mode}"
        )
        self.truncation = config.get("truncation", "error")
        # for right padding
        self.max_length = config.get("max_length", 1024)
        # for left right paddding to be consistent with RL
        self.max_prompt_length = config.get("max_prompt_length", 512)
        self.max_response_length = config.get("max_response_length", 512)
        # Get messages_key from the new multiturn config structure
        multiturn_config = config.get("multiturn", {})
        self.messages_key = multiturn_config.get("messages_key", "messages")
        self.tools_key = multiturn_config.get("tools_key", "tools")
        self.enable_thinking_key = multiturn_config.get("enable_thinking_key", "enable_thinking")
        self.rollout_params_key = multiturn_config.get("rollout_params_key", "rollout_params")
        self.apply_chat_template_kwargs = config.get("apply_chat_template_kwargs", {})
        assert self.truncation in ["error", "left", "right"]
        # for rollout
        self.add_generation_prompt = config.get("add_generation_prompt", False)

        if not isinstance(parquet_files, list | ListConfig):
            parquet_files = [parquet_files]

        self.parquet_files = parquet_files
        if isinstance(tokenizer, str):
            tokenizer = hf_tokenizer(tokenizer)
        self.tokenizer: PreTrainedTokenizer = tokenizer

        self._download()
        self._read_files_and_process()

    def _download(self):
        for i, parquet_file in enumerate(self.parquet_files):
            self.parquet_files[i] = copy_local_path_from_hdfs(parquet_file, verbose=True)

    def _read_files_and_process(self):
        def series_to_item(ls):
            import numpy
            import pandas

            while isinstance(ls, pandas.core.series.Series | numpy.ndarray) and len(ls) == 1:
                ls = ls[0]
            return ls

        dataframes = []
        for parquet_file in self.parquet_files:
            dataframe = pd.read_parquet(parquet_file)
            dataframes.append(dataframe)
        self.dataframe = pd.concat(dataframes)

        # Extract messages list from dataframe
        self.messages = (
            self.dataframe[self.messages_key]
            .apply(series_to_item)
            .apply(convert_nested_value_to_list_recursive)
            .tolist()
        )

        # Extract tools list from dataframe
        if self.tools_key in self.dataframe.columns:
            self.tools = self.dataframe[self.tools_key].apply(convert_nested_value_to_list_recursive).tolist()
        else:
            self.tools = None

        # Extract enable_thinking list from dataframe
        if self.enable_thinking_key in self.dataframe.columns:
            self.enable_thinking = self.dataframe[self.enable_thinking_key].tolist()
        else:
            self.enable_thinking = None

        # Extract rollout_params list from dataframe
        if self.rollout_params_key in self.dataframe.columns:
            self.rollout_params = self.dataframe[self.rollout_params_key].tolist()
        else:
            self.rollout_params = None

    def __len__(self):
        return len(self.messages)

    def _process_message_tokens(
        self,
        messages: list[dict[str, Any]],
        start_idx: int,
        end_idx: int,
        is_assistant: bool = False,
        enable_thinking: Optional[bool] = None,
        tools: Optional[list[dict[str, Any]]] = None,
    ) -> tuple[list[int], list[int], list[int]]:
        """
        Process tokens for a single message or a group of messages.

        Args:
            messages: List of message dictionaries
            start_idx: Start index in messages list
            end_idx: End index in messages list
            is_assistant: Whether this is an assistant message
            enable_thinking: Whether to enable thinking mode

        Returns:
            Tuple of (tokens, loss_mask, attention_mask)
        """
        if start_idx > 0:
            prev_applied_text = self.tokenizer.apply_chat_template(
                messages[:start_idx],
                tokenize=False,
                add_generation_prompt=False,
                enable_thinking=enable_thinking,
                tools=tools,
                **self.apply_chat_template_kwargs,
            )
            if is_assistant:
                prev_applied_text_w_generation_prompt = self.tokenizer.apply_chat_template(
                    messages[:start_idx],
                    tokenize=False,
                    add_generation_prompt=True,
                    enable_thinking=enable_thinking,
                    tools=tools,
                    **self.apply_chat_template_kwargs,
                )

        else:
            prev_applied_text = ""

        cur_applied_text = self.tokenizer.apply_chat_template(
            messages[:end_idx],
            tokenize=False,
            add_generation_prompt=False,
            enable_thinking=enable_thinking,
            tools=tools,
            **self.apply_chat_template_kwargs,
        )
        # Get tokens for the current message only
        if is_assistant:
            generation_prompt_text = prev_applied_text_w_generation_prompt[len(prev_applied_text) :]
            generation_prompt_tokens = self.tokenizer.encode(
                generation_prompt_text,
                add_special_tokens=False,
            )
            _message_tokens = self.tokenizer.encode(
                cur_applied_text[len(prev_applied_text_w_generation_prompt) :],
                add_special_tokens=False,
            )
            message_tokens = generation_prompt_tokens + _message_tokens
            loss_mask = [0] * (len(generation_prompt_tokens)) + [1] * (
                len(message_tokens) - len(generation_prompt_tokens)
            )
        else:
            message_tokens = self.tokenizer.encode(
                cur_applied_text[len(prev_applied_text) :],
                add_special_tokens=False,
            )
            loss_mask = [0] * len(message_tokens)

        attention_mask = [1] * len(message_tokens)

        return message_tokens, loss_mask, attention_mask

    def _validate_and_convert_tokens(
        self,
        full_tokens: torch.Tensor,
        concat_tokens: list[int],
        concat_loss_mask: list[int],
        concat_attention_mask: list[int],
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Validate tokenization and convert to tensors.

        Args:
            full_tokens: Full conversation tokens
            concat_tokens: Concatenated tokens
            concat_loss_mask: Concatenated loss mask
            concat_attention_mask: Concatenated attention mask

        Returns:
            Tuple of (input_ids, loss_mask, attention_mask) as tensors
        """
        full_tokens_list = full_tokens.tolist()

        if len(concat_tokens) != len(full_tokens_list) or not all(
            a == b for a, b in zip(concat_tokens, full_tokens_list, strict=True)
        ):
            logging.warning(
                f"Token mismatch detected! Full tokenization length: {len(full_tokens_list)}, Concatenated tokens "
                f"length: {len(concat_tokens)}. Using concatenated version."
                # f"full tokens text: {self.tokenizer.decode(full_tokens_list)}"
                # f"concat tokens text: {self.tokenizer.decode(concat_tokens)}"
            )
            return (
                torch.tensor(concat_tokens, dtype=torch.long),
                torch.tensor(concat_loss_mask, dtype=torch.long),
                torch.tensor(concat_attention_mask, dtype=torch.long),
            )

        return (
            full_tokens,
            torch.tensor(concat_loss_mask, dtype=torch.long),
            torch.tensor(concat_attention_mask, dtype=torch.long),
        )

    def __getitem__(self, item):
        tokenizer = self.tokenizer
        messages = self.messages[item]
        tools = self.tools[item] if self.tools is not None else None
        enable_thinking = self.enable_thinking[item] if self.enable_thinking is not None else None

        # First, get the full conversation tokens
        try:
            full_tokens = tokenizer.apply_chat_template(
                messages,
                tools=tools,
                tokenize=True,
                return_tensors="pt",
                add_generation_prompt=self.add_generation_prompt,
                enable_thinking=enable_thinking,
                **self.apply_chat_template_kwargs,
            )
        except Exception as e:
            logging.error(
                f"Error applying chat template: {e}\nMessages: {messages}\nTools: {tools}\nEnable thinking: "
                f"{enable_thinking}"
            )
            raise

        # Track concatenated tokens for validation
        concat_tokens = []
        concat_loss_mask = []
        concat_attention_mask = []

        i = 0
        while i < len(messages):
            cur_messages = messages[i]
            if cur_messages["role"] == "assistant":
                # Process assistant message
                tokens, loss_mask, attention_mask = self._process_message_tokens(
                    messages, i, i + 1, is_assistant=True, enable_thinking=enable_thinking, tools=tools
                )
                i += 1
            elif cur_messages["role"] == "tool":
                # Process consecutive tool messages
                st = i
                ed = i + 1
                while ed < len(messages) and messages[ed]["role"] == "tool":
                    ed += 1
                tokens, loss_mask, attention_mask = self._process_message_tokens(
                    messages, st, ed, enable_thinking=enable_thinking, tools=tools
                )
                i = ed
            elif cur_messages["role"] in ["user", "system"]:
                # Process user or system message
                if cur_messages["role"] == "system" and i != 0:
                    raise ValueError("System message should be the first message")
                tokens, loss_mask, attention_mask = self._process_message_tokens(
                    messages, i, i + 1, enable_thinking=enable_thinking, tools=tools
                )
                i += 1
            else:
                raise ValueError(f"Unknown role: {cur_messages['role']}")

            # override loss mask with mask in the dataset to handle multi-turn conversation
            override_loss_mask = cur_messages.get("loss_mask", None)
            if override_loss_mask is not None:
                if isinstance(override_loss_mask, np.ndarray):
                    override_loss_mask = override_loss_mask.item()
                assert isinstance(override_loss_mask, int), f"loss_mask should be int, got {type(override_loss_mask)}"
                assert override_loss_mask in [0, 1], f"loss_mask should be 0 or 1, got {override_loss_mask}"
                loss_mask = [override_loss_mask] * len(tokens)

            concat_tokens.extend(tokens)
            concat_loss_mask.extend(loss_mask)
            concat_attention_mask.extend(attention_mask)

        # Validate and convert tokens
        input_ids, loss_mask, attention_mask = self._validate_and_convert_tokens(
            full_tokens[0], concat_tokens, concat_loss_mask, concat_attention_mask
        )

        # encode prompt
        if messages[0]["role"] == "system":
            assert messages[1]["role"] == "user"
            assert messages[2]["role"] == "assistant"
            prompt_message_length = 2
        elif messages[0]["role"] == "user":
            assert messages[1]["role"] == "assistant"
            prompt_message_length = 1
        else:
            raise ValueError(f"Unknown role: {messages[0]['role']}")

        sequence_length = input_ids.shape[0]
        # Handle sequence length
        if self.pad_mode == "right":
            if sequence_length < self.max_length:
                # Pad sequences
                pad_token_id = self.tokenizer.pad_token_id if self.tokenizer.pad_token_id is not None else 0
                padded_input_ids = torch.full((self.max_length - sequence_length,), pad_token_id, dtype=input_ids.dtype)
                padded_attention_mask = torch.zeros((self.max_length - sequence_length,), dtype=attention_mask.dtype)
                padded_loss_mask = torch.zeros((self.max_length - sequence_length,), dtype=loss_mask.dtype)

                input_ids = torch.cat((input_ids, padded_input_ids))
                attention_mask = torch.cat((attention_mask, padded_attention_mask))
                loss_mask = torch.cat((loss_mask, padded_loss_mask))
            elif sequence_length > self.max_length:
                if self.truncation == "left":
                    input_ids = input_ids[-self.max_length :]
                    attention_mask = attention_mask[-self.max_length :]
                    loss_mask = loss_mask[-self.max_length :]
                elif self.truncation == "right":
                    input_ids = input_ids[: self.max_length]
                    attention_mask = attention_mask[: self.max_length]
                    loss_mask = loss_mask[: self.max_length]
                elif self.truncation == "error":
                    raise ValueError(f"{sequence_length=} is larger than {self.max_length=}")
                else:
                    raise ValueError(f"Unknown truncation method {self.truncation}")

            # Create position IDs
            position_ids = torch.arange(len(input_ids), dtype=torch.long)
            # Zero out position IDs for padding
            position_ids = position_ids * attention_mask

            return {
                "input_ids": input_ids,
                "attention_mask": attention_mask,
                "position_ids": position_ids,
                "loss_mask": loss_mask,
            }
        elif self.pad_mode == "left_right":
            assert self.truncation == "error", "Only support error truncation for left_right pad mode"
            prompt_str = self.tokenizer.apply_chat_template(
                messages[:prompt_message_length],
                tools=tools,
                tokenize=False,
                add_generation_prompt=True,
                enable_thinking=enable_thinking,
                **self.apply_chat_template_kwargs,
            )
            prompt_ids = self.tokenizer.encode(prompt_str, add_special_tokens=False)
            prompt_length = len(prompt_ids)
            prompt_ids = input_ids[:prompt_length].unsqueeze(0)
            prompt_attention_mask = attention_mask[:prompt_length].unsqueeze(0)
            prompt_loss_mask = loss_mask[:prompt_length].unsqueeze(0)
            response_ids = input_ids[prompt_length:].unsqueeze(0)
            response_attention_mask = attention_mask[prompt_length:].unsqueeze(0)
            response_loss_mask = loss_mask[prompt_length:].unsqueeze(0)

            assert prompt_loss_mask.sum().item() == 0

            prompt_ids, prompt_attention_mask = postprocess_data(
                input_ids=prompt_ids,
                attention_mask=prompt_attention_mask,
                max_length=self.max_prompt_length,
                pad_token_id=self.tokenizer.pad_token_id,
                left_pad=True,
                truncation=self.truncation,
            )

            response_ids, response_attention_mask = postprocess_data(
                input_ids=response_ids,
                attention_mask=response_attention_mask,
                max_length=self.max_response_length,
                pad_token_id=self.tokenizer.pad_token_id,
                left_pad=False,
                truncation=self.truncation,
            )
            response_loss_mask = pad_sequence_to_length(
                response_loss_mask, max_seq_len=self.max_response_length, pad_token_id=0, left_pad=False
            )

            prompt_ids = prompt_ids[0]
            prompt_attention_mask = prompt_attention_mask[0]
            response_ids = response_ids[0]
            response_attention_mask = response_attention_mask[0]
            response_loss_mask = response_loss_mask[0]

            assert response_attention_mask[0].item() == 1
            assert response_loss_mask[0].item() == 1

            input_ids = torch.cat((prompt_ids, response_ids), dim=0)
            attention_mask = torch.cat((prompt_attention_mask, response_attention_mask), dim=0)
            position_ids = compute_position_id_with_mask(attention_mask)

            return {
                "input_ids": input_ids,
                "attention_mask": attention_mask,
                "position_ids": position_ids,
                "responses": response_ids,
                "response_mask": response_loss_mask,
            }


CHAT_TEMPLATE = '{%- macro render_typescript_type(param_spec, required_params, is_nullable=false) -%}\n    {%- if param_spec.type == "array" -%}\n        {%- if param_spec[\'items\'] -%}\n            {%- if param_spec[\'items\'][\'type\'] == "string" -%}\n                {{- "string[]" }}\n            {%- elif param_spec[\'items\'][\'type\'] == "number" -%}\n                {{- "number[]" }}\n            {%- elif param_spec[\'items\'][\'type\'] == "integer" -%}\n                {{- "number[]" }}\n            {%- elif param_spec[\'items\'][\'type\'] == "boolean" -%}\n                {{- "boolean[]" }}\n            {%- else -%}\n                {%- set inner_type = render_typescript_type(param_spec[\'items\'], required_params) -%}\n                {%- if inner_type == "object | object" or inner_type|length > 50 -%}\n                    {{- "any[]" }}\n                {%- else -%}\n                    {{- inner_type + "[]" }}\n                {%- endif -%}\n            {%- endif -%}\n            {%- if param_spec.nullable -%}\n                {{- " | null" }}\n            {%- endif -%}\n        {%- else -%}\n            {{- "any[]" }}\n            {%- if param_spec.nullable -%}\n                {{- " | null" }}\n            {%- endif -%}\n        {%- endif -%}\n    {%- elif param_spec.type is defined and param_spec.type is iterable and param_spec.type is not string and param_spec.type is not mapping and param_spec.type[0] is defined -%}\n        {#- Handle array of types like ["object", "object"] from Union[dict, list] #}\n        {%- if param_spec.type | length > 1 -%}\n            {{- param_spec.type | join(" | ") }}\n        {%- else -%}\n            {{- param_spec.type[0] }}\n        {%- endif -%}\n    {%- elif param_spec.oneOf -%}\n        {#- Handle oneOf schemas - check for complex unions and fallback to any #}\n        {%- set has_object_variants = false -%}\n        {%- for variant in param_spec.oneOf -%}\n            {%- if variant.type == "object" -%}\n                {%- set has_object_variants = true -%}\n            {%- endif -%}\n        {%- endfor -%}\n        {%- if has_object_variants and param_spec.oneOf|length > 1 -%}\n            {{- "any" }}\n        {%- else -%}\n            {%- for variant in param_spec.oneOf -%}\n                {{- render_typescript_type(variant, required_params) -}}\n                {%- if variant.description %}\n                    {{- "// " + variant.description }}\n                {%- endif -%}\n                {%- if variant.default is defined %}\n                    {{ "// default: " + variant.default|tojson }}\n                {%- endif -%}\n                {%- if not loop.last %}\n                    {{- " | " }}\n                {% endif -%}\n            {%- endfor -%}\n        {%- endif -%}\n    {%- elif param_spec.type == "string" -%}\n        {%- if param_spec.enum -%}\n            {{- \'"\' + param_spec.enum|join(\'" | "\') + \'"\' -}}\n        {%- else -%}\n            {{- "string" }}\n            {%- if param_spec.nullable %}\n                {{- " | null" }}\n            {%- endif -%}\n        {%- endif -%}\n    {%- elif param_spec.type == "number" -%}\n        {{- "number" }}\n    {%- elif param_spec.type == "integer" -%}\n        {{- "number" }}\n    {%- elif param_spec.type == "boolean" -%}\n        {{- "boolean" }}\n    {%- elif param_spec.type == "object" -%}\n        {%- if param_spec.properties -%}\n            {{- "{\\n" }}\n            {%- for prop_name, prop_spec in param_spec.properties.items() -%}\n                {{- prop_name -}}\n                {%- if prop_name not in (param_spec.required or []) -%}\n                    {{- "?" }}\n                {%- endif -%}\n                {{- ": " }}\n                {{ render_typescript_type(prop_spec, param_spec.required or []) }}\n                {%- if not loop.last -%}\n                    {{-", " }}\n                {%- endif -%}\n            {%- endfor -%}\n            {{- "}" }}\n        {%- else -%}\n            {{- "object" }}\n        {%- endif -%}\n    {%- else -%}\n        {{- "any" }}\n    {%- endif -%}\n{%- endmacro -%}\n\n{%- macro render_tools(tools) -%}\n    {%- for tool in tools %}\n        {{- "// " + tool.description + "\\n" }}\n        {{- "type "+ tool.name + " = " }}\n        {%- if tool.parameters and tool.parameters.properties %}\n            {{- "(_: {\\n" }}\n            {%- for param_name, param_spec in tool.parameters.properties.items() %}\n                {%- if param_spec.description %}\n                    {{- "// " + param_spec.description + "\\n" }}\n                {%- endif %}\n                {{- param_name }}\n                {%- if param_name not in (tool.parameters.required or []) -%}\n                    {{- "?" }}\n                {%- endif -%}\n                {{- ": " }}\n                {{- render_typescript_type(param_spec, tool.parameters.required or []) }}\n                {%- if param_spec.default is defined -%}\n                    {%- if param_spec.enum %}\n                        {{- ", // default: " + param_spec.default }}\n                    {%- elif param_spec.oneOf %}\n                        {{- "// default: " + param_spec.default }}\n                    {%- else %}\n                        {{- ", // default: " + param_spec.default|tojson }}\n                    {%- endif -%}\n                {%- endif -%}\n                {%- if not loop.last %}\n                    {{- ",\\n" }}\n                {%- else %}\n                    {{- "\\n" }}\n                {%- endif -%}\n            {%- endfor %}\n            {{- "}) => any;" }}\n        {%- else -%}\n            {{- "() => any;" }}\n        {%- endif -%}\n        {%- if not loop.last -%}\n            {{- "\\n" }}\n        {%- endif -%}\n    {%- endfor %}\n{%- endmacro -%}\n\n{{ bos_token }}\n\n{%- set system_token = \'<|system_start|>\' -%}\n{%- set end_system_token = \'<|system_end|>\' -%}\n{%- set developer_token = \'<|developer_start|>\' -%}\n{%- set end_developer_token = \'<|developer_end|>\' -%}\n{%- set user_token = \'<|user_start|>\' -%}\n{%- set end_user_token = \'<|user_end|>\' -%}\n{%- set assistant_token = \'<|assistant_start|>\' -%}\n{%- set end_assistant_token = \'<|assistant_end|>\' -%}\n{%- set inner_token = \'<|inner_prefix|>\' -%}\n{%- set outer_token = \'<|inner_suffix|>\' -%}\n{%- set tool_calls_token = \'<|tools_prefix|>\' -%}\n{%- set end_tool_calls_token = \'<|tools_suffix|>\' -%}\n\n{%- set ns = namespace(in_assistant=false, in_tool=false, in_inner=false, waiting_for_tool_outputs=false, assistant_format=none) -%}\n\n{%- if messages and messages[0].role == \'system\' -%}\n    {%- if "content" in messages[0] -%}\n        {%- if messages[0].content is string -%}\n            {{ system_token + messages[0].content + end_system_token }}\n        {%- elif messages[0].content is mapping and "text" in messages[0].content -%}\n            {{ system_token + messages[0].content.text + end_system_token }}\n        {%- else -%}\n            {{- raise_exception("Invalid system message") -}}\n        {%- endif -%}\n    {%- else -%}\n        {{- raise_exception("Invalid system message") -}}\n    {%- endif -%}\n    {%- set loop_messages = messages[1:] -%}\n{%- else -%}\n    {{ system_token + \'You are Apertus, a helpful assistant created by the SwissAI initiative.\\nKnowledge cutoff: 2024-04\\nCurrent date: \' + strftime_now(\'%Y-%m-%d\') + end_system_token }}\n    {%- set loop_messages = messages -%}\n{%- endif -%}\n\n{{ developer_token + \'Deliberation: \' }}\n{%- if enable_thinking is defined and enable_thinking -%}\n    {{ \'enabled\\n\' }}\n{%- else -%}\n    {{ \'disabled\\n\' }}\n{%- endif -%}\n{%- if tools is defined and tools -%}\n    {{ \'Tool Capabilities:\\n\' + render_tools(tools) }}\n{%- else -%}\n    {{ \'Tool Capabilities: disabled\' }}\n{%- endif -%}\n{{ end_developer_token }}\n\n{%- for message in loop_messages -%}\n    {%- if message.role == \'user\' -%}\n        {%- set ns.in_inner = false -%}\n        {%- if ns.in_tool -%}\n            {{ \']\' }}\n            {%- set ns.in_tool = false -%}\n        {%- endif -%}\n        {%- if ns.in_assistant -%}\n            {{ end_assistant_token }}\n            {%- set ns.in_assistant = false -%}\n        {%- endif -%}\n        {%- if "content" in message -%}\n            {{ user_token }}\n            {%- if message.content is string -%}\n                {{ message.content }}\n            {%- elif message.content is mapping and "parts" in message.content -%}\n                {%- set parts = message.content.parts -%}\n                {%- for part in parts -%}\n                    {%- if part.type == "text" -%}\n                        {{ part.text }}\n                    {%- else -%}\n                        {{- raise_exception("Invalid user part: " + part.type) -}}\n                    {%- endif -%}\n                {%- endfor -%}\n            {%- else -%}\n                {{- raise_exception("Invalid user message: " + message.role) -}}\n            {%- endif -%}\n            {{ end_user_token }}\n        {%- endif -%}\n    {%- elif message.role == \'assistant\' -%}\n        {%- if not ns.in_assistant -%}\n            {{ assistant_token }}\n            {%- set ns.in_assistant = true -%}\n        {%- endif -%}\n        {%- if "content" in message -%}\n            {%- if message.content is string and (ns.assistant_format is none or ns.assistant_format == "string") -%}\n                {%- if ns.in_tool -%}\n                    {{ \']\' }}\n                    {%- set ns.in_tool = false -%}\n                {%- endif -%}\n                {%- set ns.assistant_format = "string" -%}\n                {{ message.content }}\n            {%- elif message.content is mapping and "blocks" in message.content and (ns.assistant_format is none or ns.assistant_format == "mapping") -%}\n                {%- set ns.assistant_format = "mapping" -%}\n                {%- set blocks = message.content.blocks -%}\n                {%- for block in blocks -%}\n                    {%- if block.type == \'thoughts\' -%}\n                        {%- if ns.in_tool -%}\n                            {{ \']\' }}\n                            {%- set ns.in_tool = false -%}\n                        {%- endif -%}\n                        {%- if not ns.in_inner -%}\n                            {%- set ns.in_inner = true -%}\n                            {{ inner_token }}\n                        {%- endif -%}\n                        {{ block.text }}\n                    {%- elif block.type == \'tool_calls\' -%}\n                        {%- if ns.in_tool -%}\n                            {{ \']\' }}\n                            {%- set ns.in_tool = false -%}\n                        {%- endif -%}\n                        {%- if ns.in_inner and not loop.first and block.calls|length == 1 and block.calls[0].name == \'display_answers\' -%}\n                            {%- set ns.in_inner = false -%}\n                            {{ outer_token }}\n                        {%- endif -%}\n                        {{ tool_calls_token + \'[\' }}\n                        {%- for tool_call in block.calls -%}\n                            {{- \'{"\' + tool_call.name + \'": \' + tool_call.arguments + \'}\' }}\n                            {%- if not loop.last -%}\n                                {{- ", " }}\n                            {%- endif -%}\n                        {%- endfor -%}\n                        {{ \']\' + end_tool_calls_token }}\n                        {%- set ns.waiting_for_tool_outputs = true -%}\n                    {%- elif block.type == \'tool_outputs\' -%}\n                        {%- if ns.in_tool -%}\n                            {{- raise_exception("Cannot have both tool outputs as separate messages and tool outputs as blocks") -}}\n                        {%- endif -%}\n                        {{ \'[\' }}\n                        {%- for tool_output in block.outputs -%}\n                            {{- tool_output.output }}\n                            {%- if not loop.last -%}\n                                {{- ", " }}\n                            {%- endif -%}\n                        {%- endfor -%}\n                        {{- \']\' }}\n                        {%- set ns.waiting_for_tool_outputs = false -%}\n                    {%- elif block.type == \'response\' -%}\n                        {%- if ns.in_tool -%}\n                            {{ \']\' }}\n                            {%- set ns.in_tool = false -%}\n                        {%- endif -%}\n                        {%- if (not loop.first and ns.in_inner) or (ns.in_assistant and ns.in_inner) -%}\n                            {%- set ns.in_inner = false -%}\n                            {{ outer_token }}\n                        {%- endif -%}\n                        {{ block.text }}\n                    {%- else -%}\n                        {{- raise_exception("Invalid assistant block type: " + block.type) -}}\n                    {%- endif -%}\n                {%- endfor -%}\n            {%- else -%}\n                {{- raise_exception("Invalid assistant content") -}}\n            {%- endif -%}\n        {%- else -%}\n            {{- raise_exception("Invalid assistant message") -}}\n        {%- endif -%}\n        {%- if "tool_calls" in message and message.tool_calls -%}\n            {{ tool_calls_token + \'[\' }}\n            {%- for tool_call in message.tool_calls -%}\n                {%- if tool_call.type == \'function\' -%}\n                    {%- set function = tool_call.function -%}\n                    {{- \'{"\' + function.name + \'": \' + function.arguments + \'}\' }}\n                    {%- if not loop.last -%}\n                        {{- ", " }}\n                    {%- endif -%}\n                {%- else -%}\n                    {{- raise_exception("Invalid tool call type: " + tool_call.type) -}}\n                {%- endif -%}\n            {%- endfor -%}\n            {{ \']\' + end_tool_calls_token }}\n            {%- set ns.waiting_for_tool_outputs = true -%}\n        {%- endif -%}\n    {%- elif message.role == \'tool\' -%}\n        {%- if not ns.in_assistant -%}\n            {{- raise_exception("Tool message outside of assistant") -}}\n        {%- endif -%}\n        {%- if not ns.in_tool -%}\n            {{ \'[\' }}\n            {%- set ns.in_tool = true -%}\n        {%- else -%}\n            {{ ", "}}\n        {%- endif -%}\n        {{ message.content }}\n        {%- set ns.waiting_for_tool_outputs = false -%}\n    {%- else -%}\n        {{- raise_exception("Invalid message role") -}}\n    {%- endif -%}\n{%- endfor -%}\n{%- if ns.in_tool -%}\n    {{ \']\' }}\n{%- endif -%}\n{%- if ns.in_assistant and not (continue_assistant_message is defined and continue_assistant_message) and not ns.waiting_for_tool_outputs -%}\n    {{ end_assistant_token }}\n{%- endif -%}\n{%- if add_generation_prompt -%}\n    {{ assistant_token }}\n{%- endif -%}\n'

SYSTEM_TOKEN = 61
END_SYSTEM_TOKEN = 62
DEVELOPER_TOKEN = 63
END_DEVELOPER_TOKEN = 64
USER_TOKEN = 65
END_USER_TOKEN = 66
ASSISTANT_TOKEN = 67
END_ASSISTANT_TOKEN = 68
INNER_TOKEN = 69
OUTER_TOKEN = 70
TOOL_CALLS_TOKEN = 71
END_TOOL_CALLS_TOKEN = 72


class ApertusSFTDataset(MultiTurnSFTDataset):
    def __init__(self, parquet_files: str | list[str], tokenizer, config=None):
        super().__init__(parquet_files, tokenizer, config)

        self.only_tools_special_tokens = config.get("only_tools_special_tokens", False)

    def _special_tokens_mask(self, input_ids: np.ndarray) -> np.ndarray:
        return (
            (input_ids == END_ASSISTANT_TOKEN)
            | (input_ids == INNER_TOKEN)
            | (input_ids == OUTER_TOKEN)
            | (input_ids == TOOL_CALLS_TOKEN)
            | (input_ids == END_TOOL_CALLS_TOKEN)
        )

    def __getitem__(self, item):
        tokenizer = self.tokenizer
        messages = loads(self.messages[item]) if self.messages is not None and self.messages[item] != "" else None
        tools = loads(self.tools[item]) if self.tools is not None and self.tools[item] != "" else None
        enable_thinking = self.enable_thinking[item] if self.enable_thinking is not None else None
        rollout_params = (
            loads(self.rollout_params[item])
            if self.rollout_params is not None and self.rollout_params[item] != ""
            else {}
        )
        rollout_apply_chat_template_kwargs = rollout_params.get("apply_chat_template_kwargs", {})

        output = tokenizer.apply_chat_template(
            messages,
            tools=tools,
            enable_thinking=enable_thinking,
            add_generation_prompt=self.add_generation_prompt
            and not rollout_apply_chat_template_kwargs.get("continue_assistant_message", False),
            chat_template=CHAT_TEMPLATE,
            padding="max_length",
            max_length=self.max_length,
            return_tensors="np",
            return_dict=True,
            **self.apply_chat_template_kwargs,
            **rollout_apply_chat_template_kwargs,
        )

        input_ids = np.reshape(output["input_ids"], -1)
        attention_mask = np.reshape(output["attention_mask"], -1)
        attention_mask_tensor = torch.from_numpy(attention_mask)

        if self.only_tools_special_tokens and tools is not None:
            start_tool_calls = np.cumsum(input_ids == TOOL_CALLS_TOKEN, axis=0) - (
                input_ids == TOOL_CALLS_TOKEN
            ).astype(np.int32)
            end_tool_calls = np.cumsum(input_ids == END_TOOL_CALLS_TOKEN, axis=0) - (
                input_ids == END_TOOL_CALLS_TOKEN
            ).astype(np.int32)
            mask = np.logical_not((start_tool_calls == end_tool_calls)) | self._special_tokens_mask(input_ids)
        else:
            tool_outputs_lengths = []
            if tools is not None:
                for message in messages:
                    if message["role"] == "assistant":
                        for block in message["content"]["blocks"]:
                            if block["type"] == "tool_outputs":
                                tool_outputs = block["outputs"]

                                # We format the tool outputs as it is formatted in the chat template
                                tool_outputs_str = (
                                    f"[{', '.join([tool_output['output'] for tool_output in tool_outputs])}]"
                                )
                                tool_outputs_lengths.append(
                                    len(tokenizer.encode(tool_outputs_str, add_special_tokens=False))
                                )

            # We use cumsum to get the different turns
            # Then we subtract to remove the first token of each turn because we don't want to train on it
            start_assistant = np.cumsum(input_ids == ASSISTANT_TOKEN, axis=0) - (input_ids == ASSISTANT_TOKEN).astype(
                np.int32
            )
            end_assistant = np.cumsum(input_ids == END_ASSISTANT_TOKEN, axis=0) - (
                input_ids == END_ASSISTANT_TOKEN
            ).astype(np.int32)

            # The mask is 1 if the token is not an assistant token and 0 otherwise
            mask = start_assistant == end_assistant

            if len(tool_outputs_lengths) > 0:
                # We are searching the end of the tool calls (or the start of tool outputs) in the assistant tokens
                end_tool_calls = (start_assistant != end_assistant) & (input_ids == END_TOOL_CALLS_TOKEN)

                start_tool_output_indices = np.arange(stop=input_ids.shape[0])[end_tool_calls] + 1
                for i, tol in zip(start_tool_output_indices, tool_outputs_lengths):
                    mask[i : i + tol] = 1

            mask = np.logical_not(mask)

        return {
            "input_ids": torch.from_numpy(input_ids),
            "attention_mask": attention_mask_tensor,
            "position_ids": compute_position_id_with_mask(attention_mask_tensor),
            "responses": torch.from_numpy(input_ids[1:]),
            "response_mask": torch.from_numpy(mask[1:].astype(np.int32)),
            "rollout_params": rollout_params,
        }
