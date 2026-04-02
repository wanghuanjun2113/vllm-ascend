#
# Copyright (c) 2025 Huawei Technologies Co., Ltd. All Rights Reserved.
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
# This file is a part of the vllm-ascend project.
#
# Todo: Once https://github.com/vllm-project/vllm/pull/23553 is merged in vllm. Remove this model register.
import types
from typing import Any

import torch


_MISSING = object()


def get_model_arch_config(model: Any) -> Any:
    config = getattr(model, "config", None)
    if config is None:
        raise AttributeError("model.config is required for EPLB.")

    get_text_config = getattr(config, "get_text_config", None)
    if callable(get_text_config):
        for kwargs in ({"decoder": True}, {}):
            try:
                text_config = get_text_config(**kwargs)
            except TypeError:
                continue
            if text_config is not None:
                config = text_config
                break

    while True:
        text_config = getattr(config, "text_config", None)
        if text_config is None or text_config is config:
            break
        config = text_config

    return config


def get_model_arch_attr(model: Any, attr_name: str, default: Any = _MISSING) -> Any:
    for config in (get_model_arch_config(model), getattr(model, "config", None)):
        if config is not None and hasattr(config, attr_name):
            return getattr(config, attr_name)

    if default is not _MISSING:
        return default

    raise AttributeError(f"Unable to resolve `{attr_name}` from model config for EPLB.")


def get_num_dense_layers(model: Any) -> int:
    return get_model_arch_attr(model, "first_k_dense_replace", 0)


def get_num_hidden_layers(model: Any) -> int:
    return get_model_arch_attr(model, "num_hidden_layers")


def get_expert_map(self, layer_id):
    return self.model.layers[layer_id].mlp.experts.expert_map


def get_log2phy_map(self, layer_id):
    return self.model.layers[layer_id].mlp.experts.get_log2phy_map()


def get_all_moe_loads(self):
    num_dense_layers = get_num_dense_layers(self)
    num_layers = get_num_hidden_layers(self)
    all_moe_loads = torch.stack(
        [self.model.layers[layer_id].mlp.experts.moe_load for layer_id in range(num_dense_layers, num_layers)],
        dim=0,
    )
    return all_moe_loads


def clear_all_moe_loads(self):
    num_dense_layers = get_num_dense_layers(self)
    num_layers = get_num_hidden_layers(self)
    for layer_id in range(num_dense_layers, num_layers):
        self.model.layers[layer_id].mlp.experts.clear_moe_load()


def model_register(model):
    model.get_expert_map = types.MethodType(get_expert_map, model)
    model.get_log2phy_map = types.MethodType(get_log2phy_map, model)
    model.get_all_moe_loads = types.MethodType(get_all_moe_loads, model)
    model.clear_all_moe_loads = types.MethodType(clear_all_moe_loads, model)
