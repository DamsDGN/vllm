# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from typing import Any

import torch

from vllm.model_executor.layers.fused_moe import (
    RoutedExperts,
)
from vllm.model_executor.layers.linear import LinearBase, UnquantizedLinearMethod
from vllm.model_executor.layers.quantization import QuantizationMethods
from vllm.model_executor.layers.quantization.base_config import (
    QuantizationConfig,
    QuantizeMethodBase,
)
from vllm.model_executor.layers.quantization.online.int8 import (
    Int8OnlineMoEMethod,
    Int8SerializedMoEMethod,
)


class ExpertsInt8Config(QuantizationConfig):
    """Online int8 quantization for MoE expert weights.
    Linear layers are left unquantized.

    Backward-compatible config for ``--quantization experts_int8``.
    Prefer ``--quantization int8_per_channel``

    A checkpoint whose config carries ``quantization_config:
    {"quant_method": "experts_int8"}`` already stores the int8 expert weights
    and their per-row scales, as this method would have computed them.
    """

    def __init__(self, is_checkpoint_serialized: bool = False) -> None:
        super().__init__()
        self.is_checkpoint_serialized = is_checkpoint_serialized

    @classmethod
    def get_name(cls) -> QuantizationMethods:
        return "experts_int8"

    @classmethod
    def get_supported_act_dtypes(cls) -> list[torch.dtype]:
        return [torch.bfloat16, torch.half]

    @classmethod
    def get_min_capability(cls) -> int:
        return 80

    @classmethod
    def get_config_filenames(cls) -> list[str]:
        return []

    @classmethod
    def from_config(cls, config: dict[str, Any]) -> "ExpertsInt8Config":
        return cls(is_checkpoint_serialized=config.get("quant_method") == "experts_int8")

    def get_quant_method(
        self, layer: torch.nn.Module, prefix: str
    ) -> "QuantizeMethodBase | None":
        if isinstance(layer, LinearBase):
            return UnquantizedLinearMethod()
        elif isinstance(layer, RoutedExperts):
            if self.is_checkpoint_serialized:
                return Int8SerializedMoEMethod(moe=layer.moe_config)
            return Int8OnlineMoEMethod(moe=layer.moe_config)
        return None
