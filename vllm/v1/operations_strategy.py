from dataclasses import dataclass
from typing import List, Optional

import torch

from vllm.v1 import operations
from vllm.v1.operations import Operation

@dataclass
class OperationsStrategy:
    operations: List[Operation]
    tbo_delta_stages: Optional[int] = None

    @classmethod
    def concat(cls, items: List["OperationsStrategy"]) -> "OperationsStrategy":
        return OperationsStrategy(
            operations=[x for item in items for x in item.operations],
            tbo_delta_stages=_assert_all_same(
                [item.tbo_delta_stages for item in items]
            ),
        )

    @staticmethod
    def init_new_tbo(
        layers: torch.nn.ModuleList,
        forward_mode: Optional[int] = None,
    ) -> "OperationsStrategy":
        layer_name = layers[0].__class__.__name__
        if layer_name == "DeepseekV2DecoderLayer":
            return OperationsStrategy.concat(
                [
                    _compute_moe_deepseek_layer_operations_strategy_tbo(
                        layer, forward_mode
                    )
                    for layer in layers
                ]
            )
        else:
            raise NotImplementedError


def _assert_all_same(items: List):
    assert all(item == items[0] for item in items)
    return items[0]


# -------------------------------- Strategy for DeepSeek ---------------------------------------

def _compute_mlp_deepseek_layer_operations_strategy_tbo(
    layer: torch.nn.Module,
) -> OperationsStrategy:
    assert (not layer.is_layer_sparse), "this strategy is for dense layer"
    return OperationsStrategy(
        # assume tp size 1
        # intended for layer 0-2, but not used
        tbo_delta_stages=0,
        operations=[
            layer.op_forward,
            operations.YieldOperation(),
        ],
    )

def _compute_moe_deepseek_layer_operations_strategy_tbo(
    layer: torch.nn.Module,
    forward_mode: Optional[int] = None,
) -> OperationsStrategy:
    if layer.is_layer_sparse:
        # forward_mode does not exist in vllm for now... improvising...
        if forward_mode is None:
            return _compute_moe_deepseek_blog_decode(layer)
        elif forward_mode == 0: #prefill ex
            return _compute_moe_deepseek_blog_prefill(layer)
        elif forward_mode == 1: #decode ex
            return _compute_moe_deepseek_blog_decode(layer)
        else:
            raise NotImplementedError(f"Unsupported {forward_mode=}")
    else: # intended for layer 0-2, but not used FIXME : remove?
        return _compute_mlp_deepseek_layer_operations_strategy_tbo(layer)

def _compute_moe_deepseek_blog_prefill(layer):
    return OperationsStrategy(
    # did not divide attention, norms
    # no prefill tbo-activated for now, tp size should be 1 for now
        tbo_delta_stages=0,
        operations=[
            layer.op_forward_pre_mlp,
            layer.mlp.op_enter_moe,
            layer.mlp.op_gate,
            layer.mlp.op_select_experts,
            layer.mlp.op_dispatch,
            operations.YieldOperation(),
            layer.mlp.op_shared_experts,
            layer.mlp.op_experts,
            layer.mlp.op_combine,
            operations.YieldOperation(),
            layer.mlp.op_moe_finalize_and_exit,
            layer.op_forward_post_mlp,
        ],
    )


def _compute_moe_deepseek_blog_decode(layer):
    return OperationsStrategy(
    # did not divide attention, norms
    # tp size should be 1 for now
        tbo_delta_stages=0,
        operations=[
            layer.op_forward_pre_mlp,
            layer.mlp.op_enter_moe,
            layer.mlp.op_gate,
            layer.mlp.op_select_experts,
            layer.mlp.op_dispatch,
            operations.YieldOperation(),
            layer.mlp.op_shared_experts,
            layer.mlp.op_experts,
            layer.mlp.op_combine,
            operations.YieldOperation(),
            layer.mlp.op_moe_finalize_and_exit,
            layer.op_forward_post_mlp,
        ],
    )
