# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
from typing import Optional, Dict, List
from vllm.v1.operations import execute_overlapped_operations
from vllm.v1.operations_strategy import OperationsStrategy
from vllm.v1.worker.ubatching import UBatchContext, get_ubatch_context

import torch

def model_forward_maybe_tbo(
    layers,
    positions: torch.Tensor,
    hidden_states: torch.Tensor,
    residual: Optional[torch.Tensor],
):
    # NOTE: only dp and ep are supported
    inputs = dict(
        positions=positions,
        hidden_states=hidden_states,
        residual=residual,
    )
    operations_strategy = OperationsStrategy.init_new_tbo(
        layers,
    )

    return _model_forward_tbo(
        inputs=inputs,
        operations_strategy=operations_strategy,
    )

def _model_forward_tbo(
    inputs,
    operations_strategy: OperationsStrategy,
):
    inputs_arr = _model_forward_tbo_split_inputs(
        **inputs,
    )
    del inputs

    outputs_arr = execute_overlapped_operations(
        inputs_arr=inputs_arr,
        operations_arr=[operations_strategy.operations] * 2,
        delta_stages=[0, operations_strategy.tbo_delta_stages],
    )

    return _model_forward_tbo_merge_outputs(*outputs_arr)

# divide model inputs for microbatching
def _model_forward_tbo_split_inputs(
    hidden_states: torch.Tensor,
    residual: torch.Tensor,
    positions: torch.Tensor,
) -> List[Dict]:
    # NOTE : TP is not supported for now
    inputs_arr = [
        _model_forward_filter_inputs(
            hidden_states=hidden_states,
            residual=residual,
            positions=positions,
            ubatch_ctx=get_ubatch_context(i)
        )
        for i in (0, 1)
    ]
    return inputs_arr

# per-microbatch inputs
def _model_forward_filter_inputs(
    hidden_states: torch.Tensor,
    residual: torch.Tensor,
    positions: torch.Tensor,
    ubatch_ctx: UBatchContext,
) -> Dict:
    token_slice = ubatch_ctx.get_token_slice()
    return dict(
        hidden_states=hidden_states[token_slice],
        residual=None if residual is None else residual[token_slice],
        positions=positions[token_slice],
    )

# merge microbatch outputs
def _model_forward_tbo_merge_outputs(output_a, output_b):
    def _handle_key(name):
        value_a = output_a[name]
        value_b = output_b[name]
        assert (value_a is None) == (value_b is None)
        if value_a is None:
            return None
        return torch.concat([value_a, value_b], dim=0)

    return _handle_key("hidden_states"), _handle_key("residual")
