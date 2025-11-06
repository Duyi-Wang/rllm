# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from typing import Optional

import torch

from vllm.platforms import current_platform
from vllm.utils import direct_register_custom_op, is_torch_equal_or_newer
import vllm.envs as envs
from aiter.mla import mla_decode_fwd

def get_aiter_mla_metadata(max_batch_size: int, block_size: int,
                           max_block_per_batch: int,
                           device: torch.device) -> tuple[torch.Tensor, ...]:
    paged_kv_indices = torch.zeros(max_batch_size * max_block_per_batch,
                                   dtype=torch.int32,
                                   device=device)
    paged_kv_indptr = torch.zeros(max_batch_size + 1,
                                  dtype=torch.int32,
                                  device=device)
    paged_kv_last_page_lens = torch.full((max_batch_size, ),
                                         block_size,
                                         dtype=torch.int32)
    qo_indptr = torch.zeros(max_batch_size + 1, dtype=torch.int, device=device)
    return paged_kv_indices, paged_kv_indptr, paged_kv_last_page_lens, qo_indptr


def aiter_mla_decode_fwd(
    q,
    kv_buffer,
    o,
    qo_indptr,
    kv_indptr,
    kv_indices,
    kv_last_page_lens,
    max_seqlen_q,
    sm_scale=None,  # 1.0 / (qk_head_dim**0.5)
    logit_cap=0.0,
    work_metadata=None,
    work_indptr=None,
    work_info_set=None,
    reduce_indptr=None,
    reduce_final_map=None,
    reduce_partial_map=None,
    q_scale=None,
    kv_scale=None,
):
    mla_decode_fwd(
        q,
        kv_buffer.view(-1, 1, 1, q.shape[-1]),
        o,
        qo_indptr,
        kv_indptr,
        kv_indices,
        kv_last_page_lens,
        max_seqlen_q,
        sm_scale=sm_scale,
        logit_cap=logit_cap,
        work_meta_data=work_metadata,
        work_indptr=work_indptr,
        work_info_set=work_info_set,
        reduce_indptr=reduce_indptr,
        reduce_final_map=reduce_final_map,
        reduce_partial_map=reduce_partial_map,
        q_scale=q_scale,
        kv_scale=kv_scale,
    )


def mla_decode_fwd_impl(
    q: torch.Tensor,
    kv_buffer: torch.Tensor,
    o: torch.Tensor,
    qo_indptr: torch.Tensor,
    kv_indptr: torch.Tensor,
    kv_indices: torch.Tensor,
    kv_last_page_lens: torch.Tensor,
    max_seqlen_q: int,
    sm_scale: Optional[float] = None,
    logit_cap: Optional[float] = 0.0,
    work_metadata: Optional[torch.Tensor] = None,
    work_indptr: Optional[torch.Tensor] = None,
    work_info_set: Optional[torch.Tensor] = None,
    reduce_indptr: Optional[torch.Tensor] = None,
    reduce_final_map: Optional[torch.Tensor] = None,
    reduce_partial_map: Optional[torch.Tensor] = None,
    q_scale: Optional[torch.Tensor] = None,
    kv_scale: Optional[torch.Tensor] = None,
) -> None:
    from aiter.mla import mla_decode_fwd

    mla_decode_fwd(
        q,
        kv_buffer.view(-1, 1, 1, q.shape[-1]),
        o,
        qo_indptr,
        kv_indptr,
        kv_indices,
        kv_last_page_lens,
        max_seqlen_q,
        sm_scale=sm_scale,
        logit_cap=logit_cap,
        work_meta_data=work_metadata,
        work_indptr=work_indptr,
        work_info_set=work_info_set,
        reduce_indptr=reduce_indptr,
        reduce_final_map=reduce_final_map,
        reduce_partial_map=reduce_partial_map,
        q_scale=q_scale,
        kv_scale=kv_scale,
    )


def mla_decode_fwd_fake(
    q: torch.Tensor,
    kv_buffer: torch.Tensor,
    o: torch.Tensor,
    qo_indptr: torch.Tensor,
    kv_indptr: torch.Tensor,
    kv_indices: torch.Tensor,
    kv_last_page_lens: torch.Tensor,
    max_seqlen_q: int,
    sm_scale: Optional[float] = None,
    logit_cap: Optional[float] = 0.0,
    work_metadata: Optional[torch.Tensor] = None,
    work_indptr: Optional[torch.Tensor] = None,
    work_info_set: Optional[torch.Tensor] = None,
    reduce_indptr: Optional[torch.Tensor] = None,
    reduce_final_map: Optional[torch.Tensor] = None,
    reduce_partial_map: Optional[torch.Tensor] = None,
    q_scale: Optional[torch.Tensor] = None,
    kv_scale: Optional[torch.Tensor] = None,
) -> None:
    pass


if current_platform.is_rocm():
    if is_torch_equal_or_newer("2.7.0"):
        tags = ()
    else:
        tags = (torch.Tag.needs_fixed_stride_order, ),
    direct_register_custom_op(op_name="rocm_aiter_mla_decode_fwd",
                              op_func=mla_decode_fwd_impl,
                              mutates_args=["o"],
                              fake_impl=mla_decode_fwd_fake,
                              tags=tags)
