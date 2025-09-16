# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
from typing import Optional, List

import torch

from vllm.forward_context import ForwardContext


class UBatchContext:
    """
    Context manager for micro-batching synchronization using threading events.
    """

    def __init__(self,
                 id: int,
                 stream: torch.cuda.Stream,
                 forward_context: ForwardContext,
                 compute_event: torch.cuda.Event,
                 comm_event: torch.cuda.Event,
                 token_slice: slice = None):
        self.id = id
        self.stream = stream
        self.forward_context = forward_context
        self.compute_event = compute_event
        self.comm_event = comm_event
        self.token_slice = token_slice
        self._prev_stream = torch.cuda.current_stream()
        self.hook = None

    def __enter__(self):
        torch.cuda.set_stream(self.stream)
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.hook = None
        torch.cuda.set_stream(self._prev_stream)

    def record_compute_event(self):
        self.compute_event.record(self.stream)

    def record_comm_event(self):
        self.comm_event.record(self.stream)

    def wait_event(self, wait_event):
        self.stream.wait_event(wait_event)

    def register_hook(self, hook):
        self.hook = hook 

    def maybe_run_hook(self):
        if self.hook is not None:
            self.hook()

    def set_token_slice(self, token_slice: slice):
        self.token_slice = token_slice

    def get_token_slice(self) -> slice:
        return self.token_slice

# global variable, set when doing tbo, clear when done
_ubatch_contexts: list[UBatchContext] = None

def set_ubatch_contexts(ubatch_ctxs: list[UBatchContext]):
    global _ubatch_contexts
    _ubatch_contexts = ubatch_ctxs

def unset_ubatch_contexts():
    global _ubatch_contexts
    _ubatch_contexts = None

def ubatch_context_exists() -> bool:
    return _ubatch_contexts is not None

def get_ubatch_context(i: int) -> UBatchContext:
    """Get the current ubatch context."""
    assert _ubatch_contexts is not None, (
        "ubatch context is not set. "
        "Please use `set_ubatch_contexts` to set the ubatch context.")
    return _ubatch_contexts[i]

def make_ubatch_contexts(
    num_micro_batches: int,
    streams: List[torch.cuda.Stream],
    forward_contexts: list[ForwardContext],
) -> list[UBatchContext]:
    assert num_micro_batches == 2, "only been tested with 2 micro-batches"
    """
    Create a context manager for micro-batching synchronization.
    """
    compute_events = [
        torch.cuda.Event() for _ in range(num_micro_batches)
    ]
    comm_events = [
        torch.cuda.Event() for _ in range(num_micro_batches)
    ]
    assert len(forward_contexts) == 2

    ctxs = []
    for i in range(num_micro_batches):
        ctx = UBatchContext(id=i,
                            stream=streams[i],
                            forward_context=forward_contexts[i],
                            compute_event=compute_events[i],
                            comm_event=comm_events[i])
        ctxs.append(ctx)

    return ctxs
