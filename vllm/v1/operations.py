import os
from contextlib import contextmanager
from dataclasses import dataclass
from typing import Any, Callable, Dict, Generator, List, Sequence, Union

from vllm.logger import init_logger
from vllm.v1.worker.ubatching import UBatchContext, get_ubatch_context
from vllm.forward_context import override_forward_context

import torch

logger = init_logger(__name__)

# FIXME did not delete this for future use, NOTE: seems nvtx is NVIDIA package
_ENABLE_PROFILE = bool(int(os.environ.get("VLLM_OPERATIONS_ENABLE_PROFILE", "0")))

if _ENABLE_PROFILE:
    import nvtx

def execute_operations(inputs, operations):
    stages = _convert_operations_to_stages(operations)
    executor = _StageExecutor("primary", stages, inputs=inputs)
    for _ in range(executor.num_stages):
        executor.next()
    assert executor.done
    return executor.output


def execute_overlapped_operations(
    inputs_arr: Sequence,
    operations_arr: Sequence,
    delta_stages: Sequence[int],
) -> Sequence:
    # Make it explicit for clarity; if we need multi-batch overlap, this can be generalized
    inputs_a, inputs_b = inputs_arr
    operations_a, operations_b = operations_arr
    delta_stage_a, delta_stage_b = delta_stages
    assert delta_stage_a == 0
    delta_stage = delta_stage_b

    stages_a = _convert_operations_to_stages(operations_a)
    stages_b = _convert_operations_to_stages(operations_b)

    # NOTE : only two microbatches are supported
    executor_a = _StageExecutor("a", stages_a,
                                ubatch_ctx=get_ubatch_context(0),
                                inputs=inputs_a)
    executor_b = _StageExecutor("b", stages_b,
                                ubatch_ctx=get_ubatch_context(1),
                                inputs=inputs_b)
    compute_event_a, comm_event_a = None, None
    compute_event_b, comm_event_b = None, None

    for _ in range(delta_stage):
        compute_event_a, comm_event_a = executor_a.next_with_overriding_contexts(compute_event_b,
            comm_event_b)

    for _ in range(executor_a.num_stages - delta_stage):
        compute_event_a, comm_event_a = executor_a.next_with_overriding_contexts(compute_event_b,
            comm_event_b)
        compute_event_b, comm_event_b = executor_b.next_with_overriding_contexts(compute_event_a,
            comm_event_a)

    for _ in range(delta_stage):
        compute_event_b, comm_event_b = \
            executor_b.next_with_overriding_contexts(compute_event_a, comm_event_a)

    assert executor_a.done and executor_b.done
    return [executor_a.output, executor_b.output]


class YieldOperation:
    pass


@dataclass
class ExecutionOperation:
    debug_name: str
    fn: Callable


Operation = Union[YieldOperation, ExecutionOperation, Callable]
Stage = List[ExecutionOperation]


class _StageExecutor:
    def __init__(self, debug_name: str, stages: List[Stage], ubatch_ctx: UBatchContext, inputs):
        self._debug_name = debug_name
        self._stages = stages
        self._index = 0
        self._stage_state = _StateDict()
        self._fwd_ctx = ubatch_ctx.forward_context
        self._stage_output = inputs
        self._stage_state.ubatch_ctx = ubatch_ctx

    # override global variable forward_context with forward_context of ubatch
    def next_with_overriding_contexts(self, wait_compute_event, wait_comm_event):
        logger.debug(f"Execute microbatch with ubatch_id: {self._stage_state.ubatch_ctx.id}")
        ubatch_ctx = self._stage_state.ubatch_ctx
        if wait_compute_event:
            ubatch_ctx.wait_event(wait_compute_event)
        if wait_comm_event:
            ubatch_ctx.register_hook(lambda: ubatch_ctx.wait_event(wait_comm_event))
        with override_forward_context(self._fwd_ctx), ubatch_ctx:
            self.next()
        return ubatch_ctx.compute_event, ubatch_ctx.comm_event

    def next(self):
        assert not self.done

        stage = self._stages[self._index]
        with _annotate_region(debug_name=f"{self._debug_name}{self._index}"):
            for op in stage:
                with _annotate_region(debug_name=op.debug_name):
                    self._stage_output = op.fn(
                        state=self._stage_state,
                        **(
                            self._stage_output if self._stage_output is not None else {}
                        ),
                    )

        self._index += 1

    @property
    def output(self):
        assert self.done
        return self._stage_output

    @property
    def done(self):
        return self._index >= self.num_stages

    @property
    def num_stages(self):
        return len(self._stages)


@contextmanager
def _annotate_region(debug_name):
    if _ENABLE_PROFILE:
        with torch.autograd.profiler.record_function(debug_name):
            with nvtx.annotate(debug_name):
                yield
    else:
        yield


class _StateDict:
    def __init__(self):
        self._data = {}

    def __setattr__(self, key, value):
        if key == "_data":
            super().__setattr__(key, value)
            return
        assert (
            key not in self._data
        ), f"`{key}` already exist, are you sure you want to override it?"
        self._data[key] = value

    def __getattr__(self, item):
        return self._data[item]

    def __delattr__(self, item):
        del self._data[item]

    def pop(self, item):
        return self._data.pop(item)

    def update(self, values: Dict[str, Any]):
        for k, v in values.items():
            setattr(self, k, v)

    def get(self, item):
        return self._data.get(item)

    def clear(self, expect_keys: Sequence[str]):
        if set(self._data.keys()) != set(expect_keys):
            raise Exception(
                f"Unexpected keys when clearning. This may indicate you do not release memory early enough but leave it to here. {list(self._data.keys())=} {expect_keys=}"
            )

        self._data.clear()


def _convert_operations_to_stages(operations: List[Operation]) -> List[Stage]:
    operations = _decorate_operations(operations)
    operation_chunks = list(
        _chunk_by_separator(operations, lambda op: isinstance(op, YieldOperation))
    )
    assert all(len(chunk) > 0 for chunk in operation_chunks)
    return operation_chunks


def _chunk_by_separator(
    items: List[Any], is_separator: Callable[[Any], bool]
) -> Generator[List[Any], None, None]:
    pending_items = []
    for item in items:
        if is_separator(item):
            yield pending_items
            pending_items = []
        else:
            pending_items.append(item)
    if len(pending_items) > 0:
        yield pending_items


def _decorate_operations(operations: List[Operation], debug_name_prefix: str = ""):
    return [_decorate_operation(op, debug_name_prefix) for op in operations]


def _decorate_operation(operation: Operation, debug_name_prefix: str):
    if isinstance(operation, YieldOperation):
        return operation
    return ExecutionOperation(
        debug_name=debug_name_prefix
        + getattr(operation, "__name__", "unknown").replace("op_", ""),
        fn=operation,
    )
