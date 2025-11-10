# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
import contextlib
import math
import os
import pickle
import queue
import threading
import time
from collections import defaultdict
from collections.abc import Iterator
from concurrent.futures import Future, ThreadPoolExecutor
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, List, Optional, Tuple

import msgpack
import msgspec
import numpy as np
import torch
import zmq

from vllm import envs
from vllm.attention.selector import backend_name_to_enum, get_attn_backend
from vllm.config import VllmConfig
from vllm.distributed.kv_transfer.kv_connector.v1.base import (
    KVConnectorBase_V1, KVConnectorMetadata, KVConnectorRole)
from vllm.distributed.parallel_state import (
    get_tensor_model_parallel_rank, get_tensor_model_parallel_world_size,
    get_tp_group, get_world_group)
from vllm.forward_context import ForwardContext
from vllm.logger import init_logger
from vllm.platforms import _Backend
from vllm.utils import get_ip, make_zmq_path, make_zmq_socket
from vllm.v1.core.sched.output import SchedulerOutput
from vllm.v1.request import RequestStatus

if TYPE_CHECKING:
    from vllm.attention.backends.abstract import AttentionMetadata
    from vllm.v1.core.kv_cache_manager import KVCacheBlocks
    from vllm.v1.request import Request

from dataclasses import field
from queue import Empty, Queue

Transfer = tuple[int, float]  
EngineId = str
ReqId = str
GET_META_MSG = b"get_meta_msg"
POP_DONE_RECV = b"pop_done_recv"
OVER = b"OVER"
from enum import Enum

import logging

logging.getLogger("aiter").disabled = True
@dataclass
class WriteTask:
    request_id: str
    dst_engine_id: str
    local_block_ids: list[int]
    remote_block_ids_hint: list[int] | None
    layer_name: str
    event: torch.cuda.Event
    remote_notify_port: int
    remote_ip: int
    enqueue_time: float = field(default_factory=time.perf_counter)
    retried: int = 0

@dataclass
class LayerTransferPlan:
    request_id: str
    layer_name: str
    sess_idx: int
    transfer_local_offsets: list[int]
    transfer_remote_offsets: list[int]
    transfer_sizes: list[int]
    use_batch: bool = True
    
@dataclass
class RemoteAllocInfo:
    block_ids: list[int]
    writes_done: int = 0
    decode_dp_rank: int = 0
    transfer_offset: tuple[list[int], list[int], list[int]] | None = None


class ROLE(Enum):
    PRODUCER = "producer"
    CONSUMER = "consumer"
    NOTINIT = "notinit"


_role_lock = threading.Lock()
_GLOBAL_ROLE: ROLE = ROLE.NOTINIT

def set_role(role: ROLE):
    global _GLOBAL_ROLE
    with _role_lock:
        _GLOBAL_ROLE = role

def get_role() -> ROLE:
    return _GLOBAL_ROLE


class MoRIIOMode(Enum):
    READ = "read"
    WRITE = "write"

logger = init_logger(__name__)

def get_moriio_mode() -> MoRIIOMode:
    read_mode = os.environ.get('MORIIO_CONNECTOR_READ_MODE', 'false').lower()
    logger.info(f"MoRIIO Connector Read Mode = {read_mode}")
    if read_mode in ('true', '1', 'yes', 'on'):
        return MoRIIOMode.READ
    else:
        return MoRIIOMode.WRITE


GLOBAL_MORIIO_MODE = get_moriio_mode()

try:
    import mori
    from mori.io import (BackendType, EngineDesc, IOEngine, IOEngineConfig,
                         MemoryDesc, StatusCode)
    logger.info("MoRIIO is available")
    MoRIIO_enabled = True
except ImportError:
    logger.error("MoRIIO is not available")
    MoRIIO_enabled = False


class MoRIIOWrapper:

    def __init__(self, moriio_engine=None,tp_rank=0,dp_rank=0):
        self.tp_rank=tp_rank
        self.dp_rank=dp_rank
        self.moriio_engine = moriio_engine
        self.remote_memory_metadata = None
        self.local_memory_registered = False
        self.local_memory_metadata = None
        self.transfer_status = []
        self.remote_engine_ip = None
        self.notify_port = None
        self.notify_sock = None
        self.lock = threading.Lock()
        self.done_req_ids = []
        self.done_remote_allocate_req = []
        self.done_remote_allocate_req_dict: dict[str, RemoteAllocInfo] = {}
        self.done_write_cache_req_ids = []
        self.notify_thread = None
        self.sock = None
        self.tp_rank = get_tensor_model_parallel_rank()
        self.sessions = []
        self.kv_caches = None
        self.paths = {}

    def set_moriio_engine(self, moriio_engine):
        assert moriio_engine is not None, "You Cannot pass None engine to MoRIIOWrapper!"
        self.moriio_engine = moriio_engine

    def set_backend_type(self, backend_type):
        self.moriio_engine.create_backend(backend_type)

    def get_agent_metadata(self):
        engine_metadata = self.moriio_engine.get_engine_desc()
        engine_metadata_packed = engine_metadata.pack()
        return engine_metadata_packed

    def register_remote_engine(self, remote_packed_engine_metadata):
        consumer_engine_metadata = EngineDesc.unpack(
            remote_packed_engine_metadata)
        self.moriio_engine.register_remote_engine(consumer_engine_metadata)
        return consumer_engine_metadata.key

    def register_local_tensor(self, tensor: torch.Tensor):
        try:
            self.local_memory_metadata = self.moriio_engine.register_torch_tensor(
                tensor)
            local_memory_metadata_packed = self.local_memory_metadata.pack()
        except Exception as e:
            logger.error(f"MoRIIO register local memory failed! reason = {e}")
        self.local_memory_registered = True
        return local_memory_metadata_packed

    def get_unpack_memory_metadata(self, packed_memory_metadata):
        return MemoryDesc.unpack(packed_memory_metadata)

    def build_session(self, local_memory_metadata, remote_memory_metadata):
        return self.moriio_engine.create_session(local_memory_metadata,
                                                 remote_memory_metadata)

    def read_remote_data(self,
                         transfer_size_byte,
                         local_offset=0,
                         remote_offset=0,
                         session=None):
        assert self.local_memory_registered, "You have not register local memory data!"

        transfer_status = session.batch_read(
            local_offset, remote_offset, transfer_size_byte,
            self.moriio_engine.allocate_transfer_uid())

        self.transfer_status.append(transfer_status)

    def read_remote_data_single(self,
                                transfer_size_byte,
                                local_offset=0,
                                remote_offset=0,
                                session=None):
        assert self.local_memory_registered, "You have not register local memory data!"

        transfer_status = session.read(
            local_offset, remote_offset, transfer_size_byte,
            self.moriio_engine.allocate_transfer_uid())

        self.transfer_status.append(transfer_status)

    def write_remote_data(self,
                          transfer_size_byte,
                          local_offset=0,
                          remote_offset=0,
                          session=None):
        assert self.local_memory_registered, "You have not register local memory data!"
        write_uid = self.moriio_engine.allocate_transfer_uid()

        transfer_status = session.batch_write(local_offset, remote_offset,
                                              transfer_size_byte, write_uid)
        with self.lock:
            self.transfer_status.append(transfer_status)

    def write_remote_data_single(self,
                                 transfer_size_byte,
                                 local_offset=0,
                                 remote_offset=0,
                                 sess_idx=0):
        assert self.local_memory_registered, "You have not register local memory data!"

        transfer_status = self.sessions[sess_idx].write(
            local_offset, remote_offset, transfer_size_byte,
            self.moriio_engine.allocate_transfer_uid())
        with self.lock:
            self.transfer_status.append(transfer_status)

    def waiting_for_transfer_complete(self):
        if not self.transfer_status:
            return

        transfers_to_wait = []
        with self.lock:
            transfers_to_wait = self.transfer_status[:]
            self.transfer_status.clear()

        for status in transfers_to_wait:
            try:
                status.Wait()
                if not status.Succeeded():
                    logger.error(
                        f"Transfer failed: {status.Message()}, Code: {status.Code()}"
                    )
                    raise RuntimeError(f"MoRIIO transfer failed!")
            except Exception as e:
                logger.error(f"Transfer {status} failed: {e}")
                raise

    def async_wait_reqid(self, kv_caches=None):
        if kv_caches is not None:
            self.kv_caches = kv_caches

        assert self.notify_port is not None, "Notify port cannot be None"

        if self.notify_thread is not None:
            return

        def _async_wait():
            host = "*"
            path = make_zmq_path("tcp", host, self.notify_port)
            logger.info(f"Node starting to listen notify from path = {path}")

            with zmq_ctx(zmq.ROUTER, path) as sock:
                while True:
                    try:
                        identity, msg = sock.recv_multipart()
                        self._handle_message(msg)
                    except Exception as e:
                        logger.error(f"Error processing message: {e}")
                        continue

        self.notify_thread = threading.Thread(target=_async_wait,
                                              daemon=True,
                                              name="moriio-notify-listener")
        self.notify_thread.start()

    def _handle_message(self, msg: bytes):
        handled = False
        try:
            data = msgpack.loads(msg)
            if isinstance(data, dict) and "req_id" in data:
                self._handle_structured_message(data)
                handled = True

                return
        except (msgpack.exceptions.ExtraData,
                msgpack.exceptions.UnpackException):
            pass

        try:
            msg_str = msg.decode("UTF-8")
            if msg_str.startswith("cmpl"):
                self._handle_completion_message(msg_str)
                handled = True
        except UnicodeDecodeError:
            logger.warning(f"Received non-UTF8 message: {msg}")
        assert handled, f"Unhandled message format: {msg}"

    def _handle_structured_message(self, data: dict):
        req_id = data["req_id"]
        int_list = data.get("int_list", [])
        decode_dp_rank=data.get("decode_rank",0)
        assert len(int_list) > 0, "int_list cannot be empty in remote allocate message"
        msg_type = data.get("type", "unknown")

        with self.lock:
            self.done_remote_allocate_req.append(req_id)
            self.done_remote_allocate_req_dict[req_id] = RemoteAllocInfo(block_ids=int_list,decode_dp_rank=decode_dp_rank)

    def _handle_completion_message(self, msg: str):
        # logger.info(f"MoRIIO received block message: {msg}")
        with self.lock:
            if get_role() == ROLE.PRODUCER:
                # logger.debug(f"P received req id {msg} for release")
                self.done_req_ids.append(msg)
            else:
                self.done_write_cache_req_ids.append(msg)

    def send_notify(self, req_ids, remote_ip=None, remote_port=None):
        if not remote_ip or not remote_port:
            logger.warning("Missing remote_ip or remote_port for notification")
            return

        path = make_zmq_path("tcp", remote_ip, str(remote_port))

        if path not in self.paths:
            ctx = zmq.Context()
            sock = make_zmq_socket(ctx=ctx,
                                   path=path,
                                   socket_type=zmq.DEALER,
                                   bind=False)
            self.paths[path] = sock

        req_list = req_ids if isinstance(req_ids, list) else [req_ids]

        sock = self.paths[path]
        try:
            for req_id in req_list:
                if not isinstance(req_id, str):
                    logger.warning(
                        f"Invalid req_id type: {type(req_id)}, expected str")
                    continue
                sock.send(req_id.encode("utf-8"))
        except Exception as e:
            logger.error(f"Failed to send notification to {path}: {e}")
            self.paths.pop(path, None)
            raise

    def pop_finished_req_ids(self):
        with self.lock:
            done_send = set(self.done_req_ids)
            self.done_req_ids = []
        return done_send

    def pop_finished_write_req_ids(self):
        with self.lock:
            done_write_cache = set(self.done_write_cache_req_ids)
            self.done_write_cache_req_ids = []
        return done_write_cache

    def pop_remote_allocate_req_dict(self):
        with self.lock:
            done_remote_allocate = set(self.done_remote_allocate_req)
            self.done_remote_allocate_req = []
            self.done_remote_allocate_req_dict = {}
        return done_remote_allocate


class MoRIIOAgentMetadata(
        msgspec.Struct,
        omit_defaults=True,  # type: ignore[call-arg]
        # required for @cached_property.d
        dict=True):
    engine_id: str
    agent_metadata: bytes
    kv_caches_base_addr: list[int]
    num_blocks: int
    block_len: int
    attn_backend_name: str


@dataclass
class ReqMeta:
    local_block_ids: list[int]
    remote_block_ids: list[int]
    remote_host: str
    remote_port: int
    remote_handshake_port: int
    remote_notify_port: int
    remote_engine_id: str
    tp_size: int
    remote_dp_size: int


class MoRIIOConnectorMetadata(KVConnectorMetadata):

    def __init__(self):
        self.reqs_to_recv: dict[ReqId, ReqMeta] = {}
        self.reqs_to_save: dict[ReqId, ReqMeta] = {}
        self.reqs_to_send: dict[ReqId, float] = {}

    def __repr__(self):
        return_str = ""
        for req_id, req_meta in self.reqs_to_recv.items():
            return_str += f"{req_id = },{req_meta.local_block_ids = },{req_meta.remote_block_ids = },{req_meta.remote_host = },{req_meta.remote_port = },{req_meta.remote_engine_id = },{req_meta.tp_size = }"
        return_str = f"MoRIIOConnectorMetadata:reqs_to_recv:{return_str},"

        for req_id, req_meta in self.reqs_to_send.items():
            return_str += f"{req_id = },{req_meta = }"
        return_str = f"MoRIIOConnectorMetadata:reqs_to_send:{return_str},"
        return return_str

    def add_new_req(
        self,
        request_id: ReqId,
        local_block_ids: list[int],
        kv_transfer_params: dict[str, Any],
        write_mode=False,
    ):

        _req = ReqMeta(
            local_block_ids=local_block_ids,
            remote_block_ids=kv_transfer_params["remote_block_ids"],
            remote_engine_id=kv_transfer_params["remote_engine_id"],
            remote_host=kv_transfer_params["remote_host"],
            remote_port=kv_transfer_params["remote_port"],
            remote_handshake_port=kv_transfer_params['remote_handshake_port'],
            remote_notify_port=kv_transfer_params.get('remote_notify_port'),
            # P workers don't need to receive tp_size from proxy here.
            tp_size=kv_transfer_params.get("tp_size", 1),
            remote_dp_size=kv_transfer_params.get("remote_dp_size", 8)
        )
        if write_mode:
            self.reqs_to_save[request_id] = _req
        else:
            self.reqs_to_recv[request_id] = _req


class MoRIIOConnector(KVConnectorBase_V1):

    def __init__(self, vllm_config: VllmConfig, role: KVConnectorRole):
        assert vllm_config.kv_transfer_config is not None
        # assert vllm_config.kv_transfer_config.engine_id is not None
        # self.engine_id: EngineId = vllm_config.kv_transfer_config.engine_id
        self.engine_id = str(
            get_ip()) + ":" + str(vllm_config.kv_transfer_config.
                                  kv_connector_extra_config['handshake_port'])
        if role == KVConnectorRole.SCHEDULER:
            self.connector_scheduler: Optional[MoRIIOConnectorScheduler] = \
                MoRIIOConnectorScheduler(vllm_config, self.engine_id)
            self.connector_worker: Optional[MoRIIOConnectorWorker] = None
        elif role == KVConnectorRole.WORKER:
            self.connector_scheduler = None
            self.connector_worker = MoRIIOConnectorWorker(
                vllm_config, self.engine_id)
        logger.info(f"Initialized MoRIIO Connector {self.engine_id = }")

    ############################################################
    # Scheduler Side Methods
    ############################################################

    def get_num_new_matched_tokens(
            self, request: "Request",
            num_computed_tokens: int) -> tuple[int, bool]:
        assert self.connector_scheduler is not None
        return self.connector_scheduler.get_num_new_matched_tokens(
            request, num_computed_tokens)

    def update_state_after_alloc(self, request: "Request",
                                 blocks: "KVCacheBlocks",
                                 num_external_tokens: int):
        assert self.connector_scheduler is not None
        return self.connector_scheduler.update_state_after_alloc(
            request, blocks, num_external_tokens, self.connector_worker)

    def build_connector_meta(
        self,
        scheduler_output: SchedulerOutput,
    ) -> KVConnectorMetadata:
        assert self.connector_scheduler is not None
        return self.connector_scheduler.build_connector_meta(scheduler_output)

    def request_finished(
        self,
        request: "Request",
        block_ids: list[int],
    ) -> tuple[bool, Optional[dict[str, Any]]]:
        assert self.connector_scheduler is not None
        return self.connector_scheduler.request_finished(request, block_ids)

    ############################################################
    # Worker Side Methods
    ############################################################
    def register_kv_caches(self, kv_caches: dict[str, torch.Tensor]):
        assert self.connector_worker is not None
        self.connector_worker.register_kv_caches(kv_caches)

    def get_finished(self,
                     finished_req_ids: set[str]) -> tuple[set[str], set[str]]:
        """Get the finished recving and sending requests."""
        assert self.connector_worker is not None
        return self.connector_worker.get_finished()

    def start_load_kv(self, forward_context: "ForwardContext",
                      **kwargs) -> None:

        if GLOBAL_MORIIO_MODE == MoRIIOMode.WRITE:
            if get_role() == ROLE.CONSUMER:
                self.connector_worker.moriio_wrapper.async_wait_reqid()
        assert self.connector_worker is not None
        assert isinstance(self._connector_metadata, MoRIIOConnectorMetadata)
        self.connector_worker.start_load_kv(self._connector_metadata)

    def wait_for_layer_load(self, layer_name: str) -> None:
        """NixlConnector does not do layerwise saving."""
        pass

    def save_kv_layer(self, layer_name: str, kv_layer: torch.Tensor,
                      attn_metadata: "AttentionMetadata", **kwargs) -> None:

        # Only producer/prefill saves KV Cache
        try:
            self.connector_worker.save_kv_layer(self._connector_metadata,
                                                    layer_name, kv_layer,
                                                    attn_metadata, **kwargs)
        except Exception as e:
            logger.info(f"MoRIIO save_kv_layer error: {e}")
        return None

    def wait_for_save(self):
        """NixlConnector does not save explicitly."""

        pass


class MoRIIOConnectorScheduler:
    """Implementation of Scheduler side methods"""

    def __init__(self, vllm_config: VllmConfig, engine_id: str):
        self.vllm_config = vllm_config
        self.block_size = vllm_config.cache_config.block_size
        self.engine_id: EngineId = engine_id
        self.side_channel_host = envs.VLLM_NIXL_SIDE_CHANNEL_HOST
        self.side_channel_port = (
            self.vllm_config.kv_transfer_config.kv_connector_extra_config[
                'handshake_port'],  # envs.VLLM_NIXL_SIDE_CHANNEL_PORT +
            (self.vllm_config.parallel_config.data_parallel_rank+1) *
            self.vllm_config.parallel_config.tensor_parallel_size)
        logger.info(
            f"==========> Initializing MoRIIO Scheduler {engine_id = },{self.side_channel_port = }"
        )

        self.side_notify_port = self.vllm_config.kv_transfer_config.kv_connector_extra_config[
            'notify_port']  # envs.VLLM_NIXL_SIDE_CHANNEL_PORT +
        self.tp_size = self.vllm_config.parallel_config.tensor_parallel_size
        self.dp_rank = self.vllm_config.parallel_config.data_parallel_rank
        self.is_producer = vllm_config.kv_transfer_config.kv_role == "kv_producer"
        # Requests that need to start recv/send.
        # New requests are added by update_state_after_alloc in
        # the scheduler. Used to make metadata passed to Worker.
        self._reqs_need_recv: dict[ReqId, tuple[Request, list[int]]] = {}
        self._reqs_need_save: dict[ReqId, tuple[Request, list[int]]] = {}

        # Reqs to send and their expiration time
        self._reqs_need_send: dict[ReqId, float] = {}
        self.sock = None
        self.is_producer = vllm_config.kv_transfer_config.kv_role == "kv_producer"
        self.paths = {}

    def get_num_new_matched_tokens(
        self,
        request: "Request",
        num_computed_tokens: int,
    ) -> tuple[int, bool]:
        """
        For remote prefill, pull all prompt blocks from remote
        asynchronously relative to engine execution.

        Args:
            request (Request): the request object.
            num_computed_tokens (int): the number of locally
                computed tokens for this request
        Returns:
            * the number of tokens that can be loaded from the
              external KV cache beyond what is already computed.
            * true if the external KV cache tokens will be loaded
              asynchronously (between scheduler steps).
        """
        if self.is_producer:
            return 0, False

        params = request.kv_transfer_params

        if GLOBAL_MORIIO_MODE == MoRIIOMode.WRITE:
            # MoriiO in write mode, no remote prefill

            return len(request.prompt_token_ids) - num_computed_tokens, True

        return len(request.prompt_token_ids) - 1 - num_computed_tokens, False

    def send_notify_block(self,
                          req_id: str,
                          int_list: list[int] = None,
                          host=None,
                          port=None):

        path = make_zmq_path("tcp", host, port)
        if path not in self.paths:
            ctx = zmq.Context()
            sock = make_zmq_socket(ctx=ctx,
                                   path=path,
                                   socket_type=zmq.DEALER,
                                   bind=False)
            self.paths[path] = sock

        data = {
            "req_id": req_id,
            "int_list": int_list or [],
            "decode_rank": self.dp_rank,
            "type": "remote_blocks"
        }
        # logger.info(f"MoRIIO send notify block for prefill, {data= },{host= },{port= }")
        serialized_data = msgpack.dumps(data)
        self.paths[path].send(serialized_data)

    def update_state_after_alloc(
            self,
            request: "Request",
            blocks: "KVCacheBlocks",
            num_external_tokens: int,
            connector_worker: Optional["MoRIIOConnectorWorker"] = None):

        params = request.kv_transfer_params
        # logger.info(f"enter alloc :{request.request_id}")
        if params.get("do_remote_decode"):
            local_block_ids = blocks.get_block_ids()[0]
            self._reqs_need_save[request.request_id] = (request,
                                                        local_block_ids)

        if params is not None and params.get("do_remote_prefill"):
            if GLOBAL_MORIIO_MODE == MoRIIOMode.READ:
                if remote_block_ids := params.get("remote_block_ids"):
                    if all(p in params
                           for p in ("remote_engine_id", "remote_host",
                                     "remote_port")):
                        # If remote_blocks and num_external_tokens = 0, we
                        # a full prefix cache hit on the D worker. We need to call
                        # send_notif in _read_blocks to free the memory on the P.

                        # Get unhashed blocks to pull from remote.
                        local_block_ids = blocks.get_block_ids()[0]
                        assert len(local_block_ids) <= len(remote_block_ids)
                        if len(local_block_ids) == len(remote_block_ids):
                            pass
                        else:
                            local_block_ids = remote_block_ids[
                                -len(local_block_ids):]

                        self._reqs_need_recv[request.request_id] = (
                            request, local_block_ids)
                    else:
                        logger.warning(
                            "Got invalid KVTransferParams: %s. This "
                            "request will not utilize KVTransfer", params)
                else:
                    #TODO  for read mode and push mode
                    pass
            else:
                # Moriio in write mode, do remote prefill(consumer)
                remote_dp_rank=request.kv_transfer_params[
                        'remote_dp_rank'] 
                for tp_index in range(self.tp_size):
                    
                    cur_port = request.kv_transfer_params[
                        'remote_notify_port'] + (remote_dp_rank+1)*(tp_index+1)-1
                    # logger.info(f"{request.kv_transfer_params['remote_notify_port']= },")
                    # # cur_port=self.side_notify_port+tp_index
                    # logger.debug(f"MoRIIO send notify block for prefill,{params.get("remote_host")=} ,{cur_port = }")
                    # logger.info(f"{tp_index= },{self.dp_rank= },{cur_port= }")

                    self.send_notify_block(req_id=request.request_id,
                                           int_list=blocks.get_block_ids()[0],
                                           host=params.get("remote_host"),
                                           port=cur_port)

            # Only trigger 1 KV transfer per request.

            params["do_remote_prefill"] = False

    def build_connector_meta(
        self,
        scheduler_output: SchedulerOutput,
    ) -> KVConnectorMetadata:
        meta = MoRIIOConnectorMetadata()

        if GLOBAL_MORIIO_MODE == MoRIIOMode.WRITE:
            # when async_load_kv finished, will add new reqs to scheduler_output.scheduled_new_reqs
            # should I use thread to add new req in async_wait_reqid?
            for new_req in scheduler_output.scheduled_new_reqs:
                red_id = new_req.req_id
                local_block_ids = list(new_req.block_ids)
                kv_transfer_params = new_req.sampling_params.extra_args[
                    'kv_transfer_params']
                meta.add_new_req(
                    red_id,
                    local_block_ids,
                    kv_transfer_params,
                )
        # scheduler_output.scheduled_new_reqs[0].sampling_params.extra_args['kv_transfer_params']
        # Loop through scheduled reqs and convert to ReqMeta.
        for req_id, (req, block_ids) in self._reqs_need_recv.items():
            assert req.kv_transfer_params is not None
            meta.add_new_req(
                request_id=req_id,
                local_block_ids=block_ids,
                kv_transfer_params=req.kv_transfer_params,
            )

        for req_id, (req, block_ids) in self._reqs_need_save.items():
            assert req.kv_transfer_params is not None
            meta.add_new_req(
                request_id=req_id,
                local_block_ids=block_ids,
                kv_transfer_params=req.kv_transfer_params,
                write_mode=True,
            )
        # Clear the list once workers start the transfers

        meta.reqs_to_send = self._reqs_need_send

        self._reqs_need_recv.clear()
        self._reqs_need_save.clear()
        self._reqs_need_send = {}

        return meta

    def request_finished(
        self,
        request: "Request",
        block_ids: list[int],
    ) -> tuple[bool, Optional[dict[str, Any]]]:
        """
        Once a request is finished, determine whether request blocks
        should be freed now or will be sent asynchronously and freed later.
        """

        params = request.kv_transfer_params
        logger.debug(
            "MoriioConnector request_finished, request_status=%s, "
            "kv_transfer_params=%s", request.status, params)
        if not params:
            return False, None

        if params.get("do_remote_prefill"):
            # If do_remote_prefill is still True when the request is finished,
            # update_state_after_alloc must not have been called (the request
            # must have been aborted before it was scheduled).
            # To avoid stranding the prefill blocks in the prefill instance,
            # we must add empty block_ids to _reqs_need_recv so that our
            # worker side will notify and free blocks in the prefill instance.
            self._reqs_need_recv[request.request_id] = (request, [])
            params["do_remote_prefill"] = False
            return False, None

        if (not params.get("do_remote_decode")
                or request.status != RequestStatus.FINISHED_LENGTH_CAPPED):
            return False, None

        # Get computed blocks.
        all_full = request.num_computed_tokens % self.block_size == 0
        # computed_block_ids = block_ids if all_full else block_ids[:-1]
        computed_block_ids = block_ids
        # If prompt < block_size, no xfer so free blocks immediately.
        delay_free_blocks = len(computed_block_ids) > 0

        if delay_free_blocks:
            # Prefill request on remote. It will be read from D upon completion
            self._reqs_need_send[request.request_id] = time.perf_counter(
            ) + envs.VLLM_NIXL_ABORT_REQUEST_TIMEOUT

        return delay_free_blocks, dict(
            do_remote_prefill=True,
            do_remote_decode=False,
            remote_block_ids=computed_block_ids,
            remote_engine_id=self.engine_id,
            remote_host=self.side_channel_host,
            remote_port=self.side_channel_port,
            tp_size=self.vllm_config.parallel_config.tensor_parallel_size)


class MoRIIOConnectorWorker:
    """Implementation of Worker side methods"""

    def __init__(self, vllm_config: VllmConfig, engine_id: str):
        if not MoRIIO_enabled:
            logger.error("MoRIIO is not available")
            raise RuntimeError("MoRIIO is not available")
        logger.info("Initializing MoRIIO wrapper")
        logger.info("Initializing MoRIIO worker %s", engine_id)

        # Config.
        self.vllm_config = vllm_config
        self.kv_transfer_config = vllm_config.kv_transfer_config
        self.is_producer = self.kv_transfer_config.is_kv_producer

        if self.is_producer:
            set_role(ROLE.PRODUCER)
        else:
            set_role(ROLE.CONSUMER)
        # mori engine
        self._rank = get_world_group().rank
        self._local_rank = get_world_group().local_rank
        self.tp_rank = get_tensor_model_parallel_rank()
        self.dp_rank= vllm_config.parallel_config.data_parallel_rank
        logger.info(f"MoRIIO Worker init {self.tp_rank = },{self.dp_rank= }"
                    f",{self.is_producer= }")
        self.local_ip = get_ip()
        self.local_kv_port = int(self.kv_transfer_config.kv_port)
        
        self.local_kv_port = self.local_kv_port + (self.tp_rank+1)*(self.dp_rank+1)
        
        
        self.proxy_ip = self.kv_transfer_config.kv_connector_extra_config[
            "proxy_ip"]
        self.proxy_port = int(
            self.kv_transfer_config.kv_connector_extra_config["proxy_port"])

        self.local_ping_port = int(
            self.kv_transfer_config.
            kv_connector_extra_config["local_ping_port"])

        self.local_ping_port = self.local_ping_port + (self.tp_rank+1)*(self.dp_rank+1)

        self.proxy_ping_port = int(
            self.kv_transfer_config.
            kv_connector_extra_config["proxy_ping_port"])

        self.http_port = int(
            self.kv_transfer_config.kv_connector_extra_config['http_port'])
        self.handshake_port = int(self.kv_transfer_config.
                                  kv_connector_extra_config['handshake_port'])
        self.notify_port = int(
            self.kv_transfer_config.kv_connector_extra_config['notify_port'])
        
        self.notify_port = self.notify_port + (self.tp_rank+1)*(self.dp_rank+1) -1
        # self.local_metadata_port = int(self.kv_transfer_config.kv_connector_extra_config['metadata_port'])
        '''
        ping: local_ip:local_ping_port -> proxy_ip:proxy_ping_port
        prompt request: user_ip:user_port -> proxy_ip:proxy_listening_port -> local_ip:http_port
        kvcache: local_ip:local_kv_port <-> ...
        metadata: local_ip:local_metadata_port <->
        '''
        self.zmq_context = zmq.Context()
        self.metadata_address = f"{self.local_ip}:{self.local_ping_port}"
        self.request_address = f"{self.local_ip}:{self.http_port}"
        self.ping_address = f"{self.local_ip}:{self.local_ping_port}"

        self.moriio_engine = None
        self._handle_request_thread = None
        self._ping_thread = None
        engine_suffix = str(self.local_ip) + ":" + str(
            self.handshake_port) + ":tp " + str(self.tp_rank)+":dp " + str(self.dp_rank)
        if not self.is_producer:
            self.poller = zmq.Poller()
            self.metadata_socket = self.zmq_context.socket(zmq.ROUTER)
            self.metadata_socket.bind(f"tcp://{self.metadata_address}")
            self.poller.register(self.metadata_socket, zmq.POLLIN)

            logger.info(f"build IOEngine {self.local_ip},{self.local_kv_port}")
            self.moriio_engine = IOEngine(
                "consumer:" + engine_suffix,
                IOEngineConfig(self.local_ip, self.local_kv_port))
            self._handle_request_thread = threading.Thread(
                target=self.handle_proxy_request, daemon=True)
            self._handle_request_thread.start()
        else:
            logger.info(f"build IOEngine {self.local_ip},{self.local_kv_port}")

            self.moriio_engine = IOEngine(
                "producer:" + engine_suffix,
                IOEngineConfig(self.local_ip, self.local_kv_port))
        if self._rank == 0 and self.proxy_ip != "":
            self._ping_thread = threading.Thread(target=self._ping,
                                                 args=(self.zmq_context, ),
                                                 daemon=True)
            self._ping_thread.start()

        logger.info(
            f"Initializing MoRIIO Engine ,engine = {self.moriio_engine},role = {'producer' if self.is_producer else 'consumer'}"
        )
        logger.debug(
            f"{self.local_ip = },{self._rank = },{self._local_rank = },{self.local_kv_port = },{self.proxy_ip = },{self.proxy_port = },{self.local_ping_port = },{self.proxy_ping_port = }"
        )
        # Agent.
        self.moriio_wrapper = MoRIIOWrapper(tp_rank=self.tp_rank,dp_rank=self.dp_rank)
        self.moriio_wrapper.set_moriio_engine(self.moriio_engine)

        self.moriio_wrapper.set_backend_type(BackendType.RDMA)

        self.moriio_wrapper.notify_port = self.notify_port
        self.local_kv_cache_metadata = []
        self.local_kv_cache_size = []
        self.layer_name_to_local_kv_cache_metadata: dict[str,
                                                         List[Any]] = dict()

        self.remote_kv_cache_metadata = []
        self.remote_kv_cache_size = []
        self.layer_name_to_remote_kv_cache_metadata: dict[str, dict[
            str, List[Any]]] = dict()
        self.slot_size_bytes = 0

        self.load_ready_flag = False
        self.write_ready_flags = {}
        self.kv_cache_shape = None
        self.block_shape = None
        self.kv_element_size = 0

        self.done_sending_reqs = []
        self.done_send_threads = []

        # Map of engine_id -> {rank0: agent_name0, rank1: agent_name1..}.
        self._remote_agents: dict[EngineId, dict[int, str]] = defaultdict(dict)

        # MoRIIO handshake port.
        # NOTE(rob): Within a DP group, each DP rank gets its own
        # base port (which is sent in the KVTransferParams).
        # Each TP rank listens/queries on the base_port + tp_rank.
        # self.side_channel_port: int = (
        #     envs.VLLM_NIXL_SIDE_CHANNEL_PORT +
        #     vllm_config.parallel_config.data_parallel_rank *
        #     vllm_config.parallel_config.tensor_parallel_size)
        self.side_channel_port: int = (
            self.handshake_port +
                (self.dp_rank + 1) * (self.tp_rank + 1)  # 正确的写法
        )
        #why wuxiao 
        logger.info(f"MoRIIO Worker init {self.tp_rank = },{self.dp_rank= }")
        logger.info(f"MoRIIO side channel_port port: {self.side_channel_port}, han")
        # Metadata.
        self.engine_id: EngineId = engine_id

        self.world_size = get_tensor_model_parallel_world_size()
        self.tp_group = get_tp_group()

        # KV Caches and moriio tracking data.
        self.kv_caches: dict[str, torch.Tensor] = {}

        # Map of engine_id -> kv_caches_base_addr. For TP case, each local
        # rank will still only pull from a single remote TP worker.
        self.kv_caches_base_addr: dict[EngineId, list[int]] = {}

        # Number of MoRIIO regions. Currently one region per cache
        # (so 1 per layer for MLA, otherwise 2 per layer)
        self.num_regions = 0
        self.num_layers = 0

        # moriio_prepped_dlist_handle.
        self.src_xfer_side_handle: int = 0
        # Map of engine_id -> moriio_prepped_dlist_handle (int)].
        self.dst_xfer_side_handles: dict[EngineId, int] = {}

        # Map of engine_id -> num_blocks. All ranks in the same deployment will
        # have the same number of blocks.
        self.dst_num_blocks: dict[EngineId, int] = {}
        self._registered_descs: list[Any] = []
        # In progress transfers.
        # [req_id -> list[handle]]
        self._recving_transfers = defaultdict[ReqId, list[Transfer]](list)
        # Track the expiration time of requests that are waiting to be sent.
        self._reqs_to_send: dict[ReqId, float] = {}

        # Background thread for handling new handshake requests.
        self._moriio_handshake_listener_t: Optional[threading.Thread] = None
        # Background thread for initializing new MoRIIO handshakes.
        self._handshake_initiation_executor = ThreadPoolExecutor(
            # MoRIIO is not guaranteed to be thread-safe, limit 1 worker.
            max_workers=1,
            thread_name_prefix="vllm-moriio-handshake-initiator")
        self._ready_requests = queue.Queue[tuple[ReqId, ReqMeta]]()
        self._handshake_futures: dict[EngineId, Future[dict[int, str]]] = {}
        # Protects _handshake_futures and _remote_agents.
        self._handshake_lock = threading.RLock()

        self.block_size = vllm_config.cache_config.block_size
        self.model_config = vllm_config.model_config
        self.cache_config = vllm_config.cache_config

        # TODO(mgoin): remove this once we have hybrid memory allocator
        # Optimization for models with local attention (Llama 4)
        # List of block window sizes for each layer for local attention
        self.block_window_per_layer: list[Optional[int]] = []
        self.use_mla = self.model_config.use_mla
        self.built_session = False
        self.builded_write_session: defaultdict[str, list] = defaultdict(list)
        self._write_session_lock = threading.Lock()
        self.debug_cache = []
        backend = get_attn_backend(self.model_config.get_head_size(),
                                   self.model_config.dtype,
                                   self.cache_config.cache_dtype,
                                   self.block_size,
                                   use_mla=self.use_mla)
        self.backend_name = backend.get_name()
        attn_backend = backend_name_to_enum(self.backend_name)
        self._use_flashinfer = attn_backend == _Backend.FLASHINFER
        logger.debug("Detected attention backend %s", self.backend_name)

        self._tp_size: dict[EngineId, int] = {self.engine_id: self.world_size}
        # With heterogeneous TP, P must wait for all assigned D TP workers to
        # finish reading before safely freeing the blocks.
        self.consumer_notification_counts_by_req = defaultdict[ReqId, int](int)

        ####write worker###
        self._write_task_q: Queue[WriteTask] = Queue()
        self._write_worker_started = False
        self._write_worker_lock = threading.Lock()
        self._deferred_tasks: list[WriteTask] = []
        ###

    def _ensure_write_worker(self):
        if self._write_worker_started:
            return
        with self._write_worker_lock:
            if self._write_worker_started:
                return
            t = threading.Thread(target=self._write_worker_loop,
                                 daemon=True,
                                 name="moriio-write-worker")
            t.start()
            self._write_worker_started = True

    def schedule_write_blocks(self, request_id: str, dst_engine_id: str,
                              local_block_ids: list[int],
                              remote_block_ids: list[int] | None,
                              layer_name: str, kv_layer: torch.Tensor,
                              remote_notify_port: int, remote_ip: str):
        self._ensure_write_worker()

        stream = torch.cuda.current_stream()
        event = torch.cuda.Event()
        event.record(stream)

        task = WriteTask(request_id=request_id,
                         dst_engine_id=dst_engine_id,
                         local_block_ids=local_block_ids,
                         remote_block_ids_hint=remote_block_ids,
                         layer_name=layer_name,
                         event=event,
                         remote_notify_port=remote_notify_port,
                         remote_ip=remote_ip)
        self._write_task_q.put(task)

    def _remote_blocks_ready(self, task: WriteTask) -> bool:
        return task.request_id in self.moriio_wrapper.done_remote_allocate_req_dict

    def _write_worker_loop(self):
        SLEEP_MIN = 0.001
        REQUEUE_DELAY = 0.01
        while True:
            still_defer: list[WriteTask] = []
            if self.dp_rank==0:
                c=0
            if self.dp_rank==1:
                b=0
            if self._deferred_tasks:
                for task in self._deferred_tasks:
                    if self._remote_blocks_ready(task):
                        self._execute_write_task(task)
                    else:
                        still_defer.append(task)
                self._deferred_tasks = still_defer

            try:
                task = self._write_task_q.get(timeout=0.01)
            except Empty:
                continue

            if not self._remote_blocks_ready(task):
                task.retried += 1
                self._deferred_tasks.append(task)
                continue

            self._execute_write_task(task)

    def _get_built_session(self, remote_engine_id):
        if remote_engine_id not in self.builded_write_session:
            cur_remote_engine_sessions = []
            for ln, local_meta in self.layer_name_to_local_kv_cache_metadata.items(
            ):

                unpcaked_local_memory_meta = self.moriio_wrapper.get_unpack_memory_metadata(
                    local_meta[0])
                unpcaked_remote_memory_meta = self.moriio_wrapper.get_unpack_memory_metadata(
                    self.layer_name_to_remote_kv_cache_metadata[
                        remote_engine_id][ln][0])
                cur_remote_engine_sessions.append(
                    self.moriio_wrapper.build_session(
                        unpcaked_local_memory_meta,
                        unpcaked_remote_memory_meta))
            self.builded_write_session[
                remote_engine_id] = cur_remote_engine_sessions
        return self.builded_write_session[remote_engine_id]


    def _get_remote_alloc_info(self, request_id: str) -> RemoteAllocInfo:
        try:
            return self.moriio_wrapper.done_remote_allocate_req_dict[request_id]
        except KeyError:
            raise RuntimeError(f"RemoteAllocInfo missing for request {request_id}")

  

    def _prepare_layer_transfer(self, task: WriteTask,
                                request_info: RemoteAllocInfo) -> LayerTransferPlan:
        request_id = task.request_id
        layer_name = task.layer_name
        local_block_ids = task.local_block_ids
        remote_block_ids = request_info.block_ids

        is_mla = (len(self.kv_cache_shape) == 3)
        sess_idx = list(self.layer_name_to_local_kv_cache_metadata.keys()).index(layer_name)
        sz = self.kv_caches[layer_name].element_size()
        stride = self.kv_caches[layer_name].stride()
        if is_mla:
            blknum, blksize, hs = self.kv_cache_shape
            hn = 1
            block_stride = stride[0]
            ktov_stride = None
        else:
            _, blknum, blksize, hn, hs = self.kv_cache_shape
            ktov_stride = stride[0]
            block_stride = stride[1]
        transfer_size_byte = blksize * hn * hs * sz
        
        if request_info.transfer_offset is None:
            per_block = 1 if is_mla else 2
            total = len(local_block_ids) * per_block
            offset_local = [0] * total
            offset_remote = [0] * total
            transfer_sizes = [transfer_size_byte] * total
            w = 0
            for i, lb in enumerate(local_block_ids):
                rb = remote_block_ids[i]
                # K
                offset_local[w] = sz * (lb * block_stride)
                offset_remote[w] = sz * (rb * block_stride)
                w += 1
                if not is_mla:
                    # V
                    offset_local[w] = sz * (1 * ktov_stride +
                                            lb * block_stride)
                    offset_remote[w] = sz * (1 * ktov_stride +
                                                rb * block_stride)
                    w += 1
         
            merged_l, merged_r, merged_s = self.merge_contiguous_blocks_fast_v2(
                offset_local, offset_remote, transfer_sizes, assume_sorted=True)
            request_info.transfer_offset = (merged_l, merged_r, merged_s)

        a, b, c = request_info.transfer_offset
        return LayerTransferPlan(
            request_id=request_id,
            layer_name=layer_name,
            sess_idx=sess_idx,
            transfer_local_offsets=a,
            transfer_remote_offsets=b,
            transfer_sizes=c,
            use_batch=True
        )

    def _do_layer_write(self, plan: LayerTransferPlan, sessions):
        if plan.use_batch:
            self.moriio_wrapper.write_remote_data(
                plan.transfer_sizes,
                plan.transfer_local_offsets,
                plan.transfer_remote_offsets,
                sessions[plan.sess_idx])
        else:
            for i in range(len(plan.transfer_local_offsets)):
                self.moriio_wrapper.write_remote_data_single(
                    plan.transfer_sizes[i],
                    plan.transfer_local_offsets[i],
                    plan.transfer_remote_offsets[i],
                    plan.sess_idx)

    def _finalize_write_if_finished(self, request_id: str, request_info: RemoteAllocInfo, task: WriteTask):
        request_info.writes_done += 1
        if request_info.writes_done == self.num_layers:
            #TODO:  wait current req_id transfer complete
            self.moriio_wrapper.waiting_for_transfer_complete()
            the_remote_port=task.remote_notify_port  + (self.tp_rank+1)*(request_info.decode_dp_rank+1)-1
            # logger.info(f"send notify for write req {request_id=} {the_remote_port=}")
            self.moriio_wrapper.send_notify(
                request_id,
                task.remote_ip,
                the_remote_port,
            )

    def _execute_write_task(self, task: WriteTask):
        if GLOBAL_MORIIO_MODE == MoRIIOMode.READ:
            return
        request_info = self._get_remote_alloc_info(task.request_id)
        if request_info.block_ids is None:
            # logger.debug("Request %s remote block ids not ready", task.request_id)
            return
        task.event.synchronize()
        
        task.dst_engine_id=task.dst_engine_id+"_dp"+str(request_info.decode_dp_rank)
        sessions = self._get_built_session(task.dst_engine_id)
        plan = self._prepare_layer_transfer(task, request_info)
        self._do_layer_write(plan, sessions)
        self._finalize_write_if_finished(task.request_id, request_info,task)
    

    def _ping(self, zmq_context):
        PING_INTERVAL = 5
        MAX_RETRIES =100000
        
        http_request_address = f"http://{self.request_address}/v1/completions"
        role = "P" if self.is_producer else "D"
        
        retry_count = 0
        index = 1
        
        with zmq_context.socket(zmq.DEALER) as sock:
            sock.connect(f"tcp://{self.proxy_ip}:{self.proxy_ping_port}")
            
            while True:
                try:
                    data = {
                        "type": "register",
                        "role": role,
                        "index": str(index),
                        "request_address": http_request_address,
                        "handshake_port": self.handshake_port,
                        "notify_port": self.notify_port
                    }

                    sock.send(msgpack.dumps(data))
                    # logger.debug(f"Successfully sent ping message #{index}")
                    retry_count = 0 
                    
                except ConnectionRefusedError:
                    logger.info(
                        f"Connection refused: {self.local_ip}:{self.local_ping_port} -> "
                        f"{self.proxy_ip}:{self.proxy_ping_port}"
                    )
                    retry_count += 1
                    
                except OSError as e:
                    logger.info(f"OS error when sending ping: {e}")
                    retry_count += 1
                    
                except Exception as e:
                    logger.info(f"Unexpected error when sending ping: {e}")
                    retry_count += 1
                    
                finally:
                    if retry_count >= MAX_RETRIES:
                        logger.error(f"Max retries ({MAX_RETRIES}) exceeded. Stopping ping loop.")
                        break
                        
                    time.sleep(PING_INTERVAL)
                    index += 1

    def handle_proxy_request(self):
        if self.is_producer:
            raise NotImplementedError(
                "prefill instance doesn't need to send kv cache in pull mode")
        while True:
            socks = dict(self.poller.poll())
            logger.info(f"====> handle_proxy_request: {socks = }")
            if self.metadata_socket not in socks:
                continue
            else:
                pass

    def __del__(self):
        """Cleanup background threads on destruction."""
        self._handshake_initiation_executor.shutdown(wait=False)
        if self._moriio_handshake_listener_t:
            self._moriio_handshake_listener_t.join(timeout=0)

    @staticmethod
    def _moriio_handshake_listener(
            metadata: MoRIIOAgentMetadata, ready_event: threading.Event,
            base_port: int, tp_rank: int,dp_rank:int,
            layer_name_to_local_kv_cache_metadata: dict):
        """Background thread for getting new MoRIIO handshakes."""

        encoder = msgspec.msgpack.Encoder()
        encoded_data = encoder.encode(metadata)
        size_in_bytes = len(encoded_data)
        logger.debug("Size of encoded MoRIIOAgentMetadata: %s bytes",
                     str(size_in_bytes))

        # Listen for new requests for metadata.
        host = "*"
        logger.info(f"======> mori handeshake starting listening on baseport: {base_port}")

        path = make_zmq_path("tcp", host, base_port )
        logger.info(f"======> mori handeshake sstarting listening on path: {path}")

        with zmq_ctx(zmq.ROUTER, path) as sock:
            ready_event.set()
            while True:
                identity, msg = sock.recv_multipart()
                if msg != GET_META_MSG and msg != POP_DONE_RECV:
                    logger.warning(
                        "Connection listener got unexpected message %s", msg)
                    assert False, "handhsake failed!"
                elif msg == GET_META_MSG:
                    sock.send_multipart(
                        (identity, b"",
                         encoded_data))  # send local mori io engine meta data
                    logger.info("MoRIIO handshake listener sent metadata to %s")
                    # now we send tensor meta data for each block
                    buf = pickle.dumps(layer_name_to_local_kv_cache_metadata)
                    sock.send_multipart((identity, b"", buf))
                elif msg == POP_DONE_RECV:
                    _, req_id = sock.recv_multipart()
                    logger.info("MoRIIO handshake listener received done recv for req %s",
                                req_id.decode())
                else:
                    pass

    def _moriio_handshake(
        self,
        host: str,
        port: int,
        remote_tp_size: int,
        expected_engine_id: str,
        remote_dp_rank:int,
    ) -> dict[int, str]:
        """Do a MoRIIO handshake with a remote instance."""

        start_time = time.perf_counter()

        # NOTE(rob): we need each rank to have a unique port. This is
        # a hack to keep us moving. We will switch when moving to etcd
        # or where we have a single ZMQ socket in the scheduler.

        # Handshake only with the remote TP rank that current local rank will
        # pull from. With homogeneous TP it happens to be the same rank_i.

        tp_ratio = self._tp_size[
            self.
            engine_id] // remote_tp_size
        tp_ratio = 1
        # p_remote_rank = self.tp_rank // tp_ratio
        p_remote_rank = (self.tp_rank+1)*(remote_dp_rank+1) 
        path = make_zmq_path("tcp", host, port + p_remote_rank)
        logger.info("handeshake Querying metadata on path: %s at remote rank %s", path,
                    p_remote_rank)

        # Send query for the request.
        with zmq_ctx(zmq.DEALER, path) as sock:
            logger.info(f"prepare send msg  INSTAZNCE: {path}")
            sock.send(GET_META_MSG)
            received_frame = sock.recv_multipart()
            if len(received_frame) != 2 or received_frame[0] != b"":
                assert 0, f"unexpected frame! {received_frame = }"

            metadata_bytes = received_frame[1]
            decoder = msgspec.msgpack.Decoder(MoRIIOAgentMetadata)
            metadata = decoder.decode(metadata_bytes)
            got_metadata_time = time.perf_counter()
            logger.info("MoRIIO handshake: get metadata took: %s",
                         got_metadata_time - start_time)

            # Ensure engine id matches.
            # pass for write
            # if metadata.engine_id != expected_engine_id:
            #     raise RuntimeError(f"Remote MoRIIO agent engine ID mismatch. "
            #                        f"Expected {expected_engine_id},"
            #                        f"received {metadata.engine_id}.")

            # Register Remote agent.
            # remote_agent_name = self.add_remote_agent(metadata, p_remote_rank,remote_tp_size)
            # self.moriio_wrapper.remote_handshake_port = port + p_remote_rank
            self.moriio_wrapper.remote_engine_ip = host
            remote_agent_name = self.moriio_wrapper.register_remote_engine(
                metadata.agent_metadata)
            remote_agent_name = self.add_remote_agent(metadata, p_remote_rank,
                                                      remote_tp_size)
            logger.info(f"MoRIIO handshake: registered remote agent "
                        f"{remote_agent_name=} for engine ID "
                        f"{expected_engine_id=},f{path= }")
            if len(self.local_kv_cache_metadata) > 0:
                logger.warning(
                    f"{len(self.local_kv_cache_metadata) = },maybe you didnt clear this buffer correctly"
                )
                self.local_kv_cache_metadata = []
            if len(self.remote_kv_cache_metadata) > 0:
                logger.warning(
                    f" {len(self.remote_kv_cache_metadata) = },maybe you didnt clear this buffer correctly"
                )
                self.remote_kv_cache_metadata = []

            received_frame = sock.recv_multipart()
            if len(received_frame) != 2 or received_frame[0] != b"":
                assert 0, f"Unexpected frame! {received_frame = }"
            buf = received_frame[1]
            self.layer_name_to_remote_kv_cache_metadata[
                expected_engine_id] = pickle.loads(buf)

            setup_agent_time = time.perf_counter()
            logger.debug("MoRIIO handshake: add agent took: %s",
                         setup_agent_time - got_metadata_time)

        # Remote rank -> agent name.
        return {p_remote_rank: remote_agent_name}

    def _background_moriio_handshake(self, req_id: str,
                                     remote_engine_id: EngineId,
                                     meta: ReqMeta):
        # Do MoRIIO handshake in background and add to _ready_requests when done.
        fut = None
        if remote_engine_id is not None:
            fut = self._handshake_futures.get(remote_engine_id)
        if fut is None:
            host = meta.remote_host
            port = int(meta.remote_handshake_port)
            tp_size = int(meta.tp_size)
            remote_dp_size = int(meta.remote_dp_size)
        # TODO: handle failure state of future in the
        # callback, we want to fail the request in this case.
        def request_ready(_f: Future[Any], entry=(req_id, meta)):
            logger.info("MoRIIO handshake done for request %s", req_id)
            self._ready_requests.put(entry)
            self.load_ready_flag = True
            self.write_ready_flags[remote_engine_id] = True
            
        if remote_dp_size > 1:
            # 修复1: 正确初始化future列表
            fut_list = []
            
            for cur_dp_rank in range(remote_dp_size):
                # 修复2: 为每个DP rank创建独立的engine_id
                dp_engine_id = f"{remote_engine_id}_dp{cur_dp_rank}"
                
                # 提交握手任务
                future = self._handshake_initiation_executor.submit(
                    self._moriio_handshake, host, port, tp_size, dp_engine_id, cur_dp_rank
                )
                fut_list.append(future)
                
                # 为每个future单独设置回调
                def done_callback(f: Future[dict[int, str]], eid=dp_engine_id):
                    with self._handshake_lock:
                        # 修复3: 从handshake_futures中删除对应的engine_id
                        self._handshake_futures.pop(eid, None)
                        try:
                            self._remote_agents[eid] = f.result()
                        except Exception:
                            logger.exception("Handshake with %s failed", eid)
                
                future.add_done_callback(done_callback)
                self._handshake_futures[dp_engine_id] = future
            
                # 修复4: 使用Future列表而不是单个future
            # fut = fut_list
            def wait_all_dp():
                for future in fut_list:
                    future.result()  # 等待所有future完成
                return True

            all_done_future = self._handshake_initiation_executor.submit(wait_all_dp)
            all_done_future.add_done_callback(request_ready)
            fut = all_done_future
        else:
            remote_engine_id = f"{remote_engine_id}_dp0"

            fut = self._handshake_initiation_executor.submit(
                self._moriio_handshake, host, port, tp_size, remote_engine_id)

            def done_callback(f: Future[dict[int, str]], eid=remote_engine_id):
                with self._handshake_lock:
                    del self._handshake_futures[eid]
                    try:
                        self._remote_agents[eid] = f.result()
                    except Exception:
                        logger.exception("Handshake with %s failed", eid)

            # if not self.is_producer:
            fut.add_done_callback(done_callback)
            self._handshake_futures[remote_engine_id] = fut



            fut.add_done_callback(request_ready)

    def register_kv_caches(self, kv_caches: dict[str, torch.Tensor]):
        """Register the KV Cache data in moriio."""
        for _, t in kv_caches.items():
            t = t.zero_()  # for debug,not necessary
        # kv_caches,KEY layer name,VALUE cache tensor,(2,numblocks,blocksize,headnum,headsize)
        _, first_kv_cache = next(iter(kv_caches.items()))
        kv_elem_size = first_kv_cache.element_size()

        # TODO(tms): Find a more robust way to detect and handle MLA
        # NOTE (NickLucche) To move blocks efficiently with MoRIIO, the expected
        # KV memory layout is HND, as opposed to the default NHD. Note that it
        # will only affects the strides. For MLA instead, we make require no
        # such thing and resort to the standard layout.
        use_mla = len(first_kv_cache.shape) == 3
        assert use_mla == self.use_mla

        if use_mla:
            # MLA case.
            self.num_blocks = first_kv_cache.shape[0]
            block_rank = 2  # [block_size, latent_dim]
            block_shape = first_kv_cache.shape[-block_rank:]
            block_size, kv_latent_dim = block_shape
            self.slot_size_bytes = kv_elem_size * kv_latent_dim
        else:
            # [2 (k and v), num_blocks, ...]
            if self._use_flashinfer:
                # FlashInfer swaps 2<->num_blocks dimensions.
                self.num_blocks = first_kv_cache.shape[0]
                block_rank = 4  # [2, block_size, kv_heads, head_dim]
            else:
                self.num_blocks = first_kv_cache.shape[1]
                block_rank = 3  # [block_size, kv_heads, head_dim]
            block_shape = first_kv_cache.shape[-block_rank:]
            block_size, n_kv_heads, head_dim = block_shape[-3:]
            # head size in bytes.
            self.slot_size_bytes = kv_elem_size * n_kv_heads * head_dim  # 1 token 1 layer size , slot size
        assert block_size == self.block_size
        # TODO(tms): self.block_len needs to be per-layer for sliding window,
        # hybrid attn, etc
        # block size in bytes
        self.block_len = kv_elem_size * math.prod(block_shape)
        self.kv_cache_shape = first_kv_cache.shape
        self.block_shape = block_shape
        self.kv_element_size = kv_elem_size

        # logger.info(f"Registering KV_Caches: {use_mla=}, {self.num_blocks=}, {block_shape=}, per_layer_kv_cache_shape={first_kv_cache.shape}")

        self.dst_num_blocks[self.engine_id] = self.num_blocks
        self.kv_caches = kv_caches  # layer name to kv cache
        kv_caches_base_addr = []
        caches_data = []

        # Note(tms): I modified this from the original region setup code.
        # K and V are now in different regions. Advantage is that we can
        # elegantly support MLA and any cases where the K and V tensors
        # are non-contiguous (it's not locally guaranteed that they will be)
        # Disadvantage is that the encoded MoRIIOAgentMetadata is now larger
        # (roughly 8KB vs 5KB).
        # Conversely for FlashInfer, K and V are transferred in the same tensor
        # to better exploit the memory layout (ie num_blocks is the first dim).
        kv_cache_key_list = kv_caches.keys()
        kv_cache_shape_list = [c.shape for c in kv_caches.values()]
        for cache_or_caches in kv_caches.values():

            cache_list = [
                cache_or_caches
            ] if use_mla or self._use_flashinfer else cache_or_caches
            # logger.debug(f"prepare register local kv cache tensor for local mori io engine,{len(cache_list) = },{kv_caches.keys() = }")
            for cache in cache_list:

                base_addr = cache.data_ptr()
                region_len = self.num_blocks * self.block_len
                caches_data.append(
                    (base_addr, region_len, cache.device.index, ""))
                kv_caches_base_addr.append(base_addr)

        for layer_name, kv_cache in kv_caches.items():

            if layer_name not in self.layer_name_to_local_kv_cache_metadata:
                self.layer_name_to_local_kv_cache_metadata[layer_name] = []

            # for cache in cache_list:
            # moriio_mem_metadata = self.moriio_wrapper.register_local_tensor(cache)
            moriio_mem_metadata = self.moriio_wrapper.register_local_tensor(
                kv_cache)
            self.layer_name_to_local_kv_cache_metadata[layer_name].append(
                moriio_mem_metadata)

            self.local_kv_cache_size.append(cache.nelement() *
                                            cache.element_size())

        self.kv_caches_base_addr[self.engine_id] = kv_caches_base_addr
        self.num_regions = len(caches_data)
        self.num_layers = len(self.kv_caches.keys())

        # Optimization for models with local attention (Llama 4)
        if self.vllm_config.model_config.hf_config.model_type == "llama4":
            from transformers import Llama4TextConfig
            assert isinstance(self.vllm_config.model_config.hf_text_config,
                              Llama4TextConfig)
            llama4_config = self.vllm_config.model_config.hf_text_config
            no_rope_layers = llama4_config.no_rope_layers
            chunk_size = llama4_config.attention_chunk_size
            chunk_block_size = math.ceil(chunk_size / self.block_size)
            for layer_idx in range(self.num_layers):
                # no_rope_layers[layer_idx] == 0 means NoPE (global)
                # Any other value means RoPE (local chunked)
                is_local_attention = no_rope_layers[layer_idx] != 0
                block_window = chunk_block_size if is_local_attention else None
                self.block_window_per_layer.append(block_window)
            logger.debug("Llama 4 block window per layer mapping: %s",
                         self.block_window_per_layer)
            assert len(self.block_window_per_layer) == self.num_layers

        metadata = MoRIIOAgentMetadata(
            engine_id=self.engine_id,
            agent_metadata=self.moriio_wrapper.get_agent_metadata(),
            kv_caches_base_addr=self.kv_caches_base_addr[self.engine_id],
            num_blocks=self.num_blocks,
            block_len=self.block_len,
            attn_backend_name=self.backend_name)
        ready_event = threading.Event()
        self._moriio_handshake_listener_t = threading.Thread(
            target=self._moriio_handshake_listener,
            args=(metadata, ready_event, self.side_channel_port, self.tp_rank,self.dp_rank,
                  self.layer_name_to_local_kv_cache_metadata),
            daemon=True,
            name="moriio_handshake_listener")
        self._moriio_handshake_listener_t.start()
        ready_event.wait()  # Wait for listener ZMQ socket to be ready.
        self.moriio_wrapper.async_wait_reqid(self.kv_caches)

    def add_remote_agent(self,
                         moriio_agent_meta: MoRIIOAgentMetadata,
                         remote_tp_rank: int = 0,
                         remote_tp_size: int = 1) -> str:

        engine_id = moriio_agent_meta.engine_id
        # TODO re-evaluate refreshing for scaling/recovery
        if remote_tp_rank in self._remote_agents.get(engine_id, {}):
            return self._remote_agents[engine_id][remote_tp_rank]

        if engine_id not in self._tp_size:
            self._tp_size[engine_id] = remote_tp_size
        else:
            assert self._tp_size[engine_id] == remote_tp_size
        # We may eventually enable this after asserting equality in cache
        # layout and close outputs.
        if moriio_agent_meta.attn_backend_name != self.backend_name:
            logger.info(
                f"!!!!!! Remote MoRIIO agent {engine_id} attention backend "
                f"'{moriio_agent_meta.attn_backend_name}' does not match "
                f"local backend '{self.backend_name}'.")

        remote_agent_name = "test"

        return remote_agent_name

    def get_finished(self) -> tuple[set[str], set[str]]:
        """
        Get requests that are done sending or recving on this specific worker.
        The scheduler process (via the MultiprocExecutor) will use this output
        to track which workers are done.
        """

        done_sending, done_recving = set(), set()

        if self.is_producer:
            done_sending = self.moriio_wrapper.pop_finished_req_ids()
            if GLOBAL_MORIIO_MODE == MoRIIOMode.WRITE:

                done_recving = set()
        else:
            if GLOBAL_MORIIO_MODE == MoRIIOMode.WRITE:
                self.moriio_wrapper.async_wait_reqid()
            done_sending, done_recving = set(
            ), self.moriio_wrapper.pop_finished_write_req_ids()

        return done_sending, done_recving

    def _get_new_notifs(self) -> set[str]:
        """
        Get req_ids which got a remote xfer message. When multiple consumers
        are reading from the same producer (heterogeneous TP scenario), wait
        for all consumers to be done pulling.
        """
        pass

    def _pop_done_transfers(self, done_req_ids) -> set[str]:

        return done_req_ids

    def save_kv_layer(self, metadata: MoRIIOConnectorMetadata, layer_name: str,
                      kv_layer: torch.Tensor,
                      attn_metadata: "AttentionMetadata", **kwargs):

        if not self.is_producer:
            return
        if GLOBAL_MORIIO_MODE == MoRIIOMode.READ:
            return
        remote_engine_id = None
        
        
        for req_id, meta in metadata.reqs_to_save.items():
            remote_engine_id = meta.remote_engine_id
            # we only need to check if dp0 in rank
            remote_engine_id = str(meta.remote_host) + ":" + str(
                meta.remote_handshake_port)
        
            meta.remote_engine_id = remote_engine_id

            # TODO: mz get_remote_engine_id() for engine_id mapping.
            dp0_remote_engine_id = f"{remote_engine_id}_dp0"
            if dp0_remote_engine_id not in self._remote_agents:
                # Initiate handshake with remote engine to exchange metadata.
                with self._handshake_lock:
                    if remote_engine_id not in self._remote_agents:
                        logger.info(
                            f"*****background moriio {remote_engine_id = }")
                        self._background_moriio_handshake(
                            req_id, remote_engine_id, meta)

                        continue
            self._write_blocks_for_req(req_id, meta, layer_name, kv_layer)

        while True:
            if remote_engine_id is None:
                break
            if self._ready_requests.empty() and remote_engine_id not in self.write_ready_flags:
                continue
            elif not self._ready_requests.empty() and (remote_engine_id
                                                       in self.write_ready_flags):
                self._write_blocks_for_req(*self._ready_requests.get_nowait(),
                                           layer_name, kv_layer)
                break
            else:
                break

            pass

    def start_load_kv(self, metadata: MoRIIOConnectorMetadata):
        """
        Start loading by triggering non-blocking moriio_xfer.
        We check for these trnxs to complete in each step().
        """
        # print("start load kv")
        if self.is_producer:
            self.moriio_wrapper.async_wait_reqid()
            return
        if GLOBAL_MORIIO_MODE == MoRIIOMode.WRITE:
            return

        wait_handshage_readd_req = False
        for req_id, meta in metadata.reqs_to_recv.items():
            remote_engine_id = meta.remote_engine_id
            # logger.debug(
            #     "start_load_kv for request %s from remote engine %s. "
            #     "Num local_block_ids: %s. Num remote_block_ids: %s. ", req_id,
            #     remote_engine_id, len(meta.local_block_ids),
            #     len(meta.remote_block_ids))
            if remote_engine_id not in self._remote_agents:
                # Initiate handshake with remote engine to exchange metadata.
                with self._handshake_lock:
                    if remote_engine_id not in self._remote_agents:
                        self._background_moriio_handshake(
                            req_id, remote_engine_id, meta)
                        wait_handshage_readd_req = True

                        continue

            # Handshake already completed, start async read xfer.
            self._read_blocks_for_req(req_id, meta)
        # Start transfers for requests whose handshakes have now finished.

        while True:  #TODO
            if self._ready_requests.empty(
            ) and not self.load_ready_flag and wait_handshage_readd_req:
                continue
            elif not self._ready_requests.empty() and self.load_ready_flag:
                self._read_blocks_for_req(*self._ready_requests.get_nowait())
                break
            else:
                break

        # Add to requests that are waiting to be read and track expiration.
        self._reqs_to_send.update(metadata.reqs_to_send)

        for req_id, req_meta in metadata.reqs_to_recv.items():

            self.moriio_wrapper.send_notify(
                req_id, req_meta.remote_host,
                req_meta.remote_notify_port + self.tp_rank)

    def _read_blocks_for_req(self, req_id: str, meta: ReqMeta):
        logger.debug(
            "Remote agent %s available, calling _read_blocks for req %s",
            meta.remote_engine_id, req_id)
        self._read_blocks(
            request_id=req_id,
            dst_engine_id=meta.remote_engine_id,
            local_block_ids=meta.local_block_ids,
            remote_block_ids=meta.remote_block_ids,
        )

    def _write_blocks_for_req(self, req_id: str, meta: ReqMeta, layer_name,
                              kv_layer):
        # logger.debug(f"write block for req {req_id} to remote engine "
        #             f"{meta.remote_engine_id}")
        self.schedule_write_blocks(request_id=req_id,
                                   dst_engine_id=meta.remote_engine_id,
                                   local_block_ids=meta.local_block_ids,
                                   remote_block_ids=meta.remote_block_ids,
                                   layer_name=layer_name,
                                   kv_layer=kv_layer,
                                   remote_notify_port=meta.remote_notify_port,
                                   remote_ip=meta.remote_host)

    def _is_last_layer(self, layer_name):
        if layer_name == list(self.kv_caches.keys())[-1]:
            return True
        return False

    def _is_first_layer(self, layer_name):
        if layer_name == list(self.kv_caches.keys())[0]:
            return True
        return False

    def this_layer_write_meta_offset(self):
        return self.merged_local, self.merged_remote, self.merged_sizes

    def merge_contiguous_blocks_fast_v2(
            self,
            offsets_local: List[int],
            offsets_remote: List[int],
            sizes: List[int],
            assume_sorted: bool = False
    ) -> Tuple[List[int], List[int], List[int]]:
        n = len(offsets_local)
        if n == 0:
            return [], [], []
        if not (n == len(offsets_remote) == len(sizes)):
            raise ValueError("Input list lengths mismatch")
        local_arr = np.fromiter(offsets_local, dtype=np.int64, count=n)
        remote_arr = np.fromiter(offsets_remote, dtype=np.int64, count=n)
        sizes_arr = np.fromiter(sizes, dtype=np.int64, count=n)

        if assume_sorted:
            local_sorted = local_arr
            remote_sorted = remote_arr
            sizes_sorted = sizes_arr
        else:
            if np.all(local_arr[:-1] <= local_arr[1:]):
                local_sorted = local_arr
                remote_sorted = remote_arr
                sizes_sorted = sizes_arr
            else:
                sort_idx = np.argsort(local_arr, kind="stable")
                local_sorted = local_arr[sort_idx]
                remote_sorted = remote_arr[sort_idx]
                sizes_sorted = sizes_arr[sort_idx]

        if n == 1:
            return [int(local_sorted[0])], [int(remote_sorted[0])
                                            ], [int(sizes_sorted[0])]

        diff_local = local_sorted[1:] - local_sorted[:-1]
        diff_remote = remote_sorted[1:] - remote_sorted[:-1]
        prev_size = sizes_sorted[:-1]

        contiguous = (diff_local == prev_size) & (diff_remote == prev_size)

        if not contiguous.any():
            return local_sorted.tolist(), remote_sorted.tolist(
            ), sizes_sorted.tolist()

        if contiguous.all():
            total_size = int(sizes_sorted.sum())
            return [int(local_sorted[0])], [int(remote_sorted[0])
                                            ], [total_size]

        break_positions = np.flatnonzero(~contiguous) + 1
        segment_starts = np.concatenate(([0], break_positions))
        segment_ends = np.concatenate((break_positions, [n]))

        seg_count = len(segment_starts)
        merged_local = [0] * seg_count
        merged_remote = [0] * seg_count
        merged_sizes = [0] * seg_count

        for si in range(seg_count):
            s = segment_starts[si]
            e = segment_ends[si]
            merged_local[si] = int(local_sorted[s])
            merged_remote[si] = int(remote_sorted[s])

            merged_sizes[si] = int(local_sorted[e - 1] + sizes_sorted[e - 1] -
                                   local_sorted[s])

        return merged_local, merged_remote, merged_sizes

    def _read_blocks(self, local_block_ids: list[int],
                     remote_block_ids: list[int], dst_engine_id: str,
                     request_id: str):

        if GLOBAL_MORIIO_MODE == MoRIIOMode.WRITE:
            return

        sessions = self._get_built_session(dst_engine_id)
        is_mla = (len(self.kv_cache_shape) == 3)

        a, b, c = [], [], []
        for layer_name, local_kv_cache_metadata in self.layer_name_to_local_kv_cache_metadata.items(
        ):

            if self._is_first_layer(layer_name):
                stride = self.kv_caches[layer_name].stride()
                if is_mla:
                    blknum, blksize, hs = self.kv_cache_shape
                    hn = 1
                    block_stride = stride[0]
                    ktov_stride = None
                else:
                    _, blknum, blksize, hn, hs = self.kv_cache_shape
                    ktov_stride = stride[0]
                    block_stride = stride[1]
                sz = self.kv_caches[layer_name].element_size()
                transfer_size_byte = blksize * hn * hs * sz
                per_block = 1 if is_mla else 2
                total = len(local_block_ids) * per_block
                offset_local = [0] * total
                offset_remote = [0] * total
                transfer_sizes = [transfer_size_byte] * total
                w = 0
                for i, lb in enumerate(local_block_ids):
                    rb = remote_block_ids[i]
                    # K
                    offset_local[w] = sz * (lb * block_stride)
                    offset_remote[w] = sz * (rb * block_stride)
                    w += 1
                    if not is_mla:
                        # V
                        offset_local[w] = sz * (1 * ktov_stride +
                                                lb * block_stride)
                        offset_remote[w] = sz * (1 * ktov_stride +
                                                 rb * block_stride)
                        w += 1
                    a, b, c = self.merge_contiguous_blocks_fast_v2(
                        offset_local,
                        offset_remote,
                        transfer_sizes,
                        assume_sorted=True)

            sess_idx = list(
                self.layer_name_to_local_kv_cache_metadata.keys()).index(
                    layer_name)
            use_batch = True
            if use_batch:
                self.moriio_wrapper.read_remote_data(c, a, b,
                                                     sessions[sess_idx])
            else:
                for i in range(len(a)):
                    self.moriio_wrapper.read_remote_data([c[i]], [a[i]],
                                                         [b[i]],
                                                         sessions[sess_idx])
            self.moriio_wrapper.waiting_for_transfer_complete()


@contextlib.contextmanager
def zmq_ctx(socket_type: Any, addr: str) -> Iterator[zmq.Socket]:
    """Context manager for a ZMQ socket"""

    if socket_type not in (zmq.ROUTER, zmq.REQ, zmq.DEALER):
        raise ValueError(f"Unexpected socket type: {socket_type}")

    ctx: Optional[zmq.Context] = None
    try:
        ctx = zmq.Context()  # type: ignore[attr-defined]
        yield make_zmq_socket(ctx=ctx,
                              path=addr,
                              socket_type=socket_type,
                              bind=socket_type == zmq.ROUTER)
    finally:
        if ctx is not None:
            ctx.destroy(linger=0)
