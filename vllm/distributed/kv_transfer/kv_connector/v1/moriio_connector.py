# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
import contextlib
import math
import queue
import threading
import time
import uuid
from collections import defaultdict
from collections.abc import Iterator
from concurrent.futures import Future, ThreadPoolExecutor
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, Optional,List
from vllm.utils import current_stream, get_ip
from vllm.distributed.parallel_state import get_world_group
import threading
 
import msgspec
import torch
import zmq
import msgpack
import socket
import pickle
import numpy as np
from typing import List, Tuple
from vllm import envs
from vllm.attention.selector import backend_name_to_enum, get_attn_backend
from vllm.config import VllmConfig
from vllm.distributed.kv_transfer.kv_connector.v1.base import (
    KVConnectorBase_V1, KVConnectorMetadata, KVConnectorRole)
from vllm.distributed.parallel_state import (
    get_tensor_model_parallel_rank, get_tensor_model_parallel_world_size,
    get_tp_group)
from vllm.distributed.utils import divide
from vllm.forward_context import ForwardContext
from vllm.logger import init_logger
from vllm.platforms import _Backend
from vllm.utils import make_zmq_path, make_zmq_socket, round_down
from vllm.v1.core.sched.output import SchedulerOutput
from vllm.v1.request import RequestStatus

if TYPE_CHECKING:
    from vllm.attention.backends.abstract import AttentionMetadata
    from vllm.v1.core.kv_cache_manager import KVCacheBlocks
    from vllm.v1.request import Request

Transfer = tuple[int, float]  # (xfer_handle, start_time)
EngineId = str
ReqId = str
GET_META_MSG = b"get_meta_msg"
POP_DONE_RECV = b"pop_done_recv"
OVER = b"OVER"
from enum import Enum  # 添加这行

class ROLE(Enum):
    PRODUCER = "producer"
    CONSUMER = "consumer"
    NOTINIT = "notinit"
GLOBAL_ROLE=ROLE.NOTINIT


class MoRIIOMode(Enum):
    READ = "read"
    WRITE = "write"

# 全局模式变量
GLOBAL_MORIIO_MODE = MoRIIOMode.WRITE
logger = init_logger(__name__)
def print_cur_time(strr):
    from datetime import datetime

    now = datetime.now()
    logger.info("!!!"+strr+str(now.strftime("%H:%M:%S.%f")[:-2]))
# Lazy import nixl_wrapper to avoid loading nixl_bindings if nixl is not used
try:
    import mori
    from mori.io import (
        IOEngineConfig,
        BackendType,
        IOEngine,
        EngineDesc,
        MemoryDesc,
        StatusCode,
    )
    logger.info("MoRIIO is available")
    MoRIIO_enabled = True
except ImportError:
    logger.error("MoRIIO is not available")
    MoRIIO_enabled = False

class MoRIIOWrapper():
    def __init__(self,moriio_engine = None):
        self.moriio_engine = moriio_engine
        self.remote_memory_metadata = None
        self.local_memory_registered = False
        self.local_memory_metadata = None
        self.transfer_status = []
        self.infilght_transfer_req_ids=[]
        self.remote_engine_ip = None
        self.notify_port = None
        self.notify_sock = None
        self.lock = threading.Lock()
        self.done_req_ids = []
        self.done_remote_allocate_req = []
        self.done_remote_allocate_req_dict: dict[str, list[int]] = {}
        self.done_write_cache_req_ids = []
        self.notify_thread = None
        self.sock = None
        self.tp_rank = get_tensor_model_parallel_rank()
        self.sessiones=[]
        self.has_register_remote_engine = False 
        self.kv_caches = None
        self.debug_id=1
        # self.remote_handshake_port = None # P->D 发送read完毕的reqid
        # self.local_handshake_port = None # D<-P 接收read完毕的reqid
        # self.waiting_for_remote_read_complete_thread = None # P 节点需要在D节点read完成之后才能安全释放blockid,因此这里管理
        

    def set_moriio_engine(self,moriio_engine):
        assert moriio_engine is not None,"You Cannot pass None engine to MoRIIOWrapper!"
        self.moriio_engine = moriio_engine

    def set_backend_type(self,backend_type):
        self.moriio_engine.create_backend(backend_type)

    def get_agent_metadata(self):
        engine_metadata = self.moriio_engine.get_engine_desc()
        engine_metadata_packed = engine_metadata.pack()
        return engine_metadata_packed
    
    def register_remote_engine(self,remote_packed_engine_metadata):
        consumer_engine_metadata = EngineDesc.unpack(remote_packed_engine_metadata)
        self.moriio_engine.register_remote_engine(consumer_engine_metadata)
        #TODO, bind for req
        self.has_register_remote_engine = True 
        return consumer_engine_metadata.key # str,engine name
    
    def register_local_tensor(self,tensor:torch.Tensor):
        try:
            self.local_memory_metadata = self.moriio_engine.register_torch_tensor(tensor)
            local_memory_metadata_packed = self.local_memory_metadata.pack()
        except Exception as e:
            logger.error(f"MoRIIO register local memory failed! reason = {e}")
        self.local_memory_registered = True
        return local_memory_metadata_packed
    
    def set_remote_memory_metadata(self,packed_memory_metadata):
        self.remote_memory_metadata = MemoryDesc.unpack(packed_memory_metadata)
    
    def set_local_memory_metadata(self,packed_memory_metadata):
        self.local_memory= packed_memory_metadata
        self.local_memory_metadata = MemoryDesc.unpack(packed_memory_metadata)
    
    
    def build_session(self):
        self.sessiones.append(self.moriio_engine.create_session(self.local_memory_metadata, self.remote_memory_metadata))




    def read_remote_data(self,transfer_size_byte,local_offset = 0,remote_offset = 0, sess_idx=0):
        assert self.remote_memory_metadata is not None,"You have not register remote memory data!"
        assert self.local_memory_registered,"You have not register local memory data!"
   
        transfer_status = self.sessiones[sess_idx].batch_read(
             local_offset, 
             remote_offset, 
            transfer_size_byte,
            self.moriio_engine.allocate_transfer_uid())
      
        self.transfer_status.append(transfer_status)
    def read_remote_data_s(self,transfer_size_byte,local_offset = 0,remote_offset = 0, sess_idx=0):
        assert self.remote_memory_metadata is not None,"You have not register remote memory data!"
        assert self.local_memory_registered,"You have not register local memory data!"
   
        transfer_status = self.sessiones[sess_idx].read(
             local_offset, 
             remote_offset, 
            transfer_size_byte,
            self.moriio_engine.allocate_transfer_uid())
      
        self.transfer_status.append(transfer_status)
    def write_remote_data(self,transfer_size_byte,local_offset = 0,remote_offset = 0, sess_idx=0):
        assert self.remote_memory_metadata is not None,"You have not register remote memory data!"
        assert self.local_memory_registered,"You have not register local memory data!"
        write_uid=self.moriio_engine.allocate_transfer_uid()+self.tp_rank*100000
        # print(write_uid)
        transfer_status = self.sessiones[sess_idx].batch_write(
             local_offset, 
             remote_offset, 
            transfer_size_byte,
            write_uid
            )
      
        self.transfer_status.append(transfer_status)
    def write_remote_data_s(self,transfer_size_byte,local_offset = 0,remote_offset = 0, sess_idx=0):
        assert self.remote_memory_metadata is not None,"You have not register remote memory data!"
        assert self.local_memory_registered,"You have not register local memory data!"
   
        transfer_status = self.sessiones[sess_idx].write(
             local_offset, 
             remote_offset, 
            transfer_size_byte,
            self.moriio_engine.allocate_transfer_uid())
      
        self.transfer_status.append(transfer_status)


    def waiting_for_read_complete(self):
        """等待所有传输完成的优化版本"""
        if not self.transfer_status:
            return
        
        # 批量处理，避免频繁的list操作
        transfers_to_wait = []
        with self.lock:
            transfers_to_wait = self.transfer_status[:]
            self.transfer_status.clear()
        
        # 并发等待（如果MoRIIO支持）
        for status in transfers_to_wait:
            try:
                st=time.perf_counter()
                status.Wait()
                if status.Succeeded():
                    pass
                    # logger.info(f"Transfer {status} succeeded")
                else:
                    logger.info(f"!!ggggg {status.Message()}")
                    logger.info(f"!!ggggg {status.Code()}")

                en=time.perf_counter()
                # logger.info(f"Transfer {status} completed in {en-st:.4f} seconds")
            except Exception as e:
                logger.error(f"Transfer {status} failed: {e}")
                raise
            
    def get_hash(self,n,local_block_ids):
        return self.kv_caches[list(self.kv_caches.keys())[n]][:,local_block_ids,:,:,:].sum()
    def get_all_hash(self,local_block_ids):
        hash_list = []
        for n in range(len(self.kv_caches)):
            hash_list.append(self.get_hash(n,local_block_ids).item())
        return hash_list
    def async_wait_reqid(self,kv_caches=None):
        # assert self.remote_engine_ip is not None,"remote engine ip is None!"
        # if self.tp_rank!=0:
        #     pass
        # return
        if kv_caches is not None:
            self.kv_caches = kv_caches
        assert self.notify_port is not None,"remote engine port is not None!"
        if self.notify_thread is not None:
            return
        def _async_wait():
            host = "*"
            path = make_zmq_path("tcp", host, self.notify_port)
            logger.info(f" node starting to listen notify from ..path = {path}")
            with zmq_ctx(zmq.ROUTER, path) as sock:
                while True:
                    identity, msg = sock.recv_multipart()
                    
                    try:
                        # 尝试反序列化为结构化数据
                        data = msgpack.loads(msg)
                        
                        if isinstance(data, dict) and "req_id" in data:
                            req_id = data["req_id"]
                            int_list = data.get("int_list", [])
                            msg_type = data.get("type", "unknown")
                            
                            print_cur_time(f"!!!zovlog:P received remote block msg: req_id={req_id}, type={msg_type}")
                            #TODO  更好的处理方法， 不然新旧request这里容易冲突
                            # torch.distributed.barrier(get_tp_group().device_group)
                            # 处理结构化消息 #TODO 修复初始化的问题
                            # if GLOBAL_ROLE == ROLE.PRODUCER:
                            with self.lock:
                                # 可以同时存储req_id和int_list
                                self.done_remote_allocate_req.append(req_id)
                                self.done_remote_allocate_req_dict[req_id] = int_list
                                b=0
                            # else:
                            #     assert False,"Only P node should receive this type of message!"
                                
                    except (msgpack.exceptions.ExtraData, msgpack.exceptions.UnpackException):
                    
                        
                        msg = msg.decode("UTF-8")
                        if  msg.startswith("cmpl"):
                                # assert 0,"P instance received error req id data"
                            if GLOBAL_ROLE==ROLE.PRODUCER:
                                # torch.distributed.barrier(get_tp_group().device_group)

                                # P节点执行   
                                with self.lock:  #可以释放page

                                    logger.info(f"zovlog:P received red id {msg} for release")
                                    self.done_req_ids.append(msg)
                            # D节点执行
                            else:
                            # elif GLOBAL_ROLE==ROLE.CONSUMER:
                                with self.lock:   
                                    # torch.distributed.barrier(get_tp_group().device_group)
                                    print_cur_time(f"!!!zovlog:D received write cache complete req id {msg}")
                                    self.done_write_cache_req_ids.append(msg)
                                    
                                    logger.info(f"{self.debug_id=} {str(self.get_all_hash(self.debug_id))}")
                                    self.debug_id+=1
                                    # time.sleep(5)
                    
                    
                    # else:
                    #     assert 0,"GLOBAL_ROLE is not set correctly!"
                    # TODO 没init前就send了？
        self.notify_thread = threading.Thread(target=_async_wait,daemon=True)
        self.notify_thread.start()
        
    
    
 
    
    
    # 使用msgpack序列化
  
    def send_notify(self,req_ids,meta=None):
        # logger.info(f"zovlog: enter sending notify to P...req_ids = {req_ids}")
        # if self.tp_rank!=0:
        #     pass
        # return 
        #TODO this should be assert 
        if self.remote_engine_ip is  None:
            self.remote_engine_ip=meta.remote_host
        assert self.notify_port is not None,"remote engine port is not None!"
        if not isinstance(req_ids,list):
            req_ids_ = [req_ids]
        else:
            req_ids_ = req_ids
        host = self.remote_engine_ip
        path = make_zmq_path("tcp", host, self.notify_port)
        
        #TODO make on
        if self.sock is None:
            self.ctx = zmq.Context()
            # path = make_zmq_path("tcp", host, self.notify_port)
            self.sock = make_zmq_socket(ctx=self.ctx,
                                path=path,
                                socket_type=zmq.DEALER,
                                bind=False)
            # with zmq_ctx(zmq.DEALER, path) as sock:
        for req in req_ids_:
            assert isinstance(req,str)
            print(f"zovlog: sending notify to P...req_ids_ = {req_ids_},path = {path}")
            self.sock.send(req.encode("utf-8"))
            # print(f"zovlog: sending notify to P finished")
    
    def pop_finished_req_ids(self):
        # P 节点调用
        with self.lock:
            done_send = set(self.done_req_ids)
            self.done_req_ids = []
        return done_send
    def pop_finished_write_req_ids(self):
        # D 节点调用
        with self.lock:
            if len(self.done_write_cache_req_ids)!=0:
                c=0
                # torch.distributed.barrier(get_tp_group().device_group)
            done_write_cache = set(self.done_write_cache_req_ids)
            self.done_write_cache_req_ids = []
        return done_write_cache
    def pop_remote_allocate_req_dict(self):
        # P 节点调用
        with self.lock:
            done_remote_allocate =set(self.done_remote_allocate_req)
            self.done_remote_allocate_req= []
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
    remote_handshake_port:int
    remote_engine_id: str
    tp_size: int


class MoRIIOConnectorMetadata(KVConnectorMetadata):

    def __init__(self):
        self.reqs_to_recv: dict[ReqId, ReqMeta] = {}
        self.reqs_to_save: dict[ReqId, ReqMeta] = {}
        self.reqs_to_send: dict[ReqId, float] = {}

    def __repr__(self):
        return_str = ""
        for req_id,req_meta in self.reqs_to_recv.items():
            return_str += f"{req_id = },{req_meta.local_block_ids = },{req_meta.remote_block_ids = },{req_meta.remote_host = },{req_meta.remote_port = },{req_meta.remote_engine_id = },{req_meta.tp_size = }"
        return_str = f"MoRIIOConnectorMetadata:reqs_to_recv:{return_str},"

        for req_id,req_meta in self.reqs_to_send.items():
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
        # wirte_mode=False
        # read_mode=True
        _req = ReqMeta(
            local_block_ids=local_block_ids,
            remote_block_ids=kv_transfer_params["remote_block_ids"],
            remote_engine_id=kv_transfer_params["remote_engine_id"],
            remote_host=kv_transfer_params["remote_host"],
            remote_port=kv_transfer_params["remote_port"],
            remote_handshake_port=kv_transfer_params['remote_handshake_port'],
            # P workers don't need to receive tp_size from proxy here.
            tp_size=kv_transfer_params.get("tp_size", 1),
        )


        if write_mode:
            self.reqs_to_save[request_id] = _req
        else:
            self.reqs_to_recv[request_id] = _req
class MoRIIOConnector(KVConnectorBase_V1):

    def __init__(self, vllm_config: VllmConfig, role: KVConnectorRole):
        assert vllm_config.kv_transfer_config is not None
        assert vllm_config.kv_transfer_config.engine_id is not None
        self.engine_id: EngineId = vllm_config.kv_transfer_config.engine_id

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
        return self.connector_scheduler.get_num_new_matched_tokens(request, num_computed_tokens)

    def update_state_after_alloc(self, request: "Request",
                                 blocks: "KVCacheBlocks",
                                 num_external_tokens: int):
        assert self.connector_scheduler is not None
        return self.connector_scheduler.update_state_after_alloc(
            request, blocks, num_external_tokens,self.connector_worker)

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
        if GLOBAL_MORIIO_MODE==MoRIIOMode.WRITE:
            if GLOBAL_ROLE==ROLE.CONSUMER:
                self.connector_worker.moriio_wrapper.async_wait_reqid()
        st = time.perf_counter()
        assert self.connector_worker is not None
        assert isinstance(self._connector_metadata, MoRIIOConnectorMetadata)
        self.connector_worker.start_load_kv(self._connector_metadata)
        en = time.perf_counter()
        # print(f"start_load_kv总时间{en - st} sec")

    def wait_for_layer_load(self, layer_name: str) -> None:
        """NixlConnector does not do layerwise saving."""
        pass

    def save_kv_layer(self, layer_name: str, kv_layer: torch.Tensor,
                      attn_metadata: "AttentionMetadata", **kwargs) -> None:
        
         # Only producer/prefill saves KV Cache
    
        self.connector_worker.save_kv_layer(self._connector_metadata, layer_name, kv_layer, attn_metadata, **kwargs)
        return None


        
    def wait_for_save(self):
        """NixlConnector does not save explicitly."""
        
        # self.connector_worker.moriio_wrapper.waiting_for_read_complete()
        pass


class MoRIIOConnectorScheduler:
    """Implementation of Scheduler side methods"""

    def __init__(self, vllm_config: VllmConfig, engine_id: str):
        self.vllm_config = vllm_config
        self.block_size = vllm_config.cache_config.block_size
        self.engine_id: EngineId = engine_id
        self.side_channel_host = envs.VLLM_NIXL_SIDE_CHANNEL_HOST
        self.side_channel_port = (
            self.vllm_config.kv_transfer_config.kv_connector_extra_config['handshake_port'], # envs.VLLM_NIXL_SIDE_CHANNEL_PORT +
            self.vllm_config.parallel_config.data_parallel_rank *
            self.vllm_config.parallel_config.tensor_parallel_size)
        logger.info(f"zovlog::==========> Initializing MoRIIO Scheduler {engine_id = },{self.side_channel_port = }")
        
        #todo , how to send with tp
        self.side_notify_port = self.vllm_config.kv_transfer_config.kv_connector_extra_config['notify_port'] # envs.VLLM_NIXL_SIDE_CHANNEL_PORT +
        self.tp_size=self.vllm_config.parallel_config.tensor_parallel_size
         
        self.is_producer = vllm_config.kv_transfer_config.kv_role == "kv_producer"
        # self.gotted = False
        # Requests that need to start recv/send.
        # New requests are added by update_state_after_alloc in
        # the scheduler. Used to make metadata passed to Worker.
        self._reqs_need_recv: dict[ReqId, tuple[Request, list[int]]] = {}
        self._reqs_need_save: dict[ReqId, tuple[Request, list[int]]] = {}

        # Reqs to send and their expiration time
        self._reqs_need_send: dict[ReqId, float] = {}
        self.sock = None
        self.is_producer = vllm_config.kv_transfer_config.kv_role == "kv_producer"

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
            return 0,False
        
        # logger.info(f"zovlog:==============> call get_num_new_matched_tokens,{request.kv_transfer_params = }")
        params = request.kv_transfer_params
        # logger.debug(
        #     "NIXLConnector get_num_new_matched_tokens: "
        #     "num_computed_tokens=%s, kv_transfer_params=%s",
        #     num_computed_tokens, params) #TODO mingzhi write
        if GLOBAL_MORIIO_MODE == MoRIIOMode.WRITE:
            # MoriiO in write mode, no remote prefill
           
            return len(request.prompt_token_ids)  - num_computed_tokens,True

        else: 
            return len(request.prompt_token_ids)  - num_computed_tokens,False
        # if params is not None and params.get("do_remote_prefill"):
        #     # Remote prefill: get all prompt blocks from remote.
        #     assert num_computed_tokens % self.block_size == 0
        #     rounded_num_prompt_tokens = round_down(len(request.prompt_token_ids), self.block_size)
        #     # rounded_num_prompt_tokens = len(request.prompt_token_ids)
        #     count = max(rounded_num_prompt_tokens - num_computed_tokens, 0)
        #     logger.info(f"zovlog:===============> call get_num_new_matched_tokens,{len(request.prompt_token_ids) = },{self.block_size = },{round_down(len(request.prompt_token_ids), self.block_size) = },{num_computed_tokens = },{count = }")
        #     # if count > 0:
        #     #     return count,False
        #     if count > 0:
        #         return count, True

        # No remote prefill for this request.
        return 0, False
    def send_notify_block(self, req_id: str, int_list: list[int] = None,host=None,port=None):
        
        # def, todo TP>1?

        """发送req_id和int列表"""
    #   /sz
        
    #     host = self.remote_engine_ip
        path = make_zmq_path("tcp", host, port)
        #TODO:     make once     
        # if self.sock is None:
        self.ctx = zmq.Context()
        self.sock = make_zmq_socket(ctx=self.ctx,
                        path=path,
                        socket_type=zmq.DEALER,
                        bind=False)
        
        # 构造要发送的数据结构
        data = {
            "req_id": req_id,
            "int_list": int_list or [],
            "type": "remote_blocks"
        }
        serialized_data = msgpack.dumps(data)
        logger.info(f"zovlog: sending block slots with data to P...req_id = {req_id}, , path = {path}")
        self.sock.send(serialized_data)
    def update_state_after_alloc(self, request: "Request", # 包含remote使用到的blockid
                                 blocks: "KVCacheBlocks", # local 分配好的blockid
                                 num_external_tokens: int,
                                 connector_worker: Optional["MoRIIOConnectorWorker"]=None):
        
        params = request.kv_transfer_params # zovlog: params 是none
        
        
        
        if params.get("do_remote_decode"):
            local_block_ids = blocks.get_block_ids()[0]
            self._reqs_need_save[request.request_id] = (
                        request, local_block_ids)
        # logger.info(
            # f"moriioConnector update_state_after_alloc: "
            # f"num_external_tokens={num_external_tokens}, kv_transfer_params={params},{params.get("do_remote_prefill") = },{params.get("remote_block_ids") = }")
        
        # if GLOBAL_MORIIO_MODE == MoRIIOMode.READ:
        if params is not None and params.get("do_remote_prefill"):
            if GLOBAL_MORIIO_MODE==MoRIIOMode.READ:
                if remote_block_ids := params.get("remote_block_ids"):
                    if all(p in params for p in ("remote_engine_id", "remote_host",
                                                "remote_port")):
                        # If remote_blocks and num_external_tokens = 0, we
                        # a full prefix cache hit on the D worker. We need to call
                        # send_notif in _read_blocks to free the memory on the P.

                        # local_block_ids = (blocks.get_unhashed_block_ids()
                        #                    if num_external_tokens > 0 else [])
                        # 临时修改测试,如果local分配的和remote的长度不一样,那么就说明只需要load remote的后面几个
                        # Get unhashed blocks to pull from remote.
                        local_block_ids = blocks.get_block_ids()[0]
                        assert len(local_block_ids) <= len(remote_block_ids)
                        if len(local_block_ids) == len(remote_block_ids):
                            # 全部需要load,pass
                            pass
                        else:
                            # 只需要load prefix cacheing 未命中的部分
                            local_block_ids = remote_block_ids[-len(local_block_ids):]
                            # logger.info(f"zovlog:08--------------> len(local_block_ids) < len(remote_block_ids),{local_block_ids = }")
                        # logger.info(f"zovlog:08 ------------> unhashed blocks = {local_block_ids}")
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
                # MoriiO in write mode, do remote prefill(sonsumer)
                # send block ids
                #TODO , decode allocate wich times?
                # send_no
                print_cur_time("!!!send_notify_block called!")
                for tp_index in range(self.tp_size):
                    cur_port=self.side_notify_port+tp_index
                    self.send_notify_block(req_id=request.request_id,int_list=blocks.get_block_ids()[0],host=params.get("remote_host"),port=cur_port)
                print_cur_time("!!!send_notify_block call finished!")

                # assert num_external_tokens == 0f
            # Only trigger 1 KV transfer per request.
            #这里可能是  那个get_mun_new_matched_tokens，为了显存允许decode做一点prefill
            params["do_remote_prefill"] = False

    def build_connector_meta(
        self,
        scheduler_output: SchedulerOutput,
    ) -> KVConnectorMetadata:
        meta = MoRIIOConnectorMetadata()

        if GLOBAL_MORIIO_MODE==MoRIIOMode.WRITE :
        # when aysnc_load_kv finished, will add new reqs to scheduler_output.scheduled_new_reqs
        # should I use thread to add new req in async_wait_reqid?
            for new_req in scheduler_output.scheduled_new_reqs:
                red_id=new_req.req_id
                local_block_ids = list(new_req.block_ids)
                kv_transfer_params = new_req.sampling_params.extra_args['kv_transfer_params']
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
            "NIXLConnector request_finished, request_status=%s, "
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
        # logger.info(f"zovlog:--------------> calculate all full!!!!! {request.num_computed_tokens = },{self.block_size = },{request.num_computed_tokens % self.block_size = }")
        all_full = request.num_computed_tokens % self.block_size == 0
        # computed_block_ids = block_ids if all_full else block_ids[:-1]
        # 不论是否已满,都要传输全部的blockids
        computed_block_ids = block_ids
        # If prompt < block_size, no xfer so free blocks immediately.
        delay_free_blocks = len(computed_block_ids) > 0

        if delay_free_blocks:
            # Prefill request on remote. It will be read from D upon completion
            self._reqs_need_send[request.request_id] = time.perf_counter(
            ) + envs.VLLM_NIXL_ABORT_REQUEST_TIMEOUT
        # logger.info(f"zovlog0831:----------> call moriio connector request finished {computed_block_ids = }")
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
        self.block_size = vllm_config.cache_config.block_size
        self.kv_transfer_config = vllm_config.kv_transfer_config
        self.is_producer = self.kv_transfer_config.is_kv_producer
        if self.is_producer:
            GLOBAL_ROLE=ROLE.PRODUCER
        else: 
            GLOBAL_ROLE=ROLE.CONSUMER
        # mori engine
        self._rank = get_world_group().rank 
        self._local_rank = get_world_group().local_rank 
        self.tp_rank = get_tensor_model_parallel_rank()

        self.local_ip = get_ip() # P/D节点自身的IP
        self.local_kv_port = int(self.kv_transfer_config.kv_port) # D节点拉取kvcache的时候使用的port
        self.local_kv_port = self.local_kv_port + self.tp_rank
        self.proxy_ip = self.kv_transfer_config.kv_connector_extra_config["proxy_ip"] # proxy自身的IP,也是用户唯一需要识别的IP,也是P/D节点上报信息的IP
        self.proxy_port = int(self.kv_transfer_config.kv_connector_extra_config["proxy_port"]) # 用于监听用户prompt的port,与用户交互的port
        
        self.local_ping_port = int(self.kv_transfer_config.kv_connector_extra_config["local_ping_port"]) # P/D节点上报自身信息时使用的port
        
        self.local_ping_port = self.local_ping_port+self.tp_rank

        self.proxy_ping_port = int(self.kv_transfer_config.kv_connector_extra_config["proxy_ping_port"]) # P/D节点将自身信息上报至这个port
        
        self.http_port = int(self.kv_transfer_config.kv_connector_extra_config['http_port']) # 用于接收request的port
        self.handshake_port = int(self.kv_transfer_config.kv_connector_extra_config['handshake_port']) # 用于handshake的本地port,remote的port会在运行中从proxy获取
        self.notify_port = int(self.kv_transfer_config.kv_connector_extra_config['notify_port'])
        self.notify_port=self.notify_port+self.tp_rank
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
        if not self.is_producer:
            self.poller = zmq.Poller()
            self.metadata_socket = self.zmq_context.socket(zmq.ROUTER)
            self.metadata_socket.bind(f"tcp://{self.metadata_address}")
            self.poller.register(self.metadata_socket, zmq.POLLIN)

            logger.info(f"build IOEngine {self.local_ip},{self.local_kv_port}")
            self.moriio_engine = IOEngine("consumer",IOEngineConfig(self.local_ip,self.local_kv_port))
            self._handle_request_thread = threading.Thread(target = self.handle_proxy_request,daemon=True)
            self._handle_request_thread.start()
        else:
            logger.info(f"build IOEngine {self.local_ip},{self.local_kv_port}")

            self.moriio_engine = IOEngine("producer",IOEngineConfig(self.local_ip,self.local_kv_port))
        if self._rank == 0 and self.proxy_ip != "":
            self._ping_thread = threading.Thread(target=self._ping,args=(self.zmq_context,),daemon=True)
            self._ping_thread.start() # join?

        logger.info(f"Initializing MoRIIO Engine ,engine = {self.moriio_engine},role = {'producer' if self.is_producer else 'consumer'}")
        logger.info(f"zovlog:=====>{self.local_ip = },{self._rank = },{self._local_rank = },{self.local_kv_port = },{self.proxy_ip = },{self.proxy_port = },{self.local_ping_port = },{self.proxy_ping_port = }")
        # Agent.
        self.moriio_wrapper = MoRIIOWrapper()
        logger.info(f"{self._rank = }:set_moriio_engine")
        self.moriio_wrapper.set_moriio_engine(self.moriio_engine)
        logger.info(f"{self._rank = }:set_moriio_engine end")
        
        self.moriio_wrapper.set_backend_type(BackendType.RDMA)
        logger.info(f"{self._rank = }:set_moriio_backend end")

        self.moriio_wrapper.notify_port = self.notify_port
        self.local_kv_cache_metadata = []
        self.local_kv_cache_size = []
        self.layer_name_to_local_kv_cache_metadata:dict[str, List[Any]] = dict()

        self.remote_kv_cache_metadata = []
        self.remote_kv_cache_size = []
        self.layer_name_to_remote_kv_cache_metadata:dict[str, List[Any]] = dict()
        self.slot_size_bytes = 0

        self.load_kv_flag = False # False 代表从未load过
        self.write_kv_flag=False
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
            self.vllm_config.parallel_config.data_parallel_rank * self.vllm_config.parallel_config.tensor_parallel_size)

        # Metadata.
        self.engine_id: EngineId = engine_id
   
        
        
        self.world_size = get_tensor_model_parallel_world_size()
        self.tp_group = get_tp_group()

        # KV Caches and nixl tracking data.
        self.kv_caches: dict[str, torch.Tensor] = {}

        # Map of engine_id -> kv_caches_base_addr. For TP case, each local
        # rank will still only pull from a single remote TP worker.
        self.kv_caches_base_addr: dict[EngineId, list[int]] = {}

        # Number of MoRIIO regions. Currently one region per cache
        # (so 1 per layer for MLA, otherwise 2 per layer)
        self.num_regions = 0
        self.num_layers = 0

        # nixl_prepped_dlist_handle.
        self.src_xfer_side_handle: int = 0
        # Map of engine_id -> nixl_prepped_dlist_handle (int)].
        self.dst_xfer_side_handles: dict[EngineId, int] = {}

        # Map of engine_id -> num_blocks. All ranks in the same deployment will
        # have the same number of blocks.
        self.dst_num_blocks: dict[EngineId, int] = {}
        self._registered_descs: list[Any] = []
        self.finished_int=0
        # In progress transfers.
        # [req_id -> list[handle]]
        self._recving_transfers = defaultdict[ReqId, list[Transfer]](list)
        # Track the expiration time of requests that are waiting to be sent.
        self._reqs_to_send: dict[ReqId, float] = {}

        # Background thread for handling new handshake requests.
        self._nixl_handshake_listener_t: Optional[threading.Thread] = None
        # Background thread for initializing new MoRIIO handshakes.
        self._handshake_initiation_executor = ThreadPoolExecutor(
            # MoRIIO is not guaranteed to be thread-safe, limit 1 worker.
            max_workers=1,
            thread_name_prefix="vllm-nixl-handshake-initiator")
        self._ready_requests = queue.Queue[tuple[ReqId, ReqMeta]]()
        self._handshake_futures: dict[EngineId, Future[dict[int, str]]] = {}
        # Protects _handshake_futures and _remote_agents.
        self._handshake_lock = threading.RLock()

        self.vllm_config = vllm_config
        self.block_size = vllm_config.cache_config.block_size
        self.model_config = vllm_config.model_config
        self.cache_config = vllm_config.cache_config

        # TODO(mgoin): remove this once we have hybrid memory allocator
        # Optimization for models with local attention (Llama 4)
        # List of block window sizes for each layer for local attention
        self.block_window_per_layer: list[Optional[int]] = []
        self.use_mla = self.model_config.use_mla
        self.builded_session = False
        self.builded_write_session =False
        self.debug_cache=[]
        backend = get_attn_backend(self.model_config.get_head_size(),
                                   self.model_config.dtype,
                                   self.cache_config.cache_dtype,
                                   self.block_size,
                                   self.model_config.is_attention_free,
                                   use_mla=self.use_mla)
        self.backend_name = backend.get_name()
        attn_backend = backend_name_to_enum(self.backend_name)
        self._use_flashinfer = attn_backend == _Backend.FLASHINFER_VLLM_V1
        logger.debug("Detected attention backend %s", self.backend_name)

        self._tp_size: dict[EngineId, int] = {self.engine_id: self.world_size}
        # With heterogeneous TP, P must wait for all assigned D TP workers to
        # finish reading before safely freeing the blocks.
        self.consumer_notification_counts_by_req = defaultdict[ReqId, int](int)

    def _ping(self,zmq_context):
        index = 1
        sock = zmq_context.socket(zmq.DEALER)
        sock.connect(f"tcp://{self.proxy_ip}:{self.proxy_ping_port}")
        while True:
            try:
                http_request_address = "http://" + self.request_address +"/v1/completions"
                data = {"type":"register","role":"P" if self.is_producer else "D","index":str(index),"request_address":http_request_address,"handshake_port":self.handshake_port}
                
                sock.send(msgpack.dumps(data))
                # print(f"zovlog:====>Sent: {data}")
            except ConnectionRefusedError:
                logger.info(f"zovlog:====> {(self.local_ip,self.local_ping_port)},'->',{(self.proxy_ip, self.proxy_ping_port)} send failed,connection refused")
            except OSError as e:
                logger.info(f"zovlog:===> send failed , os error {e}")
            except Exception as e:
                logger.info(f"zovlog:===> send failed , unknown error {e}")
            finally:
                time.sleep(10)
                index += 1

    def handle_proxy_request(self):
        if self.is_producer:
            raise NotImplementedError("prefill instance doesn't need to send kv cache in pull mode")
        while True:
            socks = dict(self.poller.poll())
            logger.info(f"zovlog:====> handle_proxy_request: {socks = },{self.router_socket = }")
            if self.metadata_socket not in socks:
                continue
            else:
                pass


    def __del__(self):
        """Cleanup background threads on destruction."""
        self._handshake_initiation_executor.shutdown(wait=False)
        if self._nixl_handshake_listener_t:
            self._nixl_handshake_listener_t.join(timeout=0)
        
        
    @staticmethod
    def _nixl_handshake_listener(metadata: MoRIIOAgentMetadata,
                                 ready_event: threading.Event, base_port: int,
                                 tp_rank: int,layer_name_to_local_kv_cache_metadata:dict ):
        """Background thread for getting new MoRIIO handshakes."""

        encoder = msgspec.msgpack.Encoder()
        encoded_data = encoder.encode(metadata)
        size_in_bytes = len(encoded_data)
        logger.debug("Size of encoded MoRIIOAgentMetadata: %s bytes",
                     str(size_in_bytes))

        # Listen for new requests for metadata.
        host = "*" # envs.VLLM_NIXL_SIDE_CHANNEL_HOST
        path = make_zmq_path("tcp", host, base_port + tp_rank)
        logger.info(f"zovlog:======> Starting listening on path: {path}")
        with zmq_ctx(zmq.ROUTER, path) as sock:
            ready_event.set()
            while True:
                identity, msg = sock.recv_multipart()
                if msg != GET_META_MSG and msg != POP_DONE_RECV:
                    logger.warning("Connection listener got unexpected message %s", msg)
                    assert 0,"handhsake failed!!!!!!!!!!"
                elif msg == GET_META_MSG:
                    # P instance send engine desc?
                    # 既然req数据中能够包含engine id,那么同样可以不包含P节点自身的desc,直接写入即可
                    # 但是现在暂时采用发送的方式
                    logger.info(f"zovlog:=======> P instance handshake msg received!!!!!!!,identity = {identity}")
                    sock.send_multipart((identity, b"", encoded_data)) # send local mori io engine meta data

                    # now we send tensor meta data for each block
                    buf = pickle.dumps(layer_name_to_local_kv_cache_metadata)
                    sock.send_multipart((identity, b"", buf))
                    # logger.info(f"zovlog:=====> P all sent.............. {layer_name_to_local_kv_cache_metadata = }")
                elif msg == POP_DONE_RECV:
                    _, req_id = sock.recv_multipart()
                    
                else:
                    pass


    def _nixl_handshake(
        self,
        host: str,
        port: int,
        remote_tp_size: int,
        expected_engine_id: str,
    ) -> dict[int, str]:
        """Do a MoRIIO handshake with a remote instance."""

        start_time = time.perf_counter()

        # NOTE(rob): we need each rank to have a unique port. This is
        # a hack to keep us moving. We will switch when moving to etcd
        # or where we have a single ZMQ socket in the scheduler.

        # Handshake only with the remote TP rank that current local rank will
        # pull from. With homogeneous TP it happens to be the same rank_i.
        # logger.info(f"zovlog:--------------------> call _nixl_handshake {self.engine_id = },{self._tp_size = },{remote_tp_size = },{self.tp_rank = },{host = },{port = },")
        
        tp_ratio = self._tp_size[self.engine_id] // remote_tp_size # _tp_size根据engine id 查询这个engine 的tp大小
        tp_ratio=1 #sfor debug
        p_remote_rank = self.tp_rank // tp_ratio
        path = make_zmq_path("tcp", host, port + p_remote_rank)
        logger.info("Querying metadata on path: %s at remote rank %s", path,p_remote_rank)

        # Send query for the request.
        with zmq_ctx(zmq.DEALER, path) as sock:
            # logger.info(f"zovlog:=======> prepare send msg to P INSTAZNCE")
            sock.send(GET_META_MSG)
            # logger.info(f"zovlog:=======> send finished,prepare recvive")
            received_frame = sock.recv_multipart()
            if len(received_frame) != 2 or received_frame[0] != b"":
                assert 0,f"unexpected frame! {received_frame = }"
                
            metadata_bytes = received_frame[1]
            decoder = msgspec.msgpack.Decoder(MoRIIOAgentMetadata)
            metadata = decoder.decode(metadata_bytes)
            got_metadata_time = time.perf_counter()
            logger.debug("MoRIIO handshake: get metadata took: %s",
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
            remote_agent_name = self.moriio_wrapper.register_remote_engine(metadata.agent_metadata)
            remote_agent_name = self.add_remote_agent(metadata, p_remote_rank,remote_tp_size)
            if len(self.local_kv_cache_metadata) > 0:
                logger.warning(f"zovlog:=======> {len(self.local_kv_cache_metadata) = },maybe you didnt clear this buffer correctly")
                self.local_kv_cache_metadata = []
            if len(self.remote_kv_cache_metadata) > 0:
                logger.warning(f"zovlog:=======> {len(self.remote_kv_cache_metadata) = },maybe you didnt clear this buffer correctly")
                self.remote_kv_cache_metadata = []

            received_frame = sock.recv_multipart()
            if len(received_frame) != 2 or received_frame[0] != b"":
                assert 0,f"Unexpected frame! {received_frame = }"
            buf = received_frame[1]
            self.layer_name_to_remote_kv_cache_metadata = pickle.loads(buf)
                
            setup_agent_time = time.perf_counter()
            logger.debug("MoRIIO handshake: add agent took: %s",setup_agent_time - got_metadata_time)

      
        # Remote rank -> agent name.
        # logger.info(f"zovlog:====> {p_remote_rank = },{remote_agent_name = }")
        return {p_remote_rank: remote_agent_name}

    def _background_nixl_handshake(self, req_id: str,
                                   remote_engine_id: EngineId, meta: ReqMeta):
        # Do MoRIIO handshake in background and add to _ready_requests when done.
        fut = None
        if remote_engine_id is not None:
            fut = self._handshake_futures.get(remote_engine_id)
        if fut is None:
            host = meta.remote_host
            # port = int(meta.remote_port)
            port = int(meta.remote_handshake_port)
            tp_size = int(meta.tp_size)
            fut = self._handshake_initiation_executor.submit(self._nixl_handshake, host, port,tp_size, remote_engine_id)
            
          
            
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

        # TODO: handle failure state of future in the
        # callback, we want to fail the request in this case.
        def request_ready(_f: Future[Any], entry=(req_id, meta)):
            self._ready_requests.put(entry)
            self.load_kv_flag = True 
            self.write_kv_flag=True

      
        fut.add_done_callback(request_ready)

    def register_kv_caches(self, kv_caches: dict[str, torch.Tensor]):
        """Register the KV Cache data in nixl."""
        """只会在llmengine初始化的时候调用一次,注册所有已经分配的kvcache pool"""
        for _,t in kv_caches.items():
            t = t.zero_() # for debug,not necessary
            # logger.info(f"zovlog:===========> enter register kv cache,name = {_},shape = {t.shape}")
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

        # TODO (NickLucche) not compatible with hybrid allocator. Enforce check
        # once it goes live, as a single kv layout is expected for xfers.
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
            self.slot_size_bytes = kv_elem_size * n_kv_heads * head_dim # 1 token 1 layer size , slot size
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
        self.kv_caches = kv_caches # layer name to kv cache
        kv_caches_base_addr = []
        caches_data = []


        """到此,已经确认了以下信息"""
        # 传入的kvcache是一个字典,key是每一层的名称,value是这一层开辟的所有kvcache的空间
        # 每一层kvcache的整体形状为 [2,blknum,blksize,headnum,headsize]
        # 我需要注册所有层的所有kvcache,后续传输的时候需要按照使用到的blkid,计算出这个blk对应的offset再启动传输

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
        # logger.info(f"zovlog:======> {kv_cache_key_list = },{kv_cache_shape_list = }")
        for cache_or_caches in kv_caches.values():
        # 对每一个block ,都要以基址+长度注册一个protection domain
        # 由于_nixl_handshake_listener中无法访问到这些信息
        # 因此我只能在这里注册内存地址
            
            cache_list = [cache_or_caches] if use_mla or self._use_flashinfer else cache_or_caches
            # logger.info(f"zovlog:=============> prepare register local kv cache tensor for local mori io engine,{len(cache_list) = },{kv_caches.keys() = }")
            for cache in cache_list:
                # moriio_mem_metadata = self.moriio_wrapper.register_local_tensor(cache) # register one block
                # self.local_kv_cache_metadata.append(moriio_mem_metadata)
                # self.local_kv_cache_size.append(cache.nelement() * cache.element_size())
                # logger.info(f"zovlog::===========> registered:{self.local_kv_cache_size[-1] = },{self.local_kv_cache_metadata[-1] = },{self.block_len = },{self.num_blocks = },{kv_elem_size = },{first_kv_cache.shape = },{block_shape = }")
                base_addr = cache.data_ptr()
                region_len = self.num_blocks * self.block_len
                caches_data.append(
                    (base_addr, region_len, cache.device.index, ""))
                kv_caches_base_addr.append(base_addr)

        for layer_name,kv_cache in kv_caches.items():
            # cache_list = [kv_cache] if use_mla or self._use_flashinfer else kv_cache
            # workround
            # logger.info(f"zovlog:===========>{len(cache_list) = }")
            if layer_name not in self.layer_name_to_local_kv_cache_metadata:
                self.layer_name_to_local_kv_cache_metadata[layer_name] = []

            # for cache in cache_list:
            # moriio_mem_metadata = self.moriio_wrapper.register_local_tensor(cache) 
            moriio_mem_metadata = self.moriio_wrapper.register_local_tensor(kv_cache) 
            self.layer_name_to_local_kv_cache_metadata[layer_name].append(moriio_mem_metadata)
            
            
            self.local_kv_cache_size.append(cache.nelement() * cache.element_size())
            # logger.info(f"zovlog::===========> registered:{self.local_kv_cache_size[-1] = },{self.layer_name_to_local_kv_cache_metadata[layer_name][-1] = },{self.block_len = },{self.num_blocks = },{kv_cache.shape = },{block_shape = }")


        
       
            
            
        
        
        self.kv_caches_base_addr[self.engine_id] = kv_caches_base_addr
        self.num_regions = len(caches_data)
        self.num_layers = len(self.kv_caches.keys())

        # TODO(mgoin): remove this once we have hybrid memory allocator
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
        
        # P节点在register kvcache的时候就会启动一个监听线程,等待D节点拉取数据
        metadata = MoRIIOAgentMetadata(
            engine_id=self.engine_id,
            agent_metadata=self.moriio_wrapper.get_agent_metadata(),
            kv_caches_base_addr=self.kv_caches_base_addr[self.engine_id],
            num_blocks=self.num_blocks,
            block_len=self.block_len,
            attn_backend_name=self.backend_name)
        ready_event = threading.Event()
        self._nixl_handshake_listener_t = threading.Thread(
            target=self._nixl_handshake_listener,
            args=(metadata, ready_event, self.side_channel_port, self.tp_rank,self.layer_name_to_local_kv_cache_metadata),
            daemon=True,
            name="nixl_handshake_listener")
        self._nixl_handshake_listener_t.start()
        ready_event.wait()  # Wait for listener ZMQ socket to be ready.
        self.moriio_wrapper.async_wait_reqid(self.kv_caches)

   
        '''
        descs = self.moriio_wrapper.get_reg_descs(caches_data, "VRAM")
        logger.debug("Registering descs: %s", caches_data)
        self.moriio_wrapper.register_memory(descs)
        logger.debug("Done registering descs")
        self._registered_descs.append(descs)

        # Register local/src descr for MoRIIO xfer.
        blocks_data = []
        for base_addr in self.kv_caches_base_addr[self.engine_id]:
            # NOTE With heter-TP, more blocks are prepared than what are
            # needed as self.num_blocks >= nixl_agent_meta.num_blocks. We
            # could create fewer, but then _get_block_descs_ids needs to
            # select agent_meta.num_blocks instead of self.num_blocks for
            # local descr, and that makes handling regular flow less clean.
            for block_id in range(self.num_blocks):
                block_offset = block_id * self.block_len
                addr = base_addr + block_offset
                # (addr, len, device id)
                blocks_data.append((addr, self.block_len, self.tp_rank))
        logger.debug("Created %s blocks for src engine %s and rank %s",
                     len(blocks_data), self.engine_id, self.tp_rank)

        descs = self.moriio_wrapper.get_xfer_descs(blocks_data, "VRAM")
        # NIXL_INIT_AGENT to be used for preparations of local descs.
        self.src_xfer_side_handle = self.moriio_wrapper.prep_xfer_dlist(
            "NIXL_INIT_AGENT", descs)

        # After KV Caches registered, listen for new connections.
        metadata = MoRIIOAgentMetadata(
            engine_id=self.engine_id,
            agent_metadata=self.moriio_wrapper.get_agent_metadata(),
            kv_caches_base_addr=self.kv_caches_base_addr[self.engine_id],
            num_blocks=self.num_blocks,
            block_len=self.block_len,
            attn_backend_name=self.backend_name)
        ready_event = threading.Event()
        self._nixl_handshake_listener_t = threading.Thread(
            target=self._nixl_handshake_listener,
            args=(metadata, ready_event, self.side_channel_port, self.tp_rank),
            daemon=True,
            name="nixl_handshake_listener")
        self._nixl_handshake_listener_t.start()
        ready_event.wait()  # Wait for listener ZMQ socket to be ready.
        '''

    def add_remote_agent(self,
                         nixl_agent_meta: MoRIIOAgentMetadata,
                         remote_tp_rank: int = 0,
                         remote_tp_size: int = 1) -> str:
        """
        Add the remote MoRIIO agent and prepare the descriptors for reading cache
        blocks from remote.

        In particular, handle both homogeneous and heterogeneous TP. The former
        requires local rank_i to read from remote rank_i. 
        The latter, assuming D.world_size > P.world_size, requires that two or 
        more local TP worker share the xfer from a single TP worker.

        Here's an example:

        rank_offset     p_remote_tp_rank
        (kv split no)    
        --------------------------------
            0                 0      Worker0  ---- 1st half of KV ----> Worker0  [ KV Cache ]
                                                                        /
            1                 0      Worker1  ---- 2nd half of KV -----/

            0                 1      Worker2  ---- 1st half of KV ----> Worker1  [ KV Cache ]
                                                                        /
            1                 1      Worker3  ---- 2nd half of KV -----/


                                Decoder TP workers                     Prefix TP workers
                                  (world_size=4)                         (world_size=2)
                                                 tp_ratio = 4 // 2 = 2                  
                                
        Considering the KV Caches, if P-Worker_i has cache size [2, num_blocksP, kv_heads, block_size, head_dim]  
        then D-Worker_j has [2, num_blocksD, kv_heads//tp_ratio, block_size, head_dim]. Mind the "HND" layout format.
        Assuming num_blocksD >= num_blocksP, D-Worker0 reads from P-Worker0 by preparing the kv_heads//tp_ratio 
        first heads from all the slots of all the blocks. D-Worker1 will do the same, but reading the second split
        along the kv_heads dimension, and so forth until "tp_ratio" D TP workers have pulled from P-Worker0.   
        
        Note that the above will also hold true for the homogeneous TP case, where tp_ratio evaluates to 1.

        Regarding MLA case, the cache is replicated across TP workers so the rank_offset will just always be 0
        so that the whole cache is shared by "tp_ratio" D TP workers.
        """ # noqa: E501
        engine_id = nixl_agent_meta.engine_id
        # TODO re-evaluate refreshing for scaling/recovery
        if remote_tp_rank in self._remote_agents.get(engine_id, {}):
            return self._remote_agents[engine_id][remote_tp_rank]

        if engine_id not in self._tp_size:
            self._tp_size[engine_id] = remote_tp_size
        else:
            assert self._tp_size[engine_id] == remote_tp_size
        # We may eventually enable this after asserting equality in cache
        # layout and close outputs.
        assert nixl_agent_meta.attn_backend_name == self.backend_name

        remote_agent_name = "test"

        '''
        remote_agent_name = self.moriio_wrapper.add_remote_agent(
            nixl_agent_meta.agent_metadata)

        # Number of D TP workers reading from a single P TP worker. This is
        # 1 when P and D `--tensor-parallel-size` match.
        tp_ratio = divide(self._tp_size[self.engine_id],
                          self._tp_size[engine_id])
        assert tp_ratio > 0, "Decode TP cannot be smaller than prefill TP"

        # Handle tp_size>num_kv_heads: replicate KV cache.
        total_num_kv_heads = self.model_config.get_total_num_kv_heads()
        is_kv_replicated = self._tp_size[engine_id] // total_num_kv_heads >= 1

        if self.use_mla or is_kv_replicated:
            # With MLA the only difference is in the number of blocks.
            remote_block_size = nixl_agent_meta.block_len // (
                self.slot_size_bytes)
            assert self.block_len == nixl_agent_meta.block_len
        else:
            remote_block_size = nixl_agent_meta.block_len // (
                self.slot_size_bytes * tp_ratio)
            if self._use_flashinfer:
                # Account for joint KV in FlashInfer.
                remote_block_size //= 2

            assert nixl_agent_meta.block_len == self.block_len * tp_ratio, (
                "Remote P worker KV layer cache must be of shape [2, N, "
                "local_kv_heads*tp_ratio, block_size, head_dim] and same dtype."
            )

        assert self.block_size == remote_block_size, (
            "Remote P worker with different block size is not supported "
            f"{self.block_size=} {remote_block_size=}")

        # Create dst descs and xfer side handles. TP workers have same #blocks.
        if engine_id in self.dst_num_blocks:
            assert self.dst_num_blocks[engine_id] == nixl_agent_meta.num_blocks
        else:
            self.dst_num_blocks[engine_id] = nixl_agent_meta.num_blocks

        blocks_data = []
        # With homogeneous TP, D pulls the whole kv cache from corresponding
        # rank. With heterogeneous TP, prepare the descriptors by splitting the
        # P KV cache along kv_head dim, of D worker's kv_head size (D>P).
        # Eg. PTP1 DTP2 => P0 KV:[block0-KV_0 | block0-KV_1..].
        # Only register the remote's descriptors if current rank pulls from it.
        self.kv_caches_base_addr[
            engine_id] = nixl_agent_meta.kv_caches_base_addr
        rank_offset = self.tp_rank % tp_ratio * self.block_len \
            if not (self.use_mla or is_kv_replicated) else 0
        # Register all remote blocks, but only the corresponding kv heads.
        for base_addr in nixl_agent_meta.kv_caches_base_addr:
            for block_id in range(nixl_agent_meta.num_blocks):
                block_offset = block_id * nixl_agent_meta.block_len
                # For each block, grab the heads chunk belonging to rank_i
                # of size remote_nheads // tp_ratio, which correspond to
                # self.block_len == remote_block_len//tp_ratio bytes.
                addr = base_addr + block_offset + rank_offset
                # (addr, len, device id)
                blocks_data.append((addr, self.block_len, remote_tp_rank))
        logger.debug(
            "Created %s blocks for dst engine %s with remote rank %s and "
            "local rank %s", len(blocks_data), engine_id, remote_tp_rank,
            self.tp_rank)

        # Register with MoRIIO.
        descs = self.moriio_wrapper.get_xfer_descs(blocks_data, "VRAM")
        self.dst_xfer_side_handles[
            engine_id] = self.moriio_wrapper.prep_xfer_dlist(
                remote_agent_name, descs)
        '''

        return remote_agent_name

    def get_finished(self) -> tuple[set[str], set[str]]:
        """
        Get requests that are done sending or recving on this specific worker.
        The scheduler process (via the MultiprocExecutor) will use this output
        to track which workers are done.
        """
        # done_sending = self._get_new_notifs()
        # done_recving = self._pop_done_transfers(self._recving_transfers)
        # if len(done_sending) > 0 or len(done_recving) > 0:
        #     logger.debug(
        #         "Rank %s, get_finished: %s requests done sending "
        #         "and %s requests done recving", self.tp_rank,
        #         len(done_sending), len(done_recving))

        # # Handle timeout to avoid stranding blocks on remote.
        # now = time.perf_counter()
        # while self._reqs_to_send:
        #     req_id, expires = next(iter(self._reqs_to_send.items()))
        #     # Sorted dict, oldest requests are put first so we can exit early.
        #     if now < expires:
        #         break
        #     del self._reqs_to_send[req_id]
        #     done_sending.add(req_id)
        done_sending, done_recving = set(), set()
        # done_recving = set()
        # done_sending = set(self.done_sending_reqs)
        # # since python<=3.13 has GIL,so now I just ignore multithread safty
        # for val in done_sending:
        #     done_sending.remove(val)
        
        if self.is_producer:
            # logger.info(f"zovog:======> call get_finished,my role = P")
            done_sending = self.moriio_wrapper.pop_finished_req_ids()
            if GLOBAL_MORIIO_MODE==MoRIIOMode.WRITE:
            #     #need to recv block id from the remote 
            #     done_recving = self.moriio_wrapper.pop_remote_allocate_req_dict() #get_block
            # else:
                done_recving = set()
            # logger.info(f"zovog:======> call get_finished,my role = P done_sending = {done_sending}")
        else:
            if GLOBAL_MORIIO_MODE==MoRIIOMode.WRITE:
                self.moriio_wrapper.async_wait_reqid()
            # logger.info(f"zovog:======> call get_finished,my role = D")
            done_sending, done_recving = set(), self.moriio_wrapper.pop_finished_write_req_ids()
        if len(done_recving)!=0:
            p=0
            print_cur_time("finish"+str(self.finished_int)+"   ")
        # torch.distributed.barrier()
        self.finished_int+=1
        return done_sending, done_recving

    def _get_new_notifs(self) -> set[str]:
        """
        Get req_ids which got a remote xfer message. When multiple consumers
        are reading from the same producer (heterogeneous TP scenario), wait
        for all consumers to be done pulling.
        """
        pass

        # notified_req_ids: set[str] = set()
        # for notifs in self.moriio_wrapper.get_new_notifs().values():
        #     for notif in notifs:
        #         req_id, tp_ratio = notif.decode("utf-8").rsplit(":", 1)
        #         self.consumer_notification_counts_by_req[req_id] += 1
        #         # Wait all consumers (D) to be done reading before freeing.
        #         if self.consumer_notification_counts_by_req[req_id] == int(
        #                 tp_ratio):
        #             notified_req_ids.add(req_id)
        #             del self.consumer_notification_counts_by_req[req_id]
        #             del self._reqs_to_send[req_id]
        # return notified_req_ids

    # def _pop_done_transfers(
    #         self, transfers: dict[str, list[tuple[int, float]]]) -> set[str]:
    def _pop_done_transfers(self, done_req_ids) -> set[str]:
        """
        传输完之后,需要发送传输回执至P节点,
        """
        # done_req_ids: set[str] = set()

        '''
        for req_id, handles in list(transfers.items()):
            in_progress = False
            for handle, _xfer_stime in handles:
                xfer_state = self.moriio_wrapper.check_xfer_state(handle)
                if xfer_state == "DONE":
                    self.moriio_wrapper.release_xfer_handle(handle)
                elif xfer_state == "PROC":
                    in_progress = True
                    continue
                else:
                    raise RuntimeError("Transfer failed with state %s",
                                       xfer_state)
            if not in_progress:
                done_req_ids.add(req_id)
                del transfers[req_id]
        '''
        return done_req_ids
    
    
    def save_kv_layer(self, metadata: MoRIIOConnectorMetadata,layer_name: str, kv_layer: torch.Tensor,
                      attn_metadata: "AttentionMetadata", **kwargs):
        # logger.info(f"kuqi{layer_name = }")

        if not self.is_producer:
            return 

        # print(f"mama {layer_name} save kv")
        # for
        # pass
        # logger.info(f"apaci{layer_name = }")
        for req_id, meta in metadata.reqs_to_save.items():
            # logger.info(f"log:======> enter save kv for loop,{meta.remote_host = },{meta.remote_port = },{meta.local_block_ids = },{meta.remote_block_ids = },{meta.remote_engine_id = }")
            remote_engine_id = meta.remote_engine_id
            # logger.debug(
            #     "start_save_kv for request %s from remote engine %s. "
            #     "Num local_block_ids: %s. Num remote_block_ids: %s. ", req_id,
            #     remote_engine_id, len(meta.local_block_ids),
            #     len(meta.remote_block_ids))
            # TODO: mz get_remote_engine_id() for engine_id mapping.
            # if remote_engine_id is  None:
            #     remote_engine_id="1999"
    
            if remote_engine_id not in self._remote_agents:
                # Initiate handshake with remote engine to exchange metadata.
                with self._handshake_lock:
                    if remote_engine_id not in self._remote_agents:
                        logger.info(f"*****background nixl {remote_engine_id = }")
                        self._background_nixl_handshake(req_id, remote_engine_id, meta   )
                      
                        # logger.info(f"zovlog:==============> _background_nixl_handshake launched!")
                        # time.sleep(30)
                        continue
            # logger.info(f"log:======> remote agent {remote_engine_id} available, calling _write_blocks for req {req_id}")    
            # Handshake already completed, start async read xfer.
            # logger.info(f"sisi {layer_name = }")

            self._write_blocks_for_req(req_id, meta, layer_name,kv_layer)
            
      
        while True:
            if self._ready_requests.empty() and not self.write_kv_flag: # 第一次进入,需要一直等待
                # logger.info(f"zovlog:==============> {self._ready_requests.empty() = }")
                # pass
                continue
                # return 
            elif not self._ready_requests.empty() and self.write_kv_flag:
                # logger.info(f"zovlog:==============> {self._ready_requests.empty() = }")
                self._write_blocks_for_req(*self._ready_requests.get_nowait(),layer_name,kv_layer)
                break
            else:
                break

            pass
    
    def start_load_kv(self, metadata: MoRIIOConnectorMetadata):
        """
        Start loading by triggering non-blocking nixl_xfer.
        We check for these trnxs to complete in each step().
        """
        # print("start load kv")
        if self.is_producer:
            self.moriio_wrapper.async_wait_reqid()
            return
        # time.sleep(5)
        # logger.info(f"zovlog:======> start load kv,{metadata.reqs_to_recv.items() = }")
        wait_handshage_readd_req=False
        for req_id, meta in metadata.reqs_to_recv.items():
            # logger.info(f"zovlog:======> enter load kv for loop,{meta.remote_host = },{meta.remote_port = },{meta.local_block_ids = },{meta.remote_block_ids = },{meta.remote_engine_id = }")
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
                        self._background_nixl_handshake(req_id, remote_engine_id, meta)
                        # logger.info(f"zovlog:==============> _background_nixl_handshake launched!")
                        wait_handshage_readd_req=True

                        continue
                        
            # Handshake already completed, start async read xfer.
            self._read_blocks_for_req(req_id, meta)
        # Start transfers for requests whose handshakes have now finished.

        # if GLOBAL_MORIIO_MODE==MoRIIOMode.READ:
        
        while True: #TODO
            if self._ready_requests.empty() and not self.load_kv_flag and wait_handshage_readd_req: # 第一次进入,需要一直等待
                # logger.info(f"zovlog:==============> {self._ready_requests.empty() = }")
                continue 
            elif not self._ready_requests.empty() and self.load_kv_flag:
                # logger.info(f"zovlog:==============> {self._ready_requests.empty() = }")
                self._read_blocks_for_req(*self._ready_requests.get_nowait())
                break
            else:
                break

        # while not self._ready_requests.empty():
        #     self._read_blocks_for_req(*self._ready_requests.get_nowait())

        # Add to requests that are waiting to be read and track expiration.
        self._reqs_to_send.update(metadata.reqs_to_send)
        # if GLOBAL_MORIIO_MODE==MoRIIOMode.READ:
        #TODO 现在还是需要发送， 理论上只有read需要
        # torch.distributed.barrier(get_tp_group().device_group)
        for req_id, _ in metadata.reqs_to_recv.items():    
            self.moriio_wrapper.send_notify(req_id,_)

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
    def _write_blocks_for_req(self, req_id: str, meta: ReqMeta,layer_name,kv_layer):
        self._write_blocks(
            request_id=req_id,
            dst_engine_id=meta.remote_engine_id,
            local_block_ids=meta.local_block_ids,
            remote_block_ids=meta.remote_block_ids,
            layer_name=layer_name,
            kv_layer=kv_layer
        )
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
             
    def get_hash(self,n,local_block_ids):
        return self.kv_caches[list(self.kv_caches.keys())[n]][:,local_block_ids,:,:,:].sum()
    def get_all_hash(self,local_block_ids):
        hash_list = []
        for n in range(len(self.kv_caches)):
            hash_list.append(self.get_hash(n,local_block_ids).item())
    def _write_blocks(self, 
                     local_block_ids: list[int],
                     remote_block_ids: list[int], 
                     dst_engine_id: str,
                     request_id: str,
                     layer_name: str,
                     kv_layer: torch.Tensor):
        # pass
        # TODO  self._handshake_futures[eid]
        whilewait_time=None
        if self._is_first_layer(layer_name):
            whilewait_time=time.perf_counter()
        while True:
            if request_id in self.moriio_wrapper.done_remote_allocate_req:
                remote_block_ids = self.moriio_wrapper.done_remote_allocate_req_dict[request_id]
                # self.moriio_wrapper.done_remote_allocate_req.remove(request_id)
                # self.moriio_wrapper.done_remote_allocate_req_dict.pop(request_id)
                # TODO, finished了在pop
                break
            else:
                b=0
        if self._is_first_layer(layer_name):
            whilewait_time2=time.perf_counter()
            logger.info(f"!!!!!!wait remote allocate time {whilewait_time2-whilewait_time}")
        if GLOBAL_MORIIO_MODE==MoRIIOMode.READ:
            return
        layerwise=True
        logger.info(f"mymy {local_block_ids =},{remote_block_ids = }")
        
        use_batch=True
        if not self.builded_write_session:
            for layer_namekk,local_kv_cache_metadata in self.layer_name_to_local_kv_cache_metadata.items():
                stride = self.kv_caches[layer_namekk].stride()
                # logger.info(f"mapping {layer_name} local memory {local_kv_cache_metadata[0]}, remote memory {self.layer_name_to_remote_kv_cache_metadata[layer_name][0]}")

                self.moriio_wrapper.set_local_memory_metadata(local_kv_cache_metadata[0])
                self.moriio_wrapper.set_remote_memory_metadata(self.layer_name_to_remote_kv_cache_metadata[layer_namekk][0])
                self.moriio_wrapper.build_session()
            self.builded_write_session=True
        # logger.info(f"coco {layer_name = }")

        # layername_0=list(self.layer_name_to_local_kv_cache_metadata.items())[0][0]
        # layername_5=list(self.layer_name_to_local_kv_cache_metadata.items())[5][0]
        # logger.info(f"!!)){layer_name= },  tensor:{layername_0=}:{self.kv_caches[layername_0].sum() = }")
        # logger.info(f"!!)){layer_name= },  tensor:{layername_5=}:{self.kv_caches[layername_5].sum() = }")
        
        # ...existing code...
        ###################################
        # important debug code
        self.debug_cache.append((layer_name, (self.kv_caches[layer_name][:,local_block_ids[0],:,:,:].sum())))
# # ...existing code...
        if (len(self.debug_cache)-26)%27==0:
#         if (len(self.debug_cache)-62)%63==0:
            cccccc=0
        ###################################

        if layerwise:
            _,blknum,blksize,hn,hs = self.kv_cache_shape
            sess_idx = list(self.layer_name_to_local_kv_cache_metadata.keys()).index(layer_name)
            
            local_kv_cache_metadata=self.layer_name_to_local_kv_cache_metadata[layer_name]
            stride = self.kv_caches[layer_name].stride()
            offset_local=[]
            offset_remote=[]
            transfer_sizes=[]
            sz=self.kv_caches[layer_name].element_size()
            transfer_size_byte=blksize * hn * hs * sz
            # remote_block_ids=local_block_ids

            #TODO GPT-oss etc case
            if self._is_first_layer(layer_name):
                for idx,local_blkid in enumerate(local_block_ids):
                    offset_k_local = sz * (0 * stride[0] + local_blkid * stride[1])
                    offset_v_local = sz* (1 * stride[0] + local_blkid * stride[1])
                    offset_k_remote = sz * (0 * stride[0] + remote_block_ids[idx] * stride[1])
                    offset_v_remote = sz * (1 * stride[0] + remote_block_ids[idx] * stride[1])
                    offset_local.append(offset_v_local)
                    offset_remote.append(offset_v_remote)
                    transfer_sizes.append(transfer_size_byte)

                
                    offset_local.append(offset_k_local)
                    offset_remote.append(offset_k_remote)
                    transfer_sizes.append(transfer_size_byte)
            

                    if not use_batch:
                        pass
                    
                    #[1,2], [2,5].
                        # self.moriio_wrapper.read_remote_data_s(transfer_size_byte,offset_v_local,offset_v_remote,sess_idx)
                        # self.moriio_wrapper.read_remote_data_s(transfer_size_byte,offset_k_local,offset_k_remote,sess_idx)
                        print("!!!!",transfer_size_byte,offset_k_local,offset_k_remote,sess_idx)
                        print("!!!!",transfer_size_byte,offset_v_local,offset_v_remote,sess_idx)
                t1=time.perf_counter()
                # tmp1,tmp2,tmp3=self.merge_contiguous_blocks(offset_local,offset_remote,transfer_sizes)
                t2=time.perf_counter()

                # tmp1,tmp2,tmp3=self.merge_contiguous_blocks_fast(offset_local,offset_remote,transfer_sizes)
                t3=time.perf_counter()
                # self.merged_local, self.merged_remote, self.merged_sizes=self.merge_contiguous_blocks_fast_v2(offset_local,offset_remote,transfer_sizes)
                self.merged_local, self.merged_remote, self.merged_sizes=self.merge_contiguous_blocks(offset_local,offset_remote,transfer_sizes)

                t4=time.perf_counter()
                logger.info(f"merge time v2 {t4-t3}, old {t2-t1}")
                # assert (tmp1==self.merged_local)
                # assert (tmp2==self.merged_remote)
                # assert (tmp3==self.merged_sizes)
                # assert (k1==self.merged_local)
                # assert (k2==self.merged_remote)
                # assert (k3==self.merged_sizes)
            a,b,c=self.this_layer_write_meta_offset()
            # if self.tp_rank==0:
            #     qqq=0
            if use_batch:
                # self.moriio_wrapper.read_remote_data(transfer_sizes,offset_local, offset_remote,sess_idx)
                
                # print(f"!!!!len(buffer){len(c)}")
                # for ii in range(len(c)):
                #     print(c[ii]/1024)
                # time.sleep(0.2)
                # self.moriio_wrapper.waiting_for_read_complete()
                # logger.info(f"{sess_idx =} +{str(c)}+{str(a)}+{str(b)}")
                self.moriio_wrapper.write_remote_data(c,a, b,sess_idx)
                # self.moriio_wrapper.waiting_for_read_complete()

                # self.moriio_wrapper.waiting_for_read_complete()
                # time.sleep(0.2)
            else:
                self.moriio_wrapper.waiting_for_read_complete()

                for rang_idx in range(len(a)):
                    # time.sleep(/sz.1)
                    # print("bbbb",c[rang_idx],a[rang_idx],b[rang_idx],sess_idx)
                    self.moriio_wrapper.write_remote_data_s(c[rang_idx],a[rang_idx],b[rang_idx],sess_idx)
                    # self.moriio_wrapper.waiting_for_read_complete()

            if self._is_last_layer(layer_name):
                    # time.sleep(0.1)

                self.moriio_wrapper.waiting_for_read_complete()
                
                # self.moriio_wrapper.done_req_ids.append(request_id)

                logger.info(f"send notify to D")
                # torch.distributed.barrier(get_tp_group().device_group)
                # if self.tp_rank==0:
                #     ppp=0
                self.moriio_wrapper.send_notify(request_id)
                logger.info(f"send notify to D end")
              
                    
                print_cur_time("!!!!last layer write time ")
        elif not layerwise:
        
            
            start = time.perf_counter()
           
            if not self._is_last_layer(layer_name):
                return
            _,blknum,blksize,hn,hs = self.kv_cache_shape
            # stride = [blknum*blksize*hn*hs   ,blksize*hs*hn   ,hs*hn   ,hs   ,1]
            sess_idx=0
            al=[]
            bl=[]
            cl=[]
            sl=[]
            for layer_name,local_kv_cache_metadata in self.layer_name_to_local_kv_cache_metadata.items():
                
                # logger.error(f"zovlog:--------> {layer_name = },{local_kv_cache_metadata[0] = },{len(local_kv_cache_metadata) = },{self.kv_caches[layer_name].shape = },{self.kv_caches[layer_name].stride() = }")
                stride = self.kv_caches[layer_name].stride()
                # self.moriio_wrapper.set_local_memory_metadata(local_kv_cache_metadata[0])
                # self.moriio_wrapper.set_remote_memory_metadata(self.layer_name_to_remote_kv_cache_metadata[layer_name][0])
                # 在local_block_ids这个序列中,判断一下那些是连续的
                # for start,end in zip(contiguous_ids[])
                #todo make batch_read
                offset_local=[]
                offset_remote=[]
                transfer_sizes=[]
                sess_id=[]
                sz=self.kv_caches[layer_name].element_size()
                transfer_size_byte=blksize * hn * hs * sz
                # TODO, assume remote_block_id = local_block_id for only 1P1D debug
                remote_block_ids=local_block_ids
                for idx,local_blkid in enumerate(local_block_ids):
                    offset_k_local = sz * (0 * stride[0] + local_blkid * stride[1])
                    offset_v_local = sz* (1 * stride[0] + local_blkid * stride[1])
                    offset_k_remote = sz * (0 * stride[0] + remote_block_ids[idx] * stride[1])
                    offset_v_remote = sz * (1 * stride[0] + remote_block_ids[idx] * stride[1])
                    # transfer_size_byte = blksize * hn * hs * sz
                    # logger.info(f"zovlog:===========>{self.kv_cache_shape = },{layer_name = },{offset_k = },{offset_v = },{transfer_size_byte = },{blkid = },{stride = }")
                    
                    
                    offset_local.append(offset_v_local)
                    offset_remote.append(offset_v_remote)
                    transfer_sizes.append(transfer_size_byte)

                
                    offset_local.append(offset_k_local)
                    offset_remote.append(offset_k_remote)
                    transfer_sizes.append(transfer_size_byte)
            

                    if not use_batch:
                        pass
                    
                    #[1,2], [2,5].
                        # self.moriio_wrapper.read_remote_data_s(transfer_size_byte,offset_v_local,offset_v_remote,sess_idx)
                        # self.moriio_wrapper.read_remote_data_s(transfer_size_byte,offset_k_local,offset_k_remote,sess_idx)
                        print("!!!!",transfer_size_byte,offset_k_local,offset_k_remote,sess_idx)
                        print("!!!!",transfer_size_byte,offset_v_local,offset_v_remote,sess_idx)
                a,b,c=self.merge_contiguous_blocks(offset_local,offset_remote,transfer_sizes)
                
                if use_batch:
                    # self.moriio_wrapper.read_remote_data(transfer_sizes,offset_local, offset_remote,sess_idx)
                    
                    # print(f"!!!!len(buffer){len(c)}")
                    # for ii in range(len(c)):
                    #     print(c[ii]/1024)
                    pass
                    # self.moriio_wrapper.read_remote_data(c,a, b,sess_idx)

                else:
                    for rang_idx in range(len(a)):
                        # print("bbbb",c[rang_idx],a[rang_idx],b[rang_idx],sess_idx)
                        self.moriio_wrapper.write_remote_data_s(c[rang_idx],a[rang_idx],b[rang_idx],sess_idx)
                al.append(a)
                bl.append(b)
                cl.append(c)
                sl.append(sess_idx)
            # time.sleep(15)

            # 结束计时
            end = time.perf_counter()

            # 计算耗时
            print(f"耗时：{end - start:.4f} 秒")
            time.sleep(3)

            for inb in range(len(al)):
                self.moriio_wrapper.write_remote_data(cl[inb],al[inb],bl[inb],sl[inb])
            self.moriio_wrapper.waiting_for_read_complete()
            end2=time.perf_counter()
            time.sleep(3)
            print(f"纯传输耗时：{end2 - end:.4f} 秒")

            return
        
        
        # pass
        
    
    
    

    def merge_contiguous_blocks_fast_v2(self,offsets_local: List[int],offsets_remote: List[int],sizes: List[int],assume_sorted: bool = False) -> Tuple[List[int], List[int], List[int]]:
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
            # 检测已排序避免 argsort
            if np.all(local_arr[:-1] <= local_arr[1:]):
                local_sorted = local_arr
                remote_sorted = remote_arr
                sizes_sorted = sizes_arr
            else:
                sort_idx = np.argsort(local_arr, kind="stable")
                local_sorted = local_arr[sort_idx]
                remote_sorted = remote_arr[sort_idx]
                sizes_sorted = sizes_arr[sort_idx]

        # 差分判定连续 (比构造 local_ends / 逐元素加法更省)
        # 若 diff_local == prev_size 且 diff_remote == prev_size => 连续
        if n == 1:
            return [int(local_sorted[0])], [int(remote_sorted[0])], [int(sizes_sorted[0])]

        diff_local = local_sorted[1:] - local_sorted[:-1]
        diff_remote = remote_sorted[1:] - remote_sorted[:-1]
        prev_size = sizes_sorted[:-1]

        contiguous = (diff_local == prev_size) & (diff_remote == prev_size)

        # Fast path: 没有任何可合并
        if not contiguous.any():
            return local_sorted.tolist(), remote_sorted.tolist(), sizes_sorted.tolist()

        # Fast path: 全部连续 -> 单区间
        if contiguous.all():
            total_size = int(sizes_sorted.sum())
            return [int(local_sorted[0])], [int(remote_sorted[0])], [total_size]

        # 标记断点: contiguous=False 的位置断开
        # 断点起始包含 index 0
        break_positions = np.flatnonzero(~contiguous) + 1  # 下一个片段的开始
        # 加入首尾
        segment_starts = np.concatenate(([0], break_positions))
        segment_ends = np.concatenate((break_positions, [n]))

        seg_count = len(segment_starts)
        merged_local = [0] * seg_count
        merged_remote = [0] * seg_count
        merged_sizes = [0] * seg_count

        # 逐段聚合
        for si in range(seg_count):
            s = segment_starts[si]
            e = segment_ends[si]
            merged_local[si] = int(local_sorted[s])
            merged_remote[si] = int(remote_sorted[s])
            # size = (最后一个块末尾) - (第一个块起始)
            # 末尾块末尾 = local_sorted[e-1] + sizes_sorted[e-1]
            merged_sizes[si] = int(local_sorted[e - 1] + sizes_sorted[e - 1] - local_sorted[s])

        return merged_local, merged_remote, merged_sizes
    def merge_contiguous_blocks(self, offsets_local, offsets_remote, sizes):
        """
        合并连续的存储区块
        """
        if not offsets_local:
            return [], [], []
        
        # 首先确保所有列表长度相同
        assert len(offsets_local) == len(offsets_remote) == len(sizes)
        
        # 按本地偏移量排序，同时保持关联关系
        sorted_indices = sorted(range(len(offsets_local)), key=lambda i: offsets_local[i])
        sorted_local = [offsets_local[i] for i in sorted_indices]
        sorted_remote = [offsets_remote[i] for i in sorted_indices]
        sorted_sizes = [sizes[i] for i in sorted_indices]
        
        merged_local = []
        merged_remote = [] 
        merged_sizes = []
        
        # 初始化第一个区间
        current_local_start = sorted_local[0]
        current_remote_start = sorted_remote[0]
        current_size = sorted_sizes[0]
        current_end = current_local_start + current_size
        
        for i in range(1, len(sorted_local)):
            local_offset = sorted_local[i]
            remote_offset = sorted_remote[i]
            size = sorted_sizes[i]
            
            # 检查是否连续（本地和远程都连续）
            is_contiguous = (local_offset == current_end and 
                            remote_offset == current_remote_start + current_size)
            
            if is_contiguous:
                # 扩展当前区间
                current_size += size
                current_end = local_offset + size
            else:
                # 保存当前区间
                merged_local.append(current_local_start)
                merged_remote.append(current_remote_start)
                merged_sizes.append(current_size)
                
                # 开始新区间
                current_local_start = local_offset
                current_remote_start = remote_offset
                current_size = size
                current_end = local_offset + size
        
        # 添加最后一个区间
        merged_local.append(current_local_start)
        merged_remote.append(current_remote_start)
        merged_sizes.append(current_size)
    
        return merged_local, merged_remote, merged_sizes

    def _read_blocks(self, 
                     local_block_ids: list[int],
                     remote_block_ids: list[int], 
                     dst_engine_id: str,
                     request_id: str):
        # logger.info(f"zovlog:========> start read blocks {local_block_ids = },{remote_block_ids = },{dst_engine_id = },{request_id = }")
        # return
        # 每一层的对应blkid都需要传输
        # if GLOBAL_MORIIO_MODE==MoRIIOMode.WRITE:
        #     return
        # if GLOBAL_MORIIO_MODE == MoRIIOMode.WRITE:
        #         return 
        start = time.perf_counter()

        
        
        # layername_0=list(self.layer_name_to_local_kv_cache_metadata.items())[0][0]
        # layername_5=list(self.layer_name_to_local_kv_cache_metadata.items())[5][0]
        # logger.info(f"!!))  tensor:{layername_0=}:{self.kv_caches[layername_0].sum() = }")
        # logger.info(f"!!))  tensor:{layername_5=}:{self.kv_caches[layername_5].sum() = }")
        if not self.builded_session:
            for layer_name,local_kv_cache_metadata in self.layer_name_to_local_kv_cache_metadata.items():
                # logger.info(f"session map:--------> {layer_name = },{local_kv_cache_metadata[0] = },{len(local_kv_cache_metadata) = },{self.kv_caches[layer_name].shape = },{self.kv_caches[layer_name].stride() = }")
                stride = self.kv_caches[layer_name].stride()
                self.moriio_wrapper.set_local_memory_metadata(local_kv_cache_metadata[0])
                # logger.info(f"mapping {layer_name} local memory {local_kv_cache_metadata[0]}, remote memory {self.layer_name_to_remote_kv_cache_metadata[layer_name][0]}")
                self.moriio_wrapper.set_remote_memory_metadata(self.layer_name_to_remote_kv_cache_metadata[layer_name][0])
                self.moriio_wrapper.build_session()
            self.builded_session=True
            # print("sleeping")
            # time.sleep(20)
       

        # self.kv_caches
        # contiguous_ids = search_contiguous_block_ids(local_block_ids,remote_block_ids)
        _,blknum,blksize,hn,hs = self.kv_cache_shape
        # stride = [blknum*blksize*hn*hs   ,blksize*hs*hn   ,hs*hn   ,hs   ,1]
        sess_idx=0
        use_batch=True
        al=[]
        bl=[]
        cl=[]
        sl=[] #26+27*n
        for layer_name,local_kv_cache_metadata in self.layer_name_to_local_kv_cache_metadata.items():
            self.debug_cache.append((layer_name, (self.kv_caches[layer_name][:,local_block_ids[0],:,:,:].sum())))
            if (len(self.debug_cache)-26)%27==0: 
            # if (len(self.debug_cache)-62)%63==0:

                cccccccc=0
                
            if GLOBAL_MORIIO_MODE == MoRIIOMode.WRITE:
                continue
            # use read mode to check
            for idx,local_blkid in enumerate(local_block_ids):
                assert(local_blkid==remote_block_ids[idx])
         
           
            # logger.error(f"zovlog:--------> {layer_name = },{local_kv_cache_metadata[0] = },{len(local_kv_cache_metadata) = },{self.kv_caches[layer_name].shape = },{self.kv_caches[layer_name].stride() = }")
            stride = self.kv_caches[layer_name].stride()
            # self.moriio_wrapper.set_local_memory_metadata(local_kv_cache_metadata[0])
            # self.moriio_wrapper.set_remote_memory_metadata(self.layer_name_to_remote_kv_cache_metadata[layer_name][0])
            # 在local_block_ids这个序列中,判断一下那些是连续的
            # for start,end in zip(contiguous_ids[])
            #todo make batch_read
            offset_local=[]
            offset_remote=[]
            transfer_sizes=[]
            sess_id=[]
            sz=self.kv_caches[layer_name].element_size()
            transfer_size_byte=blksize * hn * hs * sz
          
            for idx,local_blkid in enumerate(local_block_ids):
                offset_k_local = sz * (0 * stride[0] + local_blkid * stride[1])
                offset_v_local = sz* (1 * stride[0] + local_blkid * stride[1])
                offset_k_remote = sz * (0 * stride[0] + remote_block_ids[idx] * stride[1])
                offset_v_remote = sz * (1 * stride[0] + remote_block_ids[idx] * stride[1])
                assert(local_blkid==remote_block_ids[idx])
                # transfer_size_byte = blksize * hn * hs * sz
                # logger.info(f"zovlog:===========>{self.kv_cache_shape = },{layer_name = },{offset_k = },{offset_v = },{transfer_size_byte = },{blkid = },{stride = }")
                
                
                offset_local.append(offset_v_local)
                offset_remote.append(offset_v_remote)
                transfer_sizes.append(transfer_size_byte)

            
                offset_local.append(offset_k_local)
                offset_remote.append(offset_k_remote)
                transfer_sizes.append(transfer_size_byte)
        

                if not use_batch:
                    pass
                
                #[1,2], [2,5].
                     # self.moriio_wrapper.read_remote_data_s(transfer_size_byte,offset_v_local,offset_v_remote,sess_idx)
                    # self.moriio_wrapper.read_remote_data_s(transfer_size_byte,offset_k_local,offset_k_remote,sess_idx)
                    print("!!!!",transfer_size_byte,offset_k_local,offset_k_remote,sess_idx)
                    print("!!!!",transfer_size_byte,offset_v_local,offset_v_remote,sess_idx)
            a,b,c=self.merge_contiguous_blocks(offset_local,offset_remote,transfer_sizes)
            return 
            if use_batch:
                # self.moriio_wrapper.read_remote_data(transfer_sizes,offset_local, offset_remote,sess_idx)
                
                # print(f"!!!!len(buffer){len(c)}")
                # for ii in range(len(c)):
                #     print(c[ii]/1024)
                pass
                # self.moriio_wrapper.read_remote_data(c,a, b,sess_idx)

            else:
                for rang_idx in range(len(a)):
                    # print("bbbb",c[rang_idx],a[rang_idx],b[rang_idx],sess_idx)
                    self.moriio_wrapper.read_remote_data_s(c[rang_idx],a[rang_idx],b[rang_idx],sess_idx)
            al.append(a)
            bl.append(b)
            cl.append(c)
            sl.append(sess_idx)
            sess_idx+=1
        
        
        self.moriio_wrapper.waiting_for_read_complete()
        # time.sleep(15)

        # 结束计时
        end = time.perf_counter()

        # 计算耗时
        print(f"耗时：{end - start:.4f} 秒")

        for inb in range(len(al)):
            self.moriio_wrapper.read_remote_data(cl[inb],al[inb],bl[inb],sl[inb])
        self.moriio_wrapper.waiting_for_read_complete()
        end2=time.perf_counter()

        print(f"纯传输耗时：{end2 - end:.4f} 秒")

        return
        # NOTE(rob): having the staging blocks be on the READER side is
        # not going to work well (since we will have to call rearrange tensors).
        # after we detect the txn is complete (which means we cannot make the
        # read trxn async easily). If we want to make "READ" happen cleanly,
        # then we will need to have the staging blocks on the remote side.

        # NOTE(rob): according to nvidia the staging blocks are used to
        # saturate IB with heterogeneous TP sizes. We should remove the staging
        # blocks until we are ready.

        # Number of D TP workers that will read from dst P. Propagate tp_ratio
        # on notification so that dst worker can wait before freeing blocks.
        tp_ratio = self._tp_size[self.engine_id] // self._tp_size[dst_engine_id]
        notif_id = f"{request_id}:{tp_ratio}".encode()

        # Full prefix cache hit: do not need to read remote blocks,
        # just notify P worker that we have the blocks we need.
        num_local_blocks = len(local_block_ids)
        '''
        if num_local_blocks == 0:
            remote_rank = self.tp_rank // tp_ratio
            agent_name = self._remote_agents[dst_engine_id][remote_rank]
            self.moriio_wrapper.send_notif(agent_name, notif_msg=notif_id)
            return
        '''

        # Partial prefix cache hit: just read uncomputed blocks.
        num_remote_blocks = len(remote_block_ids)
        assert num_local_blocks <= num_remote_blocks
        if num_local_blocks < num_remote_blocks:
            remote_block_ids = remote_block_ids[-num_local_blocks:]

        # Get side handles.
        local_xfer_side_handle = self.src_xfer_side_handle
        remote_xfer_side_handle = self.dst_xfer_side_handles[dst_engine_id]

        # NOTE (nicolo) With homogeneous TP, each TP worker loads KV from
        # corresponding rank. With heterogeneous TP, fixing D>P, the D tp
        # workers will issue xfers to parts of the P worker remote kv caches.

        # Get descs ids.
        local_block_descs_ids: list[int] = []
        remote_block_descs_ids: list[int] = []
        if not self.block_window_per_layer:
            # Default case: assume global attention
            remote_block_descs_ids = self._get_block_descs_ids(dst_engine_id, remote_block_ids)
            local_block_descs_ids = self._get_block_descs_ids(self.engine_id, local_block_ids)
        else:
            # TODO(mgoin): remove this once we have hybrid memory allocator
            # Optimization for models with local attention (Llama 4)
            for layer_idx, block_window in enumerate(self.block_window_per_layer):
                # For each layer:
                if block_window is None:
                    # If not chunked, we just use the
                    # full block lists (global attention)
                    layer_local_block_ids = local_block_ids
                    layer_remote_block_ids = remote_block_ids
                else:
                    # If chunked, get the last block_window blocks
                    layer_local_block_ids = local_block_ids[-block_window:]
                    layer_remote_block_ids = remote_block_ids[-block_window:]

                # Get descs ids for the layer.
                layer_local_desc_ids = self._get_block_descs_ids(self.engine_id, layer_local_block_ids, layer_idx)
                layer_remote_desc_ids = self._get_block_descs_ids(dst_engine_id, layer_remote_block_ids, layer_idx)

                local_block_descs_ids.extend(layer_local_desc_ids)
                remote_block_descs_ids.extend(layer_remote_desc_ids)

        assert len(local_block_descs_ids) == len(remote_block_descs_ids)
        '''
        # Prepare transfer with Nixl.
        handle = self.moriio_wrapper.make_prepped_xfer(
            "READ",
            local_xfer_side_handle,
            local_block_descs_ids,
            remote_xfer_side_handle,
            remote_block_descs_ids,
            notif_msg=notif_id,
        )

        # Begin async xfer.
        self.moriio_wrapper.transfer(handle)
        

        # Use handle to check completion in future step().
        # TODO (NickLucche) surface xfer elapsed time
        self._recving_transfers[request_id].append(
            (handle, time.perf_counter()))
        '''

    def _get_block_descs_ids(self,
                             engine_id: str,
                             block_ids: list[int],
                             layer_idx: Optional[int] = None) -> list[int]:
        """
        Get the descs ids for a set of block ids.
        If layer_idx is provided, we use the region_ids for the given layer.
        Otherwise, we use all regions.
        """
        if layer_idx is None:
            region_ids = range(self.num_regions)
        else:
            assert layer_idx < self.num_layers
            if self.num_layers < self.num_regions:
                # If we have more regions than layers, we assume that
                # the regions are organized as [K0, V0, K1, V1, ...]
                # and we select K_i and V_i
                assert 2 * self.num_layers == self.num_regions
                region_ids = range(2 * layer_idx, 2 * layer_idx + 2)
            else:
                # Otherwise, we assume we have MLA and select i-th layer
                assert self.num_layers == self.num_regions
                region_ids = range(layer_idx, layer_idx + 1)

        num_blocks = self.dst_num_blocks[engine_id]

        # Compute the desc ids for each block.
        descs_ids: list[int] = []
        for reg_id in region_ids:
            for block_id in block_ids:
                descs_ids.append(reg_id * num_blocks + block_id)
        return descs_ids


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
