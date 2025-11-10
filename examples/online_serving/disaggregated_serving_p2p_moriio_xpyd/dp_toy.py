import argparse
import itertools
import logging
import os
import socket
import uuid
from contextlib import asynccontextmanager
import msgpack
import zmq
import copy
import threading
from quart import Quart, make_response, request
import httpx
import re
import json
from fastapi import FastAPI, Request
from fastapi.responses import StreamingResponse
from typing import Dict,List
import asyncio
logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)

import aiohttp
prefill_instances = []
decode_instances = [] 
request_nums = 0
app = Quart(__name__)

yield_chunk = set()
IP_PORT_PATTERN = re.compile(r'//(\d{1,3}\.\d{1,3}\.\d{1,3}\.\d{1,3}):(\d+)')
#re.search(r'//(\d{1,3}\.\d{1,3}\.\d{1,3}\.\d{1,3}):(\d+)', url).groups()

count=1
from itertools import count

# 使用无限计数器
counter = count(1)

def count_print(msg):
    current_count = next(counter)
    print(f"---mingzhilog[{current_count}] : {msg}")
def _append_whole_dict_unique(target_list, data_dict):
    new_filtered = {k: v for k, v in data_dict.items() if k != "index"}
    for existed in target_list:
        existed_filtered = {k: v for k, v in existed.items() if k != "index"}
        if existed_filtered == new_filtered:
            return False
    print("!!APPEND!!", data_dict)
    target_list.append(data_dict)
_list_lock = threading.RLock()

def _listen_for_register(hostname, port):
    context = zmq.Context()
    router_socket = context.socket(zmq.ROUTER)
    router_socket.bind(f"tcp://{hostname}:{port}")
    poller = zmq.Poller()
    poller.register(router_socket,zmq.POLLIN)
    global prefill_instances
    global decode_instances

    while True:
        socks = dict(poller.poll())
        if router_socket in socks:
            # c=router_socket.recv_multipart()
            # print("xxxx",c)
            # message_parts = router_socket.recv_multipart()
            # # 取第一个部分作为 remote_addr，最后一个部分作为 msg
            # remote_addr = message_parts[0]
            # msg = message_parts[1]
            remote_addr,msg = router_socket.recv_multipart()
            data = msgpack.loads(msg)
            if data['type'] == "HELLO":
                pass
            elif data['type'] == "register" and data['role'] == "P":
                if data['request_address'] not in prefill_instances:
                    # prefill_instances.append(data['request_address'])
                    with _list_lock:
                        _append_whole_dict_unique(prefill_instances, data)
                    # prefill_instances._append_whole_dict_unique(data)

            elif data["type"] == "register" and data['role'] == "D":
                if data['request_address'] not in decode_instances:
                    # decode_instances.append(data['request_address'])
                    with _list_lock:
                        _append_whole_dict_unique(decode_instances, data)
            # print(f"zovlog:====> recv {data},remote_addr={remote_addr},{prefill_instances = },{decode_instances = }")

def start_service_discovery(hostname, port):
    if not hostname:
        hostname = socket.gethostname()
    if port == 0:
        raise ValueError("Port cannot be 0")

    _listener_thread = threading.Thread(
        target = _listen_for_register,args = (hostname, port),daemon=True
    )
    _listener_thread.start()
    return _listener_thread

async def send_request_to_prefill(endpoint,req_data,request_id,p_endpoint,pip,pports):
    # print(f"zovlog:======> proxy {endpoint = }")
    req_data_copy =req_data
    
    # 本地做prefill,且decode只需要pull模式,所以prefill不需要在这里知晓远程decode任何信息
   
    req_data_copy['kv_transfer_params'] = {
        "do_remote_decode": True,
        "do_remote_prefill": False,
        "remote_handshake_port": p_endpoint['handshake_port'],
        "remote_notify_port":p_endpoint['notify_port'],
        "remote_engine_id": None,
        "remote_block_ids": None,
        "remote_host":pip ,
        "remote_port": pports,
    }
    req_data_copy["stream"] = False
    req_data_copy["max_tokens"] = 1
    if "max_completion_tokens" in req_data_copy:
        req_data_copy["max_completion_tokens"] = 1
    if "stream_options" in req_data_copy:
        del req_data_copy["stream_options"]
    # print(f"zovlog ========================== send response to prefill {req_data_copy}")
    async with aiohttp.ClientSession(timeout=aiohttp.ClientTimeout(total=6 * 6000 * 6000)) as session:
        headers = {
            "Authorization": f"Bearer {os.environ.get('OPENAI_API_KEY')}",
            "X-Request-Id": request_id
        }
        async with session.post(url=endpoint, json=req_data_copy, headers=headers) as response:
            if response.status == 200:
                return await response.json()
                # async for chunk_bytes in response.content.iter_chunked(1024):
                #         yield chunk_bytes
            else:
                raise RuntimeError("send_request_to_prefill response.status != 200,response.statuus = ",response.status)
async def start_decode_request(endpoint, req_data, request_id):
    """立即启动请求，返回响应对象"""
    session = aiohttp.ClientSession(timeout=aiohttp.ClientTimeout(total=6 * 6000 * 6000))
    headers = {
        "Authorization": f"Bearer {os.environ.get('OPENAI_API_KEY')}",
        "X-Request-Id": request_id
    }
    response = await session.post(url=endpoint, json=req_data, headers=headers)
    return session, response

async def stream_decode_response(session, response, request_id):
    """流式处理响应"""
    try:
        if response.status == 200:
            async for chunk_bytes in response.content.iter_chunked(1024):
                # if request_id not in yield_chunk:
                #     yield_chunk.add(request_id)
                # else:
                #     logger.info("!!!pass yidle")
                yield chunk_bytes
        else:
            raise RuntimeError(f"decode response.status != 200, status = {response.status}")
    finally:
        await session.close()
# to debug
async def send_request_to_decode(endpoint,req_data,request_id):
    # print(f"zovlog ========================== send response to decode {request_id}")
    async with aiohttp.ClientSession(timeout=aiohttp.ClientTimeout(total=6 * 6000 * 6000)) as session:
        headers = {
            "Authorization": f"Bearer {os.environ.get('OPENAI_API_KEY')}",
            "X-Request-Id": request_id
        }
        async with session.post(url=endpoint, json=req_data, headers=headers) as response:
            if response.status == 200:
                async for chunk_bytes in response.content.iter_chunked(1024):
                        # if request_id not in yield_chunk:
                        #     # count_print(f"zovlog yield chunk for {request_id}") #128 
                        #     yield_chunk.add(request_id)
                        #     # b=0
                        #     # print(f"!!!{chunk_bytes.decode('utf-8')}")
                        #     # try:
                        #     #     print("!!!xxx")
                        #     #     print(json.loads(chunk_bytes.decode('utf-8'))['choices'][0]['text'])
                        #     # except Exception as e:
                        #     #     print(f"no text: {e}")
                        # else:
                        #     logger.info("!!!pass yidle")
                            # print("pass yidle")
                        yield chunk_bytes
            else:
                raise RuntimeError("send_request_to_decode response.status != 200,response.statuus = ",response.status)

#user->proxy->prefill->proxy ->decode
@app.route("/v1/completions", methods=["POST"])
@app.route("/v1/chat/completions", methods=["POST"])
async def handle_request():
    # print(f"zovlog:-----------> enter request")
    try:
        import time
        
        # st1=time.perf_counter()
        global request_nums
        # extract_ip_port = lambda url: re.search(r'//(\d{1,3}\.\d{1,3}\.\d{1,3}\.\d{1,3}):(\d+)', url).groups()
        def extract_ip_port_fast(url):
            return IP_PORT_PATTERN.search(url).groups()
        req_data = await request.get_json()
        # st1p5=time.perf_counter()
        request_id = str(uuid.uuid4())

        # print(f"req_data = {req_data}")
        prefill_instance_endpoint=None
        decode_instance_endpoint=None
        # print(f"zovlog:-----------> before select instance {prefill_instances=
        # if False:
        # =0
        dp_rank=request_nums % 8
        # dp_rank=0
        req_data['data_parallel_rank'] = dp_rank
        if len(prefill_instances)==2 and len(decode_instances)==2:
            index_list=[[0,0],[1,0],[0,1],[1,1]]
            index=index_list[request_nums % len(index_list)]
            prefill_instance_endpoint = prefill_instances[index[0]]
            decode_instance_endpoint = decode_instances[index[1]]
            # print(f"P:{index[0]},D:{index[1]}")
        
        else:
            # assert False, f"prefill_instances or decode_instances not ready,"
            pid=request_nums % len(prefill_instances)
            did=request_nums % len(decode_instances)
            prefill_instance_endpoint = prefill_instances[pid]
            decode_instance_endpoint = decode_instances[did]
            # print(f"P:{pid},D:{did}")

        # print(f"{prefill_instances=},{decode_instances=}")
        # print(f"******{request_id}******,******{prefill_instance_endpoint=}, {decode_instance_endpoint=}, {request_nums=}")
        dip,dport= extract_ip_port_fast(decode_instance_endpoint['request_address'])
        # preq_data = copy.deepcopy(req_data)
        ip, port = extract_ip_port_fast(prefill_instance_endpoint['request_address'])
        # response_json['kv_transfer_params']["do_remote_decode"] = False
        # response_json['kv_transfer_params']["do_remote_prefill"] = True
        # response_json['kv_transfer_params']["remote_host"] = ip
        # response_json['kv_transfer_params']["remote_port"] = port # 似乎没用
        # response_json['kv_transfer_params']["remote_handshake_port"] = prefill_instance_endpoint['handshake_port']

    


       
        # decode_task= asyncio.create_task(  send_request_to_decode(decode_instance_endpoint['request_address'],req_data,request_id))
        req_data_to_prefill = copy.deepcopy(req_data)
        # print("send to prefill req_data:",req_data_to_prefill)
        send_prefill_task = asyncio.create_task(send_request_to_prefill(prefill_instance_endpoint['request_address'],req_data_to_prefill,request_id,decode_instance_endpoint,dip,dport))
        # 现在decode可以获取prefill的所有信息了
        ip, port = extract_ip_port_fast(prefill_instance_endpoint['request_address'])
        
        

        
        req_data['max_tokens'] -= 1
        req_data['data_parallel_rank'] = dp_rank
        req_data['kv_transfer_params'] = {
            "do_remote_decode": False,
            "do_remote_prefill": True,
            "remote_handshake_port": prefill_instance_endpoint['handshake_port'],
            "remote_notify_port":prefill_instance_endpoint['notify_port'],
            "remote_engine_id": None,
            "remote_block_ids": None,
            "remote_host":ip ,
            "remote_port": port,
        }
        if 'data_parallel_rank' in req_data:
            req_data['kv_transfer_params']['remote_dp_rank'] = req_data['data_parallel_rank']
            del req_data['data_parallel_rank']
        # print("send to decode req_data:",req_data)
        decode_request_task = asyncio.create_task(
            start_decode_request(decode_instance_endpoint['request_address'], req_data, request_id)
        )
       

        # (session, decode_response), prefill_result = await asyncio.gather(decode_request_task, send_prefill_task)
        session, decode_response = await decode_request_task
        stream_generator = stream_decode_response(session, decode_response, request_id)
        response = await make_response(stream_generator)
        # st4=time.perf_counter()

    

        request_nums += 1
        
        # print(f"{(st4-st3)=},{(st3-st2)=},{(st2-st1)=},{(st1p5-st1)},{(st4-st1)=},request_id={request_id}")
        # print(f"zovlog:-----------> quit request")
        return response
    except Exception as e:
        print(e)
        pass
async def send_profile_cmd(req_data, profiler_cmd):
    global request_nums
    
    prefill_endpoint = prefill_instances[request_nums % len(prefill_instances)]
    decode_endpoint = decode_instances[request_nums % len(decode_instances)]
    
    headers = {
        "Authorization": f"Bearer {os.environ.get('OPENAI_API_KEY')}",
        "X-Request-Id": str(uuid.uuid4())
    }

    async with aiohttp.ClientSession(timeout=aiohttp.ClientTimeout(total=6 * 6000 * 6000)) as session:
        # 发送到prefill
        prefill_response = await session.post(
            f"http://0.0.0.0:20005/{profiler_cmd}_profile",
            json=req_data, headers=headers
        )
        
        # 发送到decode
        decode_response = await session.post(
            f"http://10.194.132.29:40005/{profiler_cmd}_profile", 
            json=req_data, headers=headers
        )
        
        # 安全处理prefill响应
        if prefill_response.status == 200:
            try:
                prefill_result = await prefill_response.json()
            except:
                prefill_text = await prefill_response.text()
                prefill_result = {"status": "success", "message": prefill_text}
        else:
            prefill_result = {"error": f"HTTP {prefill_response.status}", "text": await prefill_response.text()}
        
        # 安全处理decode响应
        if decode_response.status == 200:
            try:
                decode_result = await decode_response.json()
            except:
                decode_text = await decode_response.text()
                decode_result = {"status": "success", "message": decode_text}
        else:
            decode_result = {"error": f"HTTP {decode_response.status}", "text": await decode_response.text()}
        
        return {
            "prefill": prefill_result,
            "decode": decode_result
        }

@app.post("/start_profile")
async def start_profile():
    try:
        req_data =  await request.get_json(silent=True) or {}

        return await send_profile_cmd( req_data, "start")

    except Exception as e:
        import sys
        import traceback
        exc_info = sys.exc_info()
        print("Error occurred in disagg prefill proxy server"
              " - start_profile endpoint")
        print(e)
        print("".join(traceback.format_exception(*exc_info)))


@app.post("/stop_profile")
async def stop_profile():
    try:
        req_data =  await request.get_json(silent=True) or {}

        return await send_profile_cmd( req_data, "stop")

    except Exception as e:
        import sys
        import traceback
        exc_info = sys.exc_info()
        print("Error occurred in disagg prefill proxy server"
              " - stop_profile endpoint")
        print(e)
        print("".join(traceback.format_exception(*exc_info)))
if __name__ == '__main__':
    t = start_service_discovery("0.0.0.0", 36367)
    app.debug = True 
    app.config['BODY_TIMEOUT'] = 360000
    app.config['RESPONSE_TIMEOUT'] = 360000

    app.run(host="0.0.0.0", port=10001)
    t.join()






