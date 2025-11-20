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
logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)

import aiohttp
prefill_instances = []
decode_instances = [] 
request_nums = 0
app = Quart(__name__)

yield_chunk = set()

count=1
from itertools import count

# 使用无限计数器
counter = count(1)

def count_print(msg):
    current_count = next(counter)
    print(f"---mingzhilog[{current_count}] : {msg}")
    
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
                    prefill_instances.append(data)

            elif data["type"] == "register" and data['role'] == "D":
                if data['request_address'] not in decode_instances:
                    # decode_instances.append(data['request_address'])
                    decode_instances.append(data)
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

async def send_request_to_prefill(endpoint,req_data,request_id,remote_decode_instance,dip,dport):
    # print(f"zovlog:======> proxy {endpoint = }")
    req_data_copy = copy.deepcopy(req_data)
    
    # 本地做prefill,且decode只需要pull模式,所以prefill不需要在这里知晓远程decode任何信息
    req_data_copy['kv_transfer_params'] = {
        "do_remote_decode": True,
        "do_remote_prefill": False,
        "remote_handshake_port":remote_decode_instance['handshake_port'],
        "remote_notify_port":remote_decode_instance['notify_port'],
        "remote_engine_id": None,
        "remote_block_ids": None,
        "remote_host": dip,
        "remote_handshake_port": dport
    }
    req_data_copy["stream"] = False
    req_data_copy["max_tokens"] = 1
    if "max_completion_tokens" in req_data_copy:
        req_data_copy["max_completion_tokens"] = 1
    if "stream_options" in req_data_copy:
        del req_data_copy["stream_options"]
    # print(f"zovlog ========================== send response to prefill {req_data_copy}")
    async with aiohttp.ClientSession(timeout=aiohttp.ClientTimeout(total=6 * 60 * 60)) as session:
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
# to debug
async def send_request_to_decode(endpoint,req_data,request_id):
    # print(f"zovlog ========================== send response to decode {request_id}")
    async with aiohttp.ClientSession(timeout=aiohttp.ClientTimeout(total=6 * 60 * 60)) as session:
        headers = {
            "Authorization": f"Bearer {os.environ.get('OPENAI_API_KEY')}",
            "X-Request-Id": request_id
        }
        async with session.post(url=endpoint, json=req_data, headers=headers) as response:
            if response.status == 200:
                async for chunk_bytes in response.content.iter_chunked(10240):
                        if request_id not in yield_chunk:
                            # count_print(f"zovlog yield chunk for {request_id}") #128 
                            yield_chunk.add(request_id)
                            # b=0
                            # print(f"!!!{chunk_bytes.decode('utf-8')}")
                            # try:
                            #     print("!!!xxx")
                            #     print(json.loads(chunk_bytes.decode('utf-8'))['choices'][0]['text'])
                            # except Exception as e:
                            #     print(f"no text: {e}")
                        else:
                            logger.info("!!!pass yidle")
                            # print("pass yidle")
                        yield chunk_bytes
            else:
                raise RuntimeError("send_request_to_decode response.status != 200,response.statuus = ",response.status)

#user->proxy->prefill->proxy ->decode
@app.route("/v1/completions", methods=["POST"])
@app.route("/v1/chat/completions", methods=["POST"])
async def handle_request():
    # print(f"zovlog:-----------> enter request")
    global request_nums
    extract_ip_port = lambda url: re.search(r'//(\d{1,3}\.\d{1,3}\.\d{1,3}\.\d{1,3}):(\d+)', url).groups()
    req_data = await request.get_json()
    # print(f"req_data = {req_data}")
    request_id = str(uuid.uuid4())
    prefill_instance_endpoint = prefill_instances[request_nums % len(prefill_instances)]
    decode_instance_endpoint = decode_instances[request_nums % len(decode_instances)]
    dip, dport = extract_ip_port(decode_instance_endpoint['request_address'])
    response_json = await send_request_to_prefill(prefill_instance_endpoint['request_address'],req_data,request_id,decode_instance_endpoint,dip,dport)
    # 现在decode可以获取prefill的所有信息了
    ip, port = extract_ip_port(prefill_instance_endpoint['request_address'])
    response_json['kv_transfer_params']["do_remote_decode"] = False
    response_json['kv_transfer_params']["do_remote_prefill"] = True
    response_json['kv_transfer_params']["remote_host"] = ip
    response_json['kv_transfer_params']["remote_handshake_port"] = port # 似乎没用
    response_json['kv_transfer_params']["remote_handshake_port"] = prefill_instance_endpoint['handshake_port']
    # response_json['kv_transfer_params']["remote_handshake_port"] = prefill_instance_endpoint['handshake_port']
    response_json['kv_transfer_params']["remote_notify_port"] = prefill_instance_endpoint['notify_port']
    req_data['max_tokens'] -= 1
    # req_data['prompt'] += response_json['choices'][0]['text'] # comment out for ttft testing

    kv_transfer_params = response_json.get('kv_transfer_params', {})
    # print(f"zovlog:========> proxy kv_transfer_params = {kv_transfer_params}")
    if kv_transfer_params:
        req_data["kv_transfer_params"] = kv_transfer_params

    generator = send_request_to_decode(decode_instance_endpoint['request_address'],req_data,request_id)
    response = await make_response(generator)
    request_nums += 1
    # print(f"zovlog:-----------> quit request")
    return response
async def send_profile_cmd(req_data, profiler_cmd):
    global request_nums
    
    prefill_endpoint = prefill_instances[request_nums % len(prefill_instances)]
    decode_endpoint = decode_instances[request_nums % len(decode_instances)]
    
    headers = {
        "Authorization": f"Bearer {os.environ.get('OPENAI_API_KEY')}",
        "X-Request-Id": str(uuid.uuid4())
    }

    async with aiohttp.ClientSession(timeout=aiohttp.ClientTimeout(total=6 * 60 * 60)) as session:
        # 发送到prefill
        prefill_response = await session.post(
            f"http://0.0.0.0:20005/{profiler_cmd}_profile",
            json=req_data, headers=headers
        )
        
        # 发送到decode
        decode_response = await session.post(
            f"http://10.194.132.77:40005/{profiler_cmd}_profile", 
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
    app.config['BODY_TIMEOUT'] = 3600
    app.config['RESPONSE_TIMEOUT'] = 3600

    app.run(host="0.0.0.0", port=10001)
    t.join()





