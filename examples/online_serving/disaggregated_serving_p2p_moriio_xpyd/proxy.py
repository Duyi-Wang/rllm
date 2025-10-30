# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import os
import socket
import threading
import time
import uuid
from typing import Any

import aiohttp
import msgpack
import zmq
from quart import Quart, make_response, request

count = 0
prefill_instances: dict[str, Any] = {}  # http_address: (zmq_address, stamp)
decode_instances: dict[str, Any] = {}  # http_address: (zmq_address, stamp)

prefill_cv = threading.Condition()
decode_cv = threading.Condition()

DEFAULT_PING_SECONDS = 5

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
            remote_addr,msg = router_socket.recv_multipart()
            data = msgpack.loads(msg)
            if data['type'] == "HELLO":
                pass
            elif data['type'] == "register" and data['role'] == "P":
                if data['request_address'] not in prefill_instances:
                    prefill_instances[data['request_address']] = data['request_address']
            elif data["type"] == "register" and data['role'] == "D":
                if data['request_address'] not in decode_instances:
                    decode_instances[data['request_address']] = data['request_address']
            print(f"zovlog:====> recv {data},remote_addr={remote_addr},{prefill_instances = },{decode_instances = }")

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


AIOHTTP_TIMEOUT = aiohttp.ClientTimeout(total=6 * 60 * 60)

app = Quart(__name__)


def random_uuid() -> str:
    return str(uuid.uuid4().hex)


async def forward_request(url, data):
    async with aiohttp.ClientSession(timeout=AIOHTTP_TIMEOUT) as session:
        headers = {
            "Authorization": f"Bearer {os.environ.get('OPENAI_API_KEY')}"
            # "X-Request-Id": "fasdfasdfasdf",
        }
        print(f"zovlog:====>ready to post:{url=},{data = },{headers = }")
        async with session.post(url=url, json=data, headers=headers) as response:
            if response.status == 200:
                if True:
                    async for chunk_bytes in response.content.iter_chunked(1024):
                        yield chunk_bytes
                else:
                    content = await response.read()
                    yield content


@app.route("/v1/completions", methods=["POST"])
@app.route("/v1/chat/completions", methods=["POST"])
async def handle_request():
    try:
        original_request_data = await request.get_json()

        prefill_request = original_request_data.copy()
        # change max_tokens = 1 to let it only do prefill
        prefill_request["max_tokens"] = 1
        if "max_completion_tokens" in prefill_request:
            prefill_request["max_completion_tokens"] = 1

        global count
        global prefill_instances
        global prefill_cv
        with prefill_cv:
            prefill_list = list(prefill_instances.items())
            prefill_addr = prefill_list[count % len(prefill_list)][1]

        global decode_instances
        global decode_cv
        with decode_cv:
            decode_list = list(decode_instances.items())

            decode_addr = decode_list[count % len(decode_list)][1]
        count += 1

        # # finish prefill
        # print(f"{prefill_instances = },{prefill_addr = }")
        # prefill_generator = forward_request(
        #     f"http://{prefill_addr}/v1/completions", prefill_request
        # )
        # prefill_result = make_response(prefill_generator)
        # # print(f"{prefill_result = }")
        # # return decode

        prefill_response_data = []
        async for chunk in forward_request(
            f"http://{prefill_addr}/v1/completions", prefill_request
        ):
            prefill_response_data.append(chunk)
            print(f"zovlog:====< chunk.decode() = {chunk.decode()}")
        
        full_prefill_response = b"".join(prefill_response_data)
        try:
            import json
            prefill_json = json.loads(full_prefill_response)
            print(f"zovlog:=====> {prefill_json = }")
        except:
            pass



        generator = forward_request(
            f"http://{decode_addr}/v1/completions", original_request_data
        )
        response = await make_response(generator)
        response.timeout = None
        # print(f"{response = }")
        return response
        # return None

    except Exception as e:
        import sys
        import traceback

        exc_info = sys.exc_info()
        print("Error occurred in disagg prefill proxy server")
        print(e)
        print("".join(traceback.format_exception(*exc_info)))


if __name__ == "__main__":
    t = start_service_discovery("0.0.0.0", 36367)
    app.run(host="0.0.0.0", port=10001)
    t.join()
