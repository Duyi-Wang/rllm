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
AIOHTTP_TIMEOUT = aiohttp.ClientTimeout(total=6 * 60 * 60)

url='http://10.235.192.59:40005/v1/completions'
data = {'model': 'deepseek-ai/DeepSeek-R1', 'prompt': 'Hi,how are you?', 'max_tokens': 10, 'temperature': 0, 'top_k': 1}
headers = {'Authorization': 'Bearer None'}
def forward_request():
    with aiohttp.ClientSession(timeout=AIOHTTP_TIMEOUT) as session:
        headers = {
            "Authorization": f"Bearer {os.environ.get('OPENAI_API_KEY')}"
        }
        # print(f"zovlog:====>ready to post:{url=},{data = },{headers = }")
        with session.post(url=url, json=data, headers=headers) as response:
            if response.status == 200:
                if True:
                    for chunk_bytes in response.content.iter_chunked(1024):
                        yield chunk_bytes
                else:
                    content = await response.read()
                    yield content