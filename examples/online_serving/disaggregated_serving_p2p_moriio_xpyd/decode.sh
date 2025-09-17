#!/bin/bash

# LOG_FILE="logs/vllm_serve_decode_$(date +'%Y%m%d_%H-%M-%S').log"
set -ex
export GLOO_SOCKET_IFNAME=ens14np0
export NCCL_SOCKET_IFNAME=ens14np0

export CUDA_VISIBLE_DEVICES=5,6
export HIP_VISIBLE_DEVICES=5,6
export NCCL_NCHANNELS_PER_NET_PEER=1
#export VLLM_LOGGING_CONFIG_PATH=log.conf.json
#export NCCL_DEBUG=INFO 
export VLLM_RINGBUFFER_WARNING_INTERVAL=500 
export VLLM_RPC_TIMEOUT=1800000 
# export    NCCL_IB_DISABLE=1
# mkdir -p profiler
# export VLLM_TORCH_PROFILER_DIR=./profiler

export VLLM_USE_V1=1 
export VLLM_ROCM_USE_AITER=1 
export VLLM_ENABLE_DSV3=0  
export SAFETENSORS_FAST_GPU=1   
export IBV_DRIVERS_LOG_LEVEL=4
# {
vllm serve /nfs/data/Qwen3-32B \
        -tp 2   \
        --block-size 16  \
        --max_seq_len_to_capture 6144 \
        --max-num-batched-tokens 6144 \
            --enforce-eager \
        --host 0.0.0.0 \
        --port 40005 \
        --trust-remote-code \
        --gpu-memory-utilization 0.6\
        --disable-log-request \
        --served-model-name deepseek-ai/DeepSeek-R1 \
        --kv-transfer-config '{"kv_connector":"MoRIIOConnector","kv_role":"kv_consumer","kv_port":"32988","kv_connector_extra_config":{"proxy_ip":"10.194.132.29","proxy_port":"30001","http_port":"40005","local_ping_port":"32567","proxy_ping_port":"36367","handshake_port":60020,"notify_port":49657}}'
        
                #  "--kv-transfer-config={\"kv_connector\":\"MoRIIOConnector\",\"kv_role\":\"kv_consumer\",\"kv_port\":\"32988\",\"kv_connector_extra_config\":{\"proxy_ip\":\"127.0.0.1\",\"proxy_port\":\"30001\",\"http_port\":\"40005\",\"local_ping_port\":\"32567\",\"proxy_ping_port\":\"36367\",\"handshake_port\":60020,\"notify_port\":49657}}"      

        
        # --kv-transfer-config '{"kv_connector":"MoRIIOConnector","kv_role":"kv_consumer","kv_port":"32988","kv_connector_extra_config":{"proxy_ip":"10.194.132.29","proxy_port":"30001","http_port":"40005","local_ping_port":"32567","proxy_ping_port":"36367","handshake_port":60001,"notify_port":49857}}'
#       --kv-transfer-config '{"kv_connector":"MoRIIOConnector","kv_role":"kv_producer","kv_port":"21001","kv_connector_extra_config":{"proxy_ip":"10.194.132.29","proxy_port":"30001","proxy_ping_port":"36367","local_ping_port":"7777","http_port":"20005","handshake_port":60000,"notify_port":49856}}'

# } 2>&1 &
# notify_port
# for P instance: receive done req id from D instance use this port
# for D instance: send done req id to P instance use this port