#!/bin/bash

# LOG_FILE="logs/vllm_serve_prefill_$(date +'%Y%m%d_%H-%M-%S').log"

set -ex
# export GLOO_SOCKET_IFNAME=ens14np0
# export NCCL_SOCKET_IFNAME=ens14np0
export GLOO_SOCKET_IFNAME=eth0
export NCCL_SOCKET_IFNAME=eth0
export CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7
export HIP_VISIBLE_DEVICES=0,1,2,3,4,5,6,7
#export VLLM_LOGGING_CONFIG_PATH=log.conf.json
#export NCCL_DEBUG=INFO 

# export    NCCL_IB_DISABLE=1
# mkdir -p profiler
# export VLLM_TORCH_PROFILER_DIR=./profiler


# export NCCL_NCHANNELS_PER_NET_PEER=1
# export VLLM_RINGBUFFER_WARNING_INTERVAL=500 
# export VLLM_RPC_TIMEOUT=1800000 
# export IBV_DRIVERS_LOG_LEVEL=4

export VLLM_ROCM_USE_AITER_MLA=1
export VLLM_ROCM_USE_AITER_MOE=1
export VLLM_LOGGING_LEVEL=INFO

export VLLM_USE_V1=1 
export VLLM_ROCM_USE_AITER=1 
export SAFETENSORS_FAST_GPU=1   
# export VLLM_TORCH_PROFILER_DIR=/nfs/users/mingzliu/vllm/examples/online_serving/disaggregated_serving_p2p_moriio_xpyd/write_0929
export CUDA_PROFILE_ACTIVITIES="cuda"
# MODEL_PATH=/nfs/DeepSeekV3tiny
# MODEL_PATH=/shared-inference/models_blog/DeepSeek-V3-5layer
# MODEL_PATH=/shared-inference/models_blog/DeepSeek-V3
MODEL_PATH=/mnt/m2m_nobackup/models/deepseek-ai/DeepSeek-V3
export VLLM_RPC_TIMEOUT=1800000
# MODEL_PATH=/nfs/DeepSeek-V3
# /apps/data/models/models--Qwen--Qwen3-0.6B/snapshots/e6de91484c29aa9480d55605af694f39b081c455
# {
mkdir -p /mnt/m2m_nobackup/local_logs/
#32768   60999 zmq
 vllm serve $MODEL_PATH   \
    -tp 8 \
    --block-size 1 \
    --max-num-batched-tokens 8192 \
    --max-model-len 8192 \
    --max-num-seqs 2048 \
    --trust-remote-code \
     --host 0.0.0.0 \
    --port 20005 \
    --enforce-eager \
    --disable-log-request \
    --kv-cache-dtype fp8 \
    --served-model-name QWEN \
    --kv-transfer-config '{"kv_connector":"MoRIIOConnector","kv_role":"kv_producer","kv_port":"62001","kv_connector_extra_config":{"proxy_ip":"10.158.214.178","proxy_port":"30001","proxy_ping_port":"36367","local_ping_port":"61555","http_port":"20005","handshake_port":63005,"notify_port":61005}}' \
    2>&1 | tee /mnt/m2m_nobackup/local_logs/vllm_prefill_server.log   
#32768   60999
#"--kv-transfer-config={\"kv_connector\":\"MoRIIOConnector\",\"kv_role\":\"kv_producer\",\"kv_port\":\"21001\",\"kv_connector_extra_config\":{\"proxy_ip\":\"127.0.0.1\",\"proxy_port\":\"30001\",\"proxy_ping_port\":\"36367\",\"local_ping_port\":\"7777\",\"http_port\":\"20005\",\"handshake_port\":60000,\"notify_port\":49856}}"        ],

# "--kv-transfer-config={\"kv_connector\":\"MoRIIOConnector\",\"kv_role\":\"kv_producer\",\"kv_port\":\"21001\",\"kv_connector_extra_config\":{\"proxy_ip\":\"127.0.0.1\",\"proxy_port\":\"30001\",\"proxy_ping_port\":\"36367\",\"local_ping_port\":\"7777\",\"http_port\":\"20005\",\"handshake_port\":60000,\"notify_port\":49856}}"        ],

# --kv-transfer-config '{"kv_connector":"MoRIIOConnector","kv_role":"kv_producer","kv_port":"21001","kv_connector_extra_config":{"proxy_ip":"10.194.132.29","proxy_port":"30001","proxy_ping_port":"36367","local_ping_port":"7777","http_port":"20005","handshake_port":60000,"notify_port":49856}}'
#  } 2>&1  & 