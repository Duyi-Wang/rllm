#!/bin/bash

# LOG_FILE="logs/vllm_serve_prefill_$(date +'%Y%m%d_%H-%M-%S').log"

set -ex
# export GLOO_SOCKET_IFNAME=ens14np0
# export NCCL_SOCKET_IFNAME=ens14np0

export GLOO_SOCKET_IFNAME=eth0
export NCCL_SOCKET_IFNAME=eth0
export CUDA_VISIBLE_DEVICES=1,2
export HIP_VISIBLE_DEVICES=1,2
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
export VLLM_ENABLE_DSV3=0  
export SAFETENSORS_FAST_GPU=1   
export VLLM_TORCH_PROFILER_DIR=/nfs/users/mingzliu/vllm/examples/online_serving/disaggregated_serving_p2p_moriio_xpyd/write_0929
export CUDA_PROFILE_ACTIVITIES="cuda"
MODEL_PATH=/shared-inference/models_blog/Qwen3-0.6B
PROXY_IP="10.158.214.178"
# /apps/data/models/models--Qwen--Qwen3-0.6B/snapshots/e6de91484c29aa9480d55605af694f39b081c455
# {
 vllm serve  ${MODEL_PATH} \
    -tp 2 \
    --block-size 16 \
    --max-num-batched-tokens 6144 \
    --host 0.0.0.0 \
    --port 20005 \
    --enforce-eager \
    --trust-remote-code \
    --gpu-memory-utilization 0.6 \
    --disable-log-request \
    --served-model-name QWEN \
    --kv-transfer-config '{"kv_connector":"MoRIIOConnector","kv_role":"kv_producer","kv_port":"5041","kv_connector_extra_config":{"proxy_ip":"'"${PROXY_IP}"'","proxy_port":30001,"proxy_ping_port":"36367","local_ping_port":"7777","http_port":"20005","handshake_port":8657,"notify_port":9657}}'
           
           
#           "--kv-transfer-config={\"kv_connector\":\"MoRIIOConnector\",\"kv_role\":\"kv_producer\",\"kv_port\":\"21001\",\"kv_connector_extra_config\":{\"proxy_ip\":\"127.0.0.1\",\"proxy_port\":\"30001\",\"proxy_ping_port\":\"36367\",\"local_ping_port\":\"7777\",\"http_port\":\"20005\",\"handshake_port\":60000,\"notify_port\":49856}}"        ],

            #    "--kv-transfer-config={\"kv_connector\":\"MoRIIOConnector\",\"kv_role\":\"kv_producer\",\"kv_port\":\"21001\",\"kv_connector_extra_config\":{\"proxy_ip\":\"127.0.0.1\",\"proxy_port\":\"30001\",\"proxy_ping_port\":\"36367\",\"local_ping_port\":\"7777\",\"http_port\":\"20005\",\"handshake_port\":60000,\"notify_port\":49856}}"        ],

    # --kv-transfer-config '{"kv_connector":"MoRIIOConnector","kv_role":"kv_producer","kv_port":"21001","kv_connector_extra_config":{"proxy_ip":"10.194.132.29","proxy_port":"30001","proxy_ping_port":"36367","local_ping_port":"7777","http_port":"20005","handshake_port":60000,"notify_port":49856}}'
#  } 2>&1  & 