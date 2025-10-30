#!/bin/bash

# LOG_FILE="logs/vllm_serve_decode_$(date +'%Y%m%d_%H-%M-%S').log"
pkill -9 python
set -ex
# export GLOO_SOCKET_IFNAME=ens14np0
# export NCCL_SOCKET_IFNAME=ens14np0
export GLOO_SOCKET_IFNAME=eth0
export NCCL_SOCKET_IFNAME=eth0
# export CUDA_VISIBLE_DEVICES=6,7
# export HIP_VISIBLE_DEVICES=6,7
# export    NCCL_IB_DISABLE=1
# mkdir -p profiler
# export VLLM_TORCH_PROFILER_DIR=./profiler
#export VLLM_LOGGING_CONFIG_PATH=log.conf.json
#export NCCL_DEBUG=INFO 


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
# export VLLM_TORCH_PROFILER_WITH_STACK=0
# {

# MODEL_PATH=/nfs/DeepSeekV3tiny
MODEL_PATH=/mnt/m2m_nobackup/models/deepseek-ai/DeepSeek-V3
# MODEL_PATH=/shared-inference/models_blog/DeepSeek-V3-5layer
export VLLM_RPC_TIMEOUT=1800000
export VLLM_EXECUTE_MODEL_TIMEOUT_SECONDS=300000
mkdir -p /mnt/m2m_nobackup/local_logs/

# MODEL_PATH=/nfs/DeepSeek-V3

vllm serve $MODEL_PATH \
    -tp 8  \
    --block-size 1 \
    --no-enable-prefix-caching \
    --max-model-len 8192 \
    --max-num-seqs 128 \
    --cuda-graph-sizes 128\
    --trust-remote-code \
    --kv-cache-dtype fp8 \
    --host 0.0.0.0 \
    --port 40005 \
    --disable-log-request \
        --max-num-batched-tokens 32768 \
    --served-model-name QWEN \
    --kv-transfer-config '{"kv_connector":"MoRIIOConnector","kv_role":"kv_consumer","kv_port":"2988","kv_connector_extra_config":{"proxy_ip":"10.158.214.178","proxy_port":"30001","http_port":"40005","local_ping_port":"63005","proxy_ping_port":"36367","handshake_port":62005,"notify_port":61005}}' \
    2>&1 | tee /mnt/m2m_nobackup/local_logs/vllm_decode_server.log   

         #     --enforce-eager \

                #  "--kv-transfer-config={\"kv_connector\":\"MoRIIOConnector\",\"kv_role\":\"kv_consumer\",\"kv_port\":\"32988\",\"kv_connector_extra_config\":{\"proxy_ip\":\"127.0.0.1\",\"proxy_port\":\"30001\",\"http_port\":\"40005\",\"local_ping_port\":\"32567\",\"proxy_ping_port\":\"36367\",\"handshake_port\":60020,\"notify_port\":49657}}"      

        
        # --kv-transfer-config '{"kv_connector":"MoRIIOConnector","kv_role":"kv_consumer","kv_port":"32988","kv_connector_extra_config":{"proxy_ip":"10.194.132.29","proxy_port":"30001","http_port":"40005","local_ping_port":"32567","proxy_ping_port":"36367","handshake_port":60001,"notify_port":49857}}'
#       --kv-transfer-config '{"kv_connector":"MoRIIOConnector","kv_role":"kv_producer","kv_port":"21001","kv_connector_extra_config":{"proxy_ip":"10.194.132.29","proxy_port":"30001","proxy_ping_port":"36367","local_ping_port":"7777","http_port":"20005","handshake_port":60000,"notify_port":49856}}'

# } 2>&1 &
# notify_port
# for P instance: receive done req id from D instance use this port
# for D instance: send done req id to P instance use this port
#todo    1 merge 2notifyport 3 tp 4 queue write