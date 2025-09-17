#!/bin/bash
set -ex

LOG_FILE="logs/vllm_serve_decode_$(date +'%Y%m%d_%H-%M-%S').log"
# export GLOO_SOCKET_IFNAME=ens50f0
# export NCCL_SOCKET_IFNAME=ens50f0

export GLOO_SOCKET_IFNAME=ens14np0
export NCCL_SOCKET_IFNAME=ens14np0
export CUDA_VISIBLE_DEVICES=1,2
export HIP_VISIBLE_DEVICES=1,2
export NCCL_NCHANNELS_PER_NET_PEER=1
#export VLLM_LOGGING_CONFIG_PATH=log.conf.json
#export NCCL_DEBUG=INFO 
export VLLM_RINGBUFFER_WARNING_INTERVAL=500 
export VLLM_RPC_TIMEOUT=1800000 
export    NCCL_IB_DISABLE=1
# mkdir -p profiler
# export VLLM_TORCH_PROFILER_DIR=./profiler

VLLM_USE_V1=1 VLLM_ROCM_USE_AITER=1 VLLM_ENABLE_DSV3=0  SAFETENSORS_FAST_GPU=1   vllm  serve /nfs/data/Qwen3-0.6B  \
        -tp 2   \
        --block-size 16  \
        --max_seq_len_to_capture 6144 \
        --max-num-batched-tokens 6144 \
        --host 0.0.0.0 \
        --port 40005 \
        --trust-remote-code \
        --gpu-memory-utilization 0.6 \
        --disable-log-request \
        --served-model-name deepseek-ai/DeepSeek-R1 \
        --kv-transfer-config '{"kv_connector":"P2pNcclConnector","kv_role":"kv_consumer","kv_buffer_size":"4e10","kv_port":"28001","kv_connector_extra_config":{"proxy_ip":"10.194.132.29","proxy_port":"30001","http_port":"40005","send_type":"PUT","nccl_num_channels":"16"}}'
