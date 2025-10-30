#!/bin/bash


export GLOO_SOCKET_IFNAME=eth0
export NCCL_SOCKET_IFNAME=eth0


export VLLM_ROCM_USE_AITER=1
export VLLM_ROCM_USE_AITER_MLA=1
export VLLM_ROCM_USE_AITER_MOE=1
export VLLM_LOGGING_LEVEL=INFO
export VLLM_USE_V1=1

export VLLM_TORCH_PROFILER_DIR="/home/duwang/profiling"
# export MODEL_PATH=/shared-inference/models_blog/Deepseek-r1-FP8-Dynamic
# export MODEL_PATH=/shared-inference/models_blog/DeepSeek-V3-5layer
export MODEL_PATH=/shared-inference/models_blog/DeepSeek-V3
PROXY_IP="10.158.215.60"
        # --no-enable-prefix-caching \

vllm serve ${MODEL_PATH} \
        -tp 8 \
        --served-model-name QWEN \
        --port 40005 \
        --block-size 1 \
        --distributed-executor-backend mp \
        --gpu_memory_utilization 0.9 \
        --max-model-len 8196 \
        --max_num_batched_token 32768 \
        --max-num-seqs 128 \
        --no-enable-prefix-caching \
        --cuda-graph-sizes 1 2 4 8 16 32 128 \
        --kv-cache-dtype fp8 \
        --trust-remote-code \
        --kv-transfer-config '{"kv_connector":"MoRIIOConnector","kv_role":"kv_consumer","kv_port":"32988","kv_connector_extra_config":{"proxy_ip":"'"${PROXY_IP}"'","proxy_port":"30001","http_port":"40005","local_ping_port":"32567","proxy_ping_port":"36367","handshake_port":61700,"notify_port":61800}}'


        # --no-enable-prefix-caching \