#!/bin/bash
set -ex
# export GLOO_SOCKET_IFNAME=ens50f0
# export NCCL_SOCKET_IFNAME=ens50f0
# export GLOO_SOCKET_IFNAME=ens14np0
# export NCCL_SOCKET_IFNAME=ens14np0

export GLOO_SOCKET_IFNAME=eth0
export NCCL_SOCKET_IFNAME=eth0
export CUDA_VISIBLE_DEVICES=4,5
export HIP_VISIBLE_DEVICES=4,5
export VLLM_USE_V1=1 
export VLLM_ROCM_USE_AITER=1 
export VLLM_ENABLE_DSV3=0  
export SAFETENSORS_FAST_GPU=1   
export VLLM_TORCH_PROFILER_DIR=/nfs/users/mingzliu/vllm/examples/online_serving/disaggregated_serving_p2p_moriio_xpyd/write_0929_1node
MODEL_PATH=/shared-inference/models_blog/Qwen3-0.6B

vllm serve $MODEL_PATH\
        -tp 1  \
    --block-size 16 \
    --max_seq_len_to_capture 6144 \
    --max-num-batched-tokens 6144 \
    --host 0.0.0.0 \
    --port 50005 \
    --enforce-eager \
    --trust-remote-code \
    --gpu-memory-utilization 0.6 \
    --disable-log-request \
    --served-model-name QWEN
    
    
    #--kv-transfer-config '{"kv_connector":"MoRIIOConnector","kv_role":"kv_consumer","kv_port":"25001","kv_connector_extra_config":{"proxy_ip":"10.235.192.56","proxy_port":"30001","http_port":"40005","local_ping_port":"32568","proxy_ping_port":"36367"}}'
