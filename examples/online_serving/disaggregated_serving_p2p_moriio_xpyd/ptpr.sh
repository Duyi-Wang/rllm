



pkill -9 VLLM

ulimit -c 0

mkdir -p /mnt/m2m_nobackup/local_logs/

export MORIIO_CONNECTOR_READ_MODE=1

export VLLM_ENFORCE_EPLB=1 
export VLLM_ALL2ALL_BACKEND=mori 
export VLLM_USE_V1=1    
export VLLM_ROCM_USE_AITER=1 
export VLLM_ROCM_USE_AITER_LINEAR=1 
export VLLM_ROCM_USE_AITER_MLA=1 
export VLLM_ROCM_USE_AITER_MOE=1 
export VLLM_ROCM_USE_AITER_RMSNORM=1 
export VLLM_ROCM_USE_AITER_FUSION_SHARED_EXPERTS=0 
export VLLM_ROCM_USE_AITER_SAMPLING=1 
export VLLM_ENFORCE_EPLB=0
# MODEL_PATH=/mnt/m2m_nobackup/models/deepseek-ai/DeepSeek-V3
MODEL_PATH=/mnt/m2m_nobackup/models/deepseek-ai/DeepSeek-V3-5layer

vllm serve $MODEL_PATH        \
 -tp 8  \
 --port 20005     \
 --block-size 1          \
 --max-num-batched-tokens 4096         \
 --distributed-executor-backend mp         \
 --gpu_memory_utilization 0.85         \
 --max-model-len 4096         \
 --kv-cache-dtype fp8          \
 --enforce-eager \
 --no-enable-prefix-caching \
    --kv-transfer-config '{"kv_connector":"MoRIIOConnector","kv_role":"kv_producer","kv_port":"9711","kv_connector_extra_config":{"proxy_ip":"10.158.215.60","proxy_port":"30001","proxy_ping_port":"36367","local_ping_port":"61555","http_port":"20005","handshake_port":8405,"notify_port":61005}}' \
