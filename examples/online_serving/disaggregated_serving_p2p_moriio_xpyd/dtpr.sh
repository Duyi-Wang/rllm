pkill -9 vllm




pkill -9 vllm

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
 --port 40005     \
 --block-size 1          \
 --no-enable-prefix-caching \
 --max-num-batched-tokens 4096         \
 --distributed-executor-backend mp         \
 --gpu_memory_utilization 0.85         \
 --max-model-len 8200         \
 --cuda-graph-sizes 1 256         \
 --kv-cache-dtype fp8          \
 --compilation-config '{"cuadgraph_mode": "FULL_DECODE_ONLY", "custom_ops": ["+quant_fp8"]}'         \
 --trust-remote-code \
  --max_num_seqs 256\
    --kv-transfer-config '{"kv_connector":"MoRIIOConnector","kv_role":"kv_consumer","kv_port":"6301","kv_connector_extra_config":{"proxy_ip":"10.158.215.60","proxy_port":"30001","http_port":"40005","local_ping_port":"4583","proxy_ping_port":"36367","handshake_port":7305,"notify_port":61005}}' \
    2>&1 | tee /mnt/m2m_nobackup/local_logs/vllm_prefill_server.log   