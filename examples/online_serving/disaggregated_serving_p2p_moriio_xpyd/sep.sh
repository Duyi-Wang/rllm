# export GLOO_SOCKET_IFNAME=ens14np0 
# export NCCL_SOCKET_IFNAME=ens14np0 
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
ulimit -c 0

MODEL_PATH=/mnt/m2m_nobackup/models/deepseek-ai/DeepSeek-V3

vllm serve $MODEL_PATH        \
 -tp 1  \
 -dp 8     \
 --enable-expert-parallel         \
 --port 10001     \
 --block-size 1          \
 --max-num-batched-tokens 32768         \
 --distributed-executor-backend mp         \
 --gpu_memory_utilization 0.85         \
 --max-model-len 32768         \
 --cuda-graph-sizes 1 256         \
 --kv-cache-dtype fp8          \
 --compilation-config '{"cuadgraph_mode": "FULL_DECODE_ONLY", "custom_ops": ["+quant_fp8"]}'         \
 --trust-remote-code  \
 --no-enable-prefix-caching \