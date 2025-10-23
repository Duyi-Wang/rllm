set -ex
BIN=`dirname ${0}`
BIN=`cd ${BIN}; pwd`
 
P=${BIN}/pd_profile
rm -rf ${P}
mkdir -p ${P}
 
export HIP_VISIBLE_DEVICES=0,1,2,3,4,5,6,7
#export NCCL_UCX_TLS=rc #rocm_ipc
#export v_LOG=10
#export NCCL_SOCKET_IFNAME=ens50f0
#export LD_LIBRARY_PATH=/usr/local/lib/python3.10/dist-packages/.nixl.mesonpy.libs/plugins/:${LD_LIBRARY_PATH}
# export -n UCX_NET_DEVICES
# export -n UCX_TLS
# export UCX_NET_DEVICES=bnxt_re_bond0:1 #,bnxt_re_bond1:1,bnxt_re_bond2:1,bnxt_re_bond3:1,bnxt_re_bond4:1,bnxt_re_bond5:1,bnxt_re_bond6:1,bnxt_re_bond7:1
 
#export VLLM_SERVER_DEV_MODE=1
export VLLM_NIXL_SIDE_CHANNEL_PORT=44440
export VLLM_NIXL_SIDE_CHANNEL_HOST=10.224.3.45
#export UCX_LOG_LEVEL=debug
#export VLLM_LOGGING_LEVEL=DEBUG
export VLLM_WORKER_MULTIPROC_METHOD=spawn
export VLLM_ENABLE_V1_MULTIPROCESSING=0
# MODEL_PATH=/shared-inference/models_blog/Qwen3-0.6B
# MODEL_PATH=/shared-inference/models_blog/DeepSeek-V3-5layer
MODEL_PATH=/shared-inference/models_blog/DeepSeek-V3
# M=/apps/data/models/DSV3
# M=/apps/data/models/DSV3_MINI
export LD_LIBRARY_PATH=/opt/rixl/lib/x86_64-linux-gnu:/opt/ucx/lib:/opt/rocm/lib:/usr/local/lib:$LD_LIBRARY_PATH

export VLLM_TORCH_PROFILER_DIR=${P} 
export CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 
export VLLM_RPC_TIMEOUT=1800000 
export SAFETENSORS_FAST_GPU=1 
export VLLM_USE_V1=1 
export VLLM_ROCM_USE_AITER=1 
export VLLM_ENABLE_DSV3=1 
export UCX_TLS=rc,rocm 

 vllm serve $MODEL_PATH   \
    -tp 8 \
    --block-size 1 \
    --max-num-batched-tokens 8192 \
    --no-enable-prefix-caching \
    --max-model-len 8192 \
    --max-num-seqs 2048 \
    --trust-remote-code \
     --host 0.0.0.0 \
    --port 20005 \
    --enforce-eager \
    --disable-log-request \
    --served-model-name QWEN \
    --kv-transfer-config '{"kv_connector":"NixlConnector","kv_role":"kv_both"}'
# vllm serve ${M} \
#   --port 8200 \
#   --enforce-eager --disable-log-requests \
#   --disable-log-requests --no-enable-prefix-caching --block-size 16 \
#   --tensor-parallel-size 1 \
#   --gpu-memory-utilization 0.7 \
#   --trust-remote-code \
#   --kv-transfer-config '{"kv_connector":"NixlConnector","kv_role":"kv_both"}' 2>&1 | tee decode.log