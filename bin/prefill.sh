#!/bin/bash
export VLLM_TORCH_PROFILER_DIR="./profile"
rm -rf ${VLLM_TORCH_PROFILER_DIR}
mkdir -p ${VLLM_TORCH_PROFILER_DIR}

#VLLM_ROCM_USE_AITER_FP8BMM=1 VLLM_ROCM_USE_AITER=1 VLLM_ROCM_USE_AITER_MLA=1 VLLM_ROCM_USE_AITER_MOE=1 VLLM_USE_V1=1 vllm serve /apps/data/models/DSR1 -tp 8 -dp 1  --port 30000 --block-size 1 --max-num-batched-tokens 32768 --no-enable-prefix-caching --speculative-config '{"method":"deepseek_mtp","num_speculative_tokens":1}' 2>&1 | tee mtp2.log
VLLM_ROCM_USE_AITER_FP8BMM=1 VLLM_ROCM_USE_AITER=1 VLLM_ROCM_USE_AITER_MLA=1 VLLM_ROCM_USE_AITER_MOE=1 VLLM_USE_V1=1 vllm serve /apps/data/models/DSR1 -tp 8 -dp 1  --port 30000 --block-size 1 --max-num-batched-tokens 32768 --no-enable-prefix-caching  2>&1 | tee decode.log
