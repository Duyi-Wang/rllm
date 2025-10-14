#!/bin/bash
set -ex
#--dataset-name random  --max-concurrency 4 \

python3 `pwd`/vllm/benchmarks/benchmark_serving.py \
   --port 30000 --host 10.235.192.55 \
   --model /apps/data/models/DSR1 \
   --dataset-name random  --profile \
   --random-input-len 16 --max-concurrency 1 \
   --random-output-len 32 \
   --num-prompts 1
