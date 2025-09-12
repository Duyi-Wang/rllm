ISL=3000
#1000有问题,300没问题
OSL=1000
RATIO=0
PORT=10001
# PORT=40015

CONCURRENCY=32  # "8 16 32 64 128"
PROMPTS=128
      python3 /opt/vllm/benchmarks/benchmark_serving.py  \
        --dataset-name random \
        --model  deepseek-ai/DeepSeek-R1 \
        --random-input-len $ISL \
        --random-output-len $OSL \
        --num-prompt $PROMPTS \
        --random-range-ratio $RATIO \
        --base-url "http://127.0.0.1:$PORT" \
        --backend vllm \
        --max-concurrency $CONCURRENCY
        # --dataset-path /nfs/users/mingzliu/ShareGPT_V3_unfiltered_cleaned_split.json \
