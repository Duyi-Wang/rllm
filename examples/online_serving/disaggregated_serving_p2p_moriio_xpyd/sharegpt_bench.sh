# ISL=3000
# OSL=1000
# RATIO=0
PORT=10001
# PORT=40015

CONCURRENCY=32  # "8 16 32 64 128"
PROMPTS=32
      python3 /opt/vllm/benchmarks/benchmark_serving.py  \
        --dataset-name sharegpt \
        --dataset-path /nfs/users/mingzliu/ShareGPT_V3_unfiltered_cleaned_split.json \
        --model  deepseek-ai/DeepSeek-R1 \
        --num-prompt $PROMPTS \
        --base-url "http://127.0.0.1:$PORT" \
        --backend vllm \
        --max-concurrency $CONCURRENCY