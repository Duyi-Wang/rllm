# ISL=3000
# OSL=1000
# RATIO=0
PORT=10001
# PORT=40015

CONCURRENCY=1  # "8 16 32 64 128"
PROMPTS=32
     vllm bench serve  \
        --dataset-name sharegpt \
        --dataset-path /nfs/users/mingzliu/ShareGPT_V3_unfiltered_cleaned_split.json \
        --model  /nfs/data/Qwen3-0.6B  \
        --num-prompt $PROMPTS \
        --base-url "http://127.0.0.1:$PORT" \
        --backend vllm \
        --max-concurrency $CONCURRENCY